"""Frozen card-content interventions and simulator-only immediate-play scoring.

Scores are computed after choices, never supplied to the encoder or decoder.
Permutations are observation interventions, not edits to simulator internals.
"""

from __future__ import annotations

from itertools import combinations
import hashlib

import numpy as np

from flylatro.env.upstream_contract import (
    HAND_MAX, MAX_CARD_PICKS, empty_action_batch, validate_strict_action_batch,
)
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.seeds import derive_seed


VERSION = "frozen-card-selection-diagnostic-v1"


def permutations(hand_len: int, count: int, seed: int) -> list[np.ndarray]:
    """Unique derangements of occupied slots; padding never moves."""
    if not 2 <= hand_len <= HAND_MAX or count < 1:
        raise ValueError("need at least two occupied cards and positive permutations")
    rng = np.random.default_rng(seed)
    result = []
    seen = set()
    for _ in range(max(256, count * 128)):
        order = rng.permutation(hand_len)
        if np.any(order == np.arange(hand_len)) or tuple(order) in seen:
            continue
        seen.add(tuple(order))
        full = np.arange(HAND_MAX)
        full[:hand_len] = order
        result.append(full)
        if len(result) == count:
            break
    return result


def permute_hand(observations, masks, order):
    """order[new_slot] = original_slot, for every card feature and legal bit."""
    order = np.asarray(order)
    size = int(observations["hand_len"][0])
    if (order.shape != (HAND_MAX,) or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order[:size]), np.arange(size))
            or not np.array_equal(order[size:], np.arange(size, HAND_MAX))):
        raise ValueError("permutation must cover occupied slots and preserve padding")
    obs = {key: value.copy() for key, value in observations.items()}
    legal = {key: value.copy() for key, value in masks.items()}
    obs["hand"] = obs["hand"][:, order].copy()
    legal["card_select_mask"] = legal["card_select_mask"][:, order].copy()
    return obs, legal


def play_action(cards):
    action = empty_action_batch(1)
    action["action_type"][0] = 0
    action["n_cards"][0] = len(cards)
    action["cards"][0, :len(cards)] = cards
    return action


def score_plays(env, snapshot, observations, masks):
    """Enumerate canonical subsets in the real simulator, restoring each trial.

    Card order is canonical original-slot order, including decoded selections.
    This measures card membership/count, not within-play ordering strategy.
    """
    original = env.snapshot(0)
    scores = {}
    try:
        env.restore(0, snapshot)
        restored_obs, restored_masks = env.observe()
        if any(not np.array_equal(value, restored_obs[key]) for key, value in observations.items()):
            raise ValueError("snapshot does not reproduce corpus observations")
        if any(not np.array_equal(value, restored_masks[key]) for key, value in masks.items()):
            raise ValueError("snapshot does not reproduce constrained corpus masks")
        available = np.flatnonzero(masks["card_select_mask"][0]).tolist()
        if not masks["action_type_mask"][0, 0] or len(available) < 2:
            raise ValueError("card diagnostic requires a competing legal PLAY state")
        for count in range(1, min(MAX_CARD_PICKS, len(available)) + 1):
            for cards in combinations(available, count):
                env.restore(0, snapshot)
                action = play_action(cards)
                validate_strict_action_batch(action, masks)
                value = float(env.step(action).rewards[0])
                if not np.isfinite(value):
                    raise ValueError("nonfinite simulator play reward")
                scores[cards] = value
    finally:
        env.restore(0, original)
    return scores


def plastic_hashes(state):
    """All persistent and fast plastic state, including initial efficacies."""
    return {key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in state.state_dict().items()}


def _cards(action):
    return tuple(int(x) for x in action["cards"][0, :int(action["n_cards"][0])])


def _jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right)


def _mean(values):
    return float(np.mean(values)) if values else None


def _quality(cards, scores):
    value = scores[tuple(sorted(cards))]
    best, worst = max(scores.values()), min(scores.values())
    return {"immediate_reward": value, "best_available_immediate_reward": best,
            "regret": best - value, "informative": best - worst > 1e-9,
            "normalized_regret": (best - value) / (best - worst) if best - worst > 1e-9 else None,
            "best_subset_selected": abs(best - value) < 1e-8}


def diagnose_learner(agent, learner, cases, *, repeats, permutation_count, seed, progress=None):
    """Direct frozen processor/decoder calls; never record credit or learn.

    A PLAY-only mask isolates the existing card heads after neural processing.
    The unconstrained greedy action is also recorded; no policy is deployed.
    """
    if repeats < 1 or permutation_count < 1 or not cases:
        raise ValueError("need cases and positive repeat/permutation counts")
    state = agent.plasticity.state
    if not 0 <= learner < state.learners:
        raise ValueError("learner index outside checkpoint")
    before = plastic_hashes(state)
    efficacy_hashes = {label: hashlib.sha256(state_numpy(value[learner:learner + 1]).tobytes()).hexdigest()
                      for label, value in (("initial", state.initial_efficacy), ("trained", state.efficacy))}
    rows = []
    for case in cases:
        index, obs, masks = case["index"], case["observations"], case["masks"]
        orders = permutations(int(obs["hand_len"][0]), permutation_count,
                              derive_seed("card-permutation", seed, index))
        for repeat in range(repeats):
            neural_seed = derive_seed("card-diagnostic", seed, index, repeat)
            for label, efficacy in (("initial", state.initial_efficacy), ("trained", state.efficacy)):
                def choose(observations, legal):
                    neural = agent.processor.process(observations, fly_seeds=(neural_seed,),
                        efficacy=efficacy[learner:learner + 1])
                    output = np.asarray(neural.output_activity).copy()
                    if not np.isfinite(output).all():
                        raise ValueError("nonfinite neural output")
                    natural = agent.motor.decode(output, legal, deterministic=True)
                    play_masks = {key: value.copy() for key, value in legal.items()}
                    play_masks["action_type_mask"][:] = False
                    play_masks["action_type_mask"][:, 0] = True
                    action = agent.motor.decode(output, play_masks, deterministic=True)
                    validate_strict_action_batch(action, play_masks)
                    return output, action, int(natural["action_type"][0]), agent.motor.head_scores(output)["card"][0].copy()

                original, action, natural, original_scores = choose(obs, masks)
                duplicate, duplicate_action, duplicate_natural, _ = choose(obs, masks)
                if (not np.array_equal(original, duplicate) or natural != duplicate_natural
                        or any(not np.array_equal(action[key], duplicate_action[key]) for key in action)):
                    raise ValueError("identical input/seed/weights failed exact neural/action repeat")
                base_cards = _cards(action)
                row = {"state_index": index, "repeat": repeat, "neural_seed": neural_seed,
                       "weights": label, "natural_action_type": natural,
                       "cards": list(base_cards), "card_scores": original_scores.tolist(),
                       "quality": _quality(base_cards, case["scores"]), "permutations": []}
                for order in orders:
                    altered, legal = permute_hand(obs, masks, order)
                    _, selected, natural_type, card_scores = choose(altered, legal)
                    slots = _cards(selected)
                    mapped = tuple(int(order[slot]) for slot in slots)
                    n_legal = int(masks["card_select_mask"].sum())
                    informative = len(slots) == len(base_cards) and 0 < len(base_cards) < n_legal
                    # Physical-card identity and visible feature multiset are separate:
                    # two cards may have identical observable features.
                    features = lambda cards: sorted(obs["hand"][0, card].tobytes() for card in cards)
                    available = masks["card_select_mask"][0]
                    inverse = np.argsort(order)
                    row["permutations"].append({
                        "new_slot_to_original_slot": order.tolist(), "cards_in_new_slots": list(slots),
                        "mapped_original_cards": list(mapped), "natural_action_type": natural_type,
                        "card_scores": card_scores.tolist(), "same_count": len(slots) == len(base_cards),
                        "membership_comparison_informative": informative,
                        "same_card_set": set(base_cards) == set(mapped),
                        "same_feature_multiset": features(base_cards) == features(mapped),
                        "same_slot_set": set(base_cards) == set(slots),
                        "card_jaccard": _jaccard(base_cards, mapped),
                        "slot_jaccard": _jaccard(base_cards, slots),
                        "slot_aligned_score_mae": float(np.abs(card_scores[available] - original_scores[available]).mean()),
                        "card_aligned_score_mae": float(np.abs(card_scores[inverse][available] - original_scores[available]).mean()),
                        "quality": _quality(mapped, case["scores"]),
                    })
                rows.append(row)
        if progress:
            progress(index)
    after = plastic_hashes(state)
    if before != after:
        raise ValueError("frozen diagnostic changed plastic state")
    summaries = {}
    for label in ("initial", "trained"):
        selected = [row for row in rows if row["weights"] == label]
        informative = [row for row in selected if row["quality"]["informative"]]
        trials = [trial for row in selected for trial in row["permutations"]]
        membership = [trial for trial in trials if trial["membership_comparison_informative"]]
        summaries[label] = {
            "original_trials": len(selected), "informative_quality_trials": len(informative),
            "mean_immediate_reward": _mean([r["quality"]["immediate_reward"] for r in selected]),
            "mean_regret": _mean([r["quality"]["regret"] for r in selected]),
            "mean_normalized_regret_informative": _mean([r["quality"]["normalized_regret"] for r in informative]),
            "best_subset_fraction_informative": _mean([r["quality"]["best_subset_selected"] for r in informative]),
            "mean_selected_count": _mean([len(r["cards"]) for r in selected]),
            "natural_play_fraction": _mean([r["natural_action_type"] == 0 for r in selected]),
            "permutation_trials": len(trials), "same_count_fraction": _mean([t["same_count"] for t in trials]),
            "membership_trials": len(membership),
            "same_card_set_fraction": _mean([t["same_card_set"] for t in membership]),
            "same_feature_multiset_fraction": _mean([t["same_feature_multiset"] for t in membership]),
            "same_slot_set_fraction": _mean([t["same_slot_set"] for t in membership]),
            "mean_card_jaccard": _mean([t["card_jaccard"] for t in membership]),
            "mean_slot_jaccard": _mean([t["slot_jaccard"] for t in membership]),
            "mean_slot_aligned_score_mae": _mean([t["slot_aligned_score_mae"] for t in trials]),
            "mean_card_aligned_score_mae": _mean([t["card_aligned_score_mae"] for t in trials]),
        }
    paired = []
    for index in sorted({row["state_index"] for row in rows}):
        state_rows = [row for row in rows if row["state_index"] == index]
        delta = [_mean([r["quality"]["immediate_reward"] for r in state_rows if r["weights"] == label])
                 for label in ("initial", "trained")]
        paired.append({"state_index": index, "mean_trained_minus_initial_reward": delta[1] - delta[0]})
    return {"learner_index": learner, "efficacy_sha256": efficacy_hashes,
            "plastic_state_hashes_before": before, "plastic_state_hashes_after": after,
            "plastic_state_unchanged": True, "exact_repeat_checks": len(rows),
            "summary": summaries, "paired_state_changes": paired,
            "mean_paired_reward_change": _mean([p["mean_trained_minus_initial_reward"] for p in paired]),
            "rows": rows}
