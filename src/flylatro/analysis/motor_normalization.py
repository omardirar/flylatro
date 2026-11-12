"""Reward-free fixed-pool normalization and held-out cached-response analysis."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from itertools import combinations

import numpy as np

from flylatro.analysis.card_information import (
    CONDITIONS, change_card, contrast_scores, response_metrics, summarize, visible_card,
)
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorNormalization
from flylatro.learning.config import payload_sha256


VERSION = "fixed-pool-250ms-normalization-v1"
CRITERION = (
    "in a strict majority of trained choice learners, held-out dominant top-slot fraction "
    "decreases to at most 0.5, and both rank/suit card-contrast effect/noise power ratios "
    "are positive and exceed the old normalization; mean ratios improve for both edits; "
    "all source/frozen-state/disjointness/health checks pass"
)


def membership_hash(mapping):
    return payload_sha256({"mode": mapping.mode, "roots": mapping.output_root_ids.tolist(),
                           "pools": mapping.pool_indices, "routing": mapping.routing.sha256})


def assert_only_normalization_changed(source, candidate):
    if membership_hash(source) != membership_hash(candidate):
        raise ValueError("normalization experiment changed motor pool membership or routing")
    for field in ("version", "selection_method", "candidate_set_sha256", "exploration_epsilon",
                  "exploration_temperature", "exploration_seed"):
        if getattr(source, field) != getattr(candidate, field):
            raise ValueError("normalization experiment changed " + field)


def fit_fixed_normalization(source, pool_rates, *, provenance):
    """Accept only neural rates, with no actions, rewards or quality labels."""
    values = np.asarray(pool_rates, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != len(source.pool_indices) or len(values) < 4
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("fit needs finite nonnegative sample-by-fixed-pool rates")
    floor = source.normalization.minimum_scale_hz if source.normalization else .5
    normalization = MotorNormalization.from_pool_activity(values, minimum_scale_hz=floor)
    metadata = {"version": VERSION, "reward_used": False, "hand_quality_used": False,
                "unit_initial_weights_only": True, "pool_membership_sha256": membership_hash(source),
                "source_motor_sha256": source.sha256, "provenance": provenance,
                "normalization_sha256": normalization.sha256,
                "qualification": "normalization-only diagnostic; no full-context or gameplay qualification"}
    candidate = replace(source, normalization=normalization,
                        calibration_sha256=payload_sha256(metadata), calibration_stats=metadata)
    assert_only_normalization_changed(source, candidate)
    return candidate


def heldout_cases(calibration, test, *, states):
    """One state per unseen run; test the first, sixth and last eligible slots."""
    if states < 1 or not calibration.records_run_provenance or not test.records_run_provenance:
        raise ValueError("positive state count and complete run provenance required")
    if set(calibration.environment_seeds) & set(test.environment_seeds):
        raise ValueError("calibration/test root seeds overlap")
    fit_hashes, fit_runs = set(calibration.state_hashes), set(calibration.source_run_seed)
    candidates, seen_runs = [], set()
    for index in range(len(test)):
        if test.state_hashes[index] in fit_hashes or test.source_run_seed[index] in fit_runs:
            continue
        if test.source_run_seed[index] in seen_runs or not test.masks["action_type_mask"][index, 0]:
            continue
        obs, masks = test.row(index)
        slots = [slot for slot in range(int(obs["hand_len"][0]))
                 if masks["card_select_mask"][0, slot] and visible_card(obs["hand"][0, slot])]
        if 5 not in slots or len(slots) < 6:
            continue
        seen_runs.add(test.source_run_seed[index])
        candidates.append((index, sorted({slots[0], 5, slots[-1]})))
    if len(candidates) < states:
        raise ValueError(f"need {states} unseen playable runs with sixth slot, found {len(candidates)}")
    cases = []
    for position in np.linspace(0, len(candidates) - 1, states, dtype=int):
        index, slots = candidates[int(position)]
        for slot in slots:
            obs, masks = test.row(index)
            edits = [change_card(obs, slot, kind) for kind in CONDITIONS[1:]]
            cases.append({"state_index": index, "state_hash": test.state_hashes[index], "slot": slot,
                          "masks": masks, "observations": [obs, *(item[0] for item in edits)],
                          "changes": [item[1] for item in edits]})
    return cases


def rescore(pool_rates, mapping):
    route = np.asarray(mapping.routing.head_routes["card"])
    normalization = mapping.normalization
    return (np.asarray(pool_rates, dtype=np.float64) - normalization.baseline[route]) / normalization.scale[route]


def preference_summary(scores, cases):
    histogram, agreements, seen = Counter(), [], set()
    for index, case in enumerate(cases):
        if case["state_index"] in seen:
            continue
        seen.add(case["state_index"])
        top = np.argmax(np.where(case["masks"]["card_select_mask"], scores[index, 0], -np.inf), axis=1)
        histogram.update(top.tolist())
        agreements.append(float(np.mean([top[a] == top[b] for a, b in combinations(range(len(top)), 2)])))
    return {"top_slot_histogram_zero_based": dict(sorted(histogram.items())),
            "dominant_slot_fraction": histogram.most_common(1)[0][1] / histogram.total(),
            "top_card_repeat_agreement": float(np.mean(agreements))}


def evaluate_cached(output, result, source, candidate, cases):
    """Old and new scores share exactly the same recorded neural responses."""
    assert_only_normalization_changed(source, candidate)
    evaluated = {}
    for weight, entry in result["weights"].items():
        path = output / entry["activity_file"]
        if sha256_file(path) != entry["activity_sha256"]:
            raise ValueError("cached neural activity hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            rates = arrays["card_pool_rates"]
            original = rescore(rates, source)
            if not np.array_equal(original, arrays["card_scores"]):
                raise ValueError("cached original scores do not reproduce source normalization")
            scores = rescore(rates, candidate)
            contrasts = np.empty_like(scores)
            for index, case in enumerate(cases):
                for condition in range(len(CONDITIONS)):
                    for repeat in range(scores.shape[2]):
                        contrasts[index, condition, repeat] = contrast_scores(scores[index, condition, repeat], case["masks"]["card_select_mask"][0])
            health = {stage: {"minimum_active_fraction": float(np.mean(arrays[stage] > 0, axis=-1).min()),
                              "maximum_high_rate_fraction": float(np.mean(arrays[stage] > 200, axis=-1).max())}
                      for stage in ("kc", "card_pool_neurons")}
        rows = deepcopy(entry["rows"])
        for row in rows:
            i, condition = row["case_index"], CONDITIONS.index(row["condition"])
            for name, values in (("card_scores", scores), ("card_contrasts", contrasts)):
                row["populations"][name] = response_metrics(values[i, 0], values[i, condition])
            row["target_contrast_change"] = (contrasts[i, condition, :, row["slot"]] - contrasts[i, 0, :, row["slot"]]).tolist()
            mask = cases[i]["masks"]["card_select_mask"]
            row["preferred_legal_card_changed_fraction"] = float(np.mean(
                np.argmax(np.where(mask, scores[i, 0], -np.inf), axis=1)
                != np.argmax(np.where(mask, scores[i, condition], -np.inf), axis=1)))
        derived = output / (path.stem + "-fresh-normalization.npz")
        np.savez_compressed(derived, card_scores=scores, card_contrasts=contrasts)
        evaluated[weight] = {"health": health,
            "source_neural_sha256": entry["activity_sha256"], "derived_file": derived.name,
            "derived_sha256": sha256_file(derived),
            "old": {"summary": summarize(entry["rows"]), "preference": preference_summary(original, cases)},
            "fresh": {"summary": summarize(rows), "preference": preference_summary(scores, cases)},
            "fresh_rows": rows}
    return evaluated


def assess(learners, *, minimum_kc_active=.01, minimum_motor_active=.01):
    rows = []
    healthy = True
    for learner in learners:
        if not learner["plastic_state_unchanged"] or learner["plastic_state_hashes_before"] != learner["plastic_state_hashes_after"]:
            raise ValueError("normalization study changed plastic state")
        for result in learner["normalization_comparison"].values():
            for stage, floor in (("kc", minimum_kc_active), ("card_pool_neurons", minimum_motor_active)):
                metrics = result["health"][stage]
                healthy &= metrics["minimum_active_fraction"] >= floor and metrics["maximum_high_rate_fraction"] <= .25
        result = learner["normalization_comparison"]["trained"]
        before, after = result["old"], result["fresh"]
        bias = after["preference"]["dominant_slot_fraction"] < before["preference"]["dominant_slot_fraction"] and after["preference"]["dominant_slot_fraction"] <= .5
        signals = {}
        for kind in ("rank", "suit"):
            old, fresh = [side["summary"][kind]["populations"]["card_contrasts"]["ratio_of_mean_effect_to_noise_power"] for side in (before, after)]
            signals[kind] = fresh is not None and old is not None and fresh > max(0, old)
        rows.append({"learner_index": learner["learner_index"], "bias_screen": bool(bias),
                     "edit_signals": signals, "follow_up_signal": bool(bias and all(signals.values()))})
    def means(kind, side):
        return [l["normalization_comparison"]["trained"][side]["summary"][kind]["populations"]["card_contrasts"]["ratio_of_mean_effect_to_noise_power"] for l in learners]
    mean_improves = all(all(value is not None for value in means(kind, "old") + means(kind, "fresh"))
                        and np.mean(means(kind, "fresh")) > np.mean(means(kind, "old")) for kind in ("rank", "suit"))
    return {"version": VERSION, "criterion": CRITERION, "all_health_checks_pass": bool(healthy),
            "learners": rows, "mean_signal_improves_for_both_edits": bool(mean_improves),
            "predeclared_follow_up_signal": bool(healthy and mean_improves and sum(row["follow_up_signal"] for row in rows) > len(rows)/2),
            "interpretation": "descriptive held-out-from-calibration screen; no gameplay or learning claim"}
