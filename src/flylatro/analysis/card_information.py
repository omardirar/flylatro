"""Read-only single-card input interventions through the frozen neural stack."""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

from flylatro.analysis.card_selection import plastic_hashes
from flylatro.fly.plastic_features import observation_features
from flylatro.seeds import derive_seed


VERSION = "frozen-card-information-flow-v1"
CONDITIONS = ("original", "rank", "suit")
STAGE_UNITS = {
    "kc": "Hz", "mbon": "Hz", "card_pool_neurons": "Hz",
    "mbon_outside_card_pools": "Hz", "mbon_unused_by_motor": "Hz",
    "card_pool_rates": "Hz", "card_scores": "normalized motor score",
    "card_contrasts": "normalized motor score relative to legal-card mean",
}


def visible_card(card):
    """Pinned contract: rank [0,13), suit [13,17), Stone 23, face-down 37."""
    return bool(card[23] == 0 and card[37] == 0
                and np.isin(card[:17], (0, 1)).all()
                and card[:13].sum() == 1 and card[13:17].sum() == 1)


def change_card(observations, slot, kind):
    if kind not in ("rank", "suit"):
        raise ValueError("intervention must be rank or suit")
    if not 0 <= slot < int(observations["hand_len"][0]):
        raise ValueError("card slot outside occupied hand")
    card = observations["hand"][0, slot]
    if not visible_card(card):
        raise ValueError("intervention requires a visible one-hot rank and suit")
    start, width = (0, 13) if kind == "rank" else (13, 4)
    old = int(np.argmax(card[start:start + width]))
    new = (old + 1) % width
    altered = {key: value.copy() for key, value in observations.items()}
    altered["hand"][0, slot, start + old] = 0
    altered["hand"][0, slot, start + new] = 1
    return altered, {"kind": kind, "slot": slot, "old_category": old,
                     "new_category": new, "changed_hand_columns": [start + old, start + new]}


def prepare_cases(corpus, *, states, cards_per_state):
    if min(states, cards_per_state) < 1:
        raise ValueError("state and card counts must be positive")
    candidates = {}
    for index in range(len(corpus)):
        if not corpus.masks["action_type_mask"][index, 0]:
            continue
        hand = corpus.observations["hand"][index]
        slots = [slot for slot in range(int(corpus.observations["hand_len"][index]))
                 if corpus.masks["card_select_mask"][index, slot] and visible_card(hand[slot])]
        if slots and corpus.masks["card_select_mask"][index].sum() > 1:
            candidates.setdefault(corpus.state_hashes[index], (index, slots))
    available = list(candidates.values())
    if not available:
        raise ValueError("corpus has no competing PLAY states with visible card identities")
    selected = [available[int(i)] for i in np.linspace(0, len(available) - 1, min(states, len(available)), dtype=int)]
    cases = []
    for index, slots in selected:
        for position in np.linspace(0, len(slots) - 1, min(cards_per_state, len(slots)), dtype=int):
            slot = slots[int(position)]
            obs, masks = corpus.row(index)
            changes = [change_card(obs, slot, kind) for kind in CONDITIONS[1:]]
            cases.append({"state_index": index, "state_hash": corpus.state_hashes[index],
                          "slot": slot, "masks": masks,
                          "observations": [obs, *(item[0] for item in changes)],
                          "changes": [item[1] for item in changes]})
    return cases


def _unclipped_drive(encoder, observations):
    mapping = encoder.mapping
    features = observation_features(observations)
    rates = np.zeros((1, mapping.neuron_count), dtype=np.float32)
    for width in range(mapping.population_indices.shape[1]):
        np.add.at(rates[0], mapping.population_indices[:, width], features[0])
    rates *= mapping.max_rate_hz
    encoded = np.asarray(encoder.encode(observations).rates_hz)
    if not np.array_equal(np.clip(rates, 0, mapping.max_rate_hz), encoded):
        raise ValueError("diagnostic reconstruction differs from live clipped-sum encoder")
    return features[0], rates[0], encoded[0]


def sensory_effects(encoder, cases):
    """Measure deterministic input loss; unclipped rates are never simulated."""
    mapping = encoder.mapping
    alpn_roots = dict(zip(mapping.available_alpn_indices.tolist(), mapping.available_alpn_root_ids.tolist()))
    if mapping.collision_policy != "clipped_sum_preserve_sparse_indicators":
        raise ValueError("diagnostic requires the declared clipped-sum encoder")
    records, vectors = [], []
    for case in cases:
        encoded = [_unclipped_drive(encoder, obs) for obs in case["observations"]]
        base_features, base_raw, base = encoded[0]
        details = []
        for change, (features, raw, clipped) in zip(case["changes"], encoded[1:]):
            channels = np.flatnonzero(features != base_features)
            if len(channels) != 2:
                raise ValueError("single categorical change must alter exactly two feature channels")
            affected = np.unique(mapping.population_indices[channels].ravel())
            raw_delta, delta = raw - base_raw, clipped - base
            raw_l1, clipped_l1 = float(np.abs(raw_delta).sum()), float(np.abs(delta).sum())
            details.append({**change, "changed_feature_names": [mapping.feature_names[i] for i in channels],
                "affected_alpn_root_ids": [alpn_roots[int(index)] for index in affected],
                "affected_neuron_indices": affected.tolist(),
                "before_unclipped_hz": base_raw[affected].tolist(), "after_unclipped_hz": raw[affected].tolist(),
                "before_clipped_hz": base[affected].tolist(), "after_clipped_hz": clipped[affected].tolist(),
                "unclipped_l1_change_hz": raw_l1, "clipped_l1_change_hz": clipped_l1,
                "changed_input_neurons": int(np.count_nonzero(delta)),
                "retained_l1_fraction": clipped_l1 / raw_l1 if raw_l1 else None,
                "projection_alias": raw_l1 == 0, "identical_encoded_input": clipped_l1 == 0,
                "clipping_erased_nonzero_change": raw_l1 > 0 and clipped_l1 == 0,
                "baseline_alpn_saturation_fraction": float(np.mean(base[mapping.available_alpn_indices] >= mapping.max_rate_hz)),
            })
        records.append({"state_index": case["state_index"], "slot": case["slot"], "changes": details})
        vectors.append(np.stack([entry[2][mapping.available_alpn_indices] for entry in encoded]))
    return records, np.stack(vectors)


def verify_erased_input_controls(result, sensory_records):
    """Identical encoded drive and seed must reproduce every measured response."""
    erased = {(row["state_index"], row["slot"], change["kind"])
              for row in sensory_records for change in row["changes"]
              if change["identical_encoded_input"]}
    checks = 0
    for weights in result["weights"].values():
        for row in weights["rows"]:
            if (row["state_index"], row["slot"], row["condition"]) in erased:
                for stage in row["populations"].values():
                    if stage["population_size"] and stage["same_seed_identical_fraction"] != 1:
                        raise ValueError("identical encoded input changed a measured neural response")
                checks += 1
    return checks


def contrast_scores(scores, legal):
    values = np.asarray(scores, dtype=np.float64)
    available = np.asarray(legal, dtype=bool)
    if not available.any():
        raise ValueError("card contrast requires legal cards")
    result = np.zeros_like(values)
    result[available] = values[available] - values[available].mean()
    return result


def response_metrics(original, altered):
    """Descriptive common-RNG effects and independent-seed repeat variability.

    Cross-repeat effect power retains its sign. Negative finite-sample estimates
    are not clipped into an apparent positive signal. None is a decoding test.
    """
    base, changed = np.asarray(original, dtype=np.float64), np.asarray(altered, dtype=np.float64)
    if base.ndim != 2 or changed.shape != base.shape or base.shape[0] < 2:
        raise ValueError("responses require matching [repeats >= 2, neurons] arrays")
    if not np.isfinite(base).all() or not np.isfinite(changed).all():
        raise ValueError("nonfinite population response")
    if base.shape[1] == 0:
        return {"population_size": 0, "paired_rms": None, "baseline_repeat_rms": None,
                "effect_to_repeat_rms_ratio": None, "cross_repeat_effect_power": None,
                "trial_noise_power": None, "effect_to_trial_noise_power_ratio": None,
                "same_seed_identical_fraction": None}
    pairs = list(combinations(range(len(base)), 2))
    difference = changed - base
    baseline_rms = float(np.mean([np.sqrt(np.mean((base[a] - base[b])**2)) for a, b in pairs]))
    paired_rms = float(np.sqrt(np.mean(difference**2, axis=1)).mean())
    power = float(np.mean([np.mean(difference[a] * difference[b]) for a, b in pairs]))
    noise = float((np.var(base, axis=0, ddof=1).mean() + np.var(changed, axis=0, ddof=1).mean()) / 2)
    return {"population_size": base.shape[1], "paired_rms": paired_rms,
            "baseline_repeat_rms": baseline_rms,
            "effect_to_repeat_rms_ratio": paired_rms / baseline_rms if baseline_rms else None,
            "cross_repeat_effect_power": power, "trial_noise_power": noise,
            "effect_to_trial_noise_power_ratio": power / noise if noise else None,
            "same_seed_identical_fraction": float(np.mean(np.all(base == changed, axis=1)))}


def population_layout(processor, motor):
    mbon_roots = np.asarray(processor.mbon_root_ids)
    output_roots = np.asarray(processor.output_root_ids)
    if not np.isin(output_roots, mbon_roots).all():
        raise ValueError("card information flow requires MBON-direct motor outputs")
    routes = np.asarray(motor.mapping.routing.head_routes["card"])
    card_columns = np.unique(np.concatenate([motor.mapping.pool_indices[i] for i in routes])).astype(int)
    all_columns = np.unique(np.concatenate(motor.mapping.pool_indices)).astype(int)
    lookup = {int(root): i for i, root in enumerate(mbon_roots)}
    columns = {
        "card_pool_neurons": np.asarray([lookup[int(root)] for root in output_roots[card_columns]], dtype=int),
        "mbon_outside_card_pools": np.flatnonzero(~np.isin(mbon_roots, output_roots[card_columns])),
        "mbon_unused_by_motor": np.flatnonzero(~np.isin(mbon_roots, output_roots[all_columns])),
    }
    roots = {"kc": np.asarray(processor.kc_root_ids).tolist(), "mbon": mbon_roots.tolist(),
             **{name: mbon_roots[index].tolist() for name, index in columns.items()}}
    return columns, roots


def measure_learner(agent, learner, cases, *, repeats, seed, output_dir, label, progress=None):
    if repeats < 2 or not cases:
        raise ValueError("need cases and at least two independent neural repeats")
    state, processor, motor = agent.plasticity.state, agent.processor, agent.motor
    if not 0 <= learner < state.learners:
        raise ValueError("learner index outside checkpoint")
    columns, roots = population_layout(processor, motor)
    before = plastic_hashes(state)
    results, repeat_checks = {}, 0
    for weight_label, weights in (("initial", state.initial_efficacy), ("trained", state.efficacy)):
        collected = {name: [] for name in STAGE_UNITS}
        trial_rows = []
        base_cache = {}
        for case_index, case in enumerate(cases):
            condition_values = {name: [] for name in STAGE_UNITS}
            seeds = [derive_seed("card-information-flow", seed, case["state_index"], repeat)
                     for repeat in range(repeats)]
            for condition, obs in zip(CONDITIONS, case["observations"]):
                responses = []
                for repeat, neural_seed in enumerate(seeds):
                    def sample():
                        activity = processor.process(obs, fly_seeds=(neural_seed,), efficacy=weights[learner:learner + 1])
                        kc = np.asarray(activity.kc_activity)[0].copy()
                        mbon = np.asarray(activity.mbon_activity)[0].copy()
                        output = np.asarray(activity.output_activity).copy()
                        scores = motor.head_scores(output)["card"][0]
                        pooled = motor.pool_activity(output)[0, list(motor.mapping.routing.head_routes["card"])]
                        return {"kc": kc, "mbon": mbon,
                                **{name: mbon[index].copy() for name, index in columns.items()},
                                "card_pool_rates": pooled.copy(), "card_scores": scores.copy(),
                                "card_contrasts": contrast_scores(scores, case["masks"]["card_select_mask"][0])}
                    cache_key = (case["state_index"], repeat)
                    if condition == "original" and cache_key in base_cache:
                        values = base_cache[cache_key]
                    else:
                        values = sample()
                        if repeat == 0:
                            duplicate = sample()
                            if any(not np.array_equal(values[key], duplicate[key]) for key in values):
                                raise ValueError("exact neural repeat check failed")
                            repeat_checks += 1
                        if condition == "original":
                            base_cache[cache_key] = values
                    if any(not np.isfinite(value).all() for value in values.values()):
                        raise ValueError("nonfinite neural readout")
                    responses.append(values)
                for name in STAGE_UNITS:
                    condition_values[name].append(np.stack([value[name] for value in responses]))
            for name in STAGE_UNITS:
                collected[name].append(np.stack(condition_values[name]))
            for condition_index, condition in enumerate(CONDITIONS[1:], 1):
                metrics = {name: response_metrics(values[0], values[condition_index])
                           for name, values in condition_values.items()}
                base = condition_values["card_contrasts"][0]
                altered = condition_values["card_contrasts"][condition_index]
                trial_rows.append({"case_index": case_index, "state_index": case["state_index"],
                    "slot": case["slot"], "condition": condition, "neural_seeds": seeds,
                    "populations": metrics,
                    "target_contrast_change": (altered[:, case["slot"]] - base[:, case["slot"]]).tolist(),
                    "preferred_legal_card_changed_fraction": float(np.mean(
                        np.argmax(np.where(case["masks"]["card_select_mask"], altered, -np.inf), axis=1)
                        != np.argmax(np.where(case["masks"]["card_select_mask"], base, -np.inf), axis=1))),
                })
            if progress:
                progress(weight_label, case["state_index"], case["slot"])
        arrays = {name: np.stack(values) for name, values in collected.items()}
        array_path = output_dir / f"{label}-learner-{learner}-{weight_label}.npz"
        np.savez_compressed(array_path, **arrays)
        results[weight_label] = {"activity_file": array_path.name,
            "activity_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "array_axes": ["case", "condition", "neural_repeat", "population_column"],
            "rows": trial_rows}
    after = plastic_hashes(state)
    if before != after:
        raise ValueError("frozen information-flow diagnostic changed plastic state")
    return {"learner_index": learner, "plastic_state_hashes_before": before,
            "plastic_state_hashes_after": after, "plastic_state_unchanged": True,
            "exact_repeat_checks": repeat_checks, "population_root_ids": roots,
            "weights": results}


def summarize(rows):
    """State-balanced descriptive estimates; slots and repeats are not replicates."""
    result = {}
    for condition in CONDITIONS[1:]:
        subset = [row for row in rows if row["condition"] == condition]
        stage_summary = {}
        for stage in STAGE_UNITS:
            keys = list(subset[0]["populations"][stage]) if subset else []
            values = {}
            for key in keys:
                state_means = []
                for state in sorted({row["state_index"] for row in subset}):
                    entries = [row["populations"][stage][key] for row in subset
                               if row["state_index"] == state and row["populations"][stage][key] is not None]
                    if entries:
                        state_means.append(float(np.mean(entries)))
                values[key] = float(np.mean(state_means)) if state_means else None
            # A ratio of state-balanced mean powers is less unstable than
            # averaging per-case ratios when some noise estimates approach zero.
            effect, noise = values.get("cross_repeat_effect_power"), values.get("trial_noise_power")
            values["ratio_of_mean_effect_to_noise_power"] = effect / noise if effect is not None and noise else None
            stage_summary[stage] = values
        result[condition] = {"states": len({row["state_index"] for row in subset}),
                             "cases": len(subset), "populations": stage_summary}
    return result
