"""Reward-free allocation for distinct PLAY/DISCARD card and count populations."""
from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

from flylatro.interface.motor import MotorMapping, MotorNormalization, MotorRouting
from flylatro.interface.motor_calibration import (
    MotorCalibrationError, MotorCalibrationThresholds, _assign_pools, _group_evidence,
    _refine_pairs, validate_reward_free_motor,
)
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.interface.separated_motor import _anneal_pairs, _joint_refine, _match_slots, normalize_fixed
from flylatro.learning.config import payload_sha256


VERSION = "reward-free-play-discard-separated-motor-v1"
CORE_GROUPS = ("action_type", "card_count", "card_slot", "discard_card_count", "discard_card_slot")


def structural_mapping(roots, *, mode="mbon_direct", pool_width=2, **exploration):
    """Uncalibrated structural fixture; only save for real use after qualification.

    Core groups are exclusive (90 neurons at width two). Context targets may
    reuse noncard action/count neurons, as in the previous separated mapping.
    """
    roots = np.asarray(roots, dtype=np.int64)
    routing = MotorRouting.action_conditioned_v2()
    needed = sum(len(routing.group_pool_ids(g)) for g in CORE_GROUPS)*pool_width
    if pool_width < 1 or len(roots) < needed:
        raise MotorCalibrationError(f"action-conditioned core needs {needed} distinct outputs")
    pools = [None]*routing.pool_count
    cursor = 0
    for group in CORE_GROUPS:
        for pool in routing.group_pool_ids(group):
            pools[pool] = tuple(range(cursor, cursor+pool_width)); cursor += pool_width
    noncard = [i for group in ("action_type", "card_count", "discard_card_count")
               for pool in routing.group_pool_ids(group) for i in pools[pool]]
    for index, pool in enumerate(routing.group_pool_ids("context_slot")):
        pools[pool] = tuple(noncard[index*pool_width:(index+1)*pool_width])
    return MotorMapping(version="fixed-action-conditioned-population-motor-v1", mode=mode,
        output_root_ids=roots.copy(), pool_indices=tuple(pools), routing=routing,
        normalization=MotorNormalization.identity(routing.pool_count),
        selection_method="uncalibrated-action-conditioned-structural-fixture", **exploration)


def separation_summary(mapping):
    groups = {g: {i for pool in mapping.routing.group_pool_ids(g) for i in mapping.pool_indices[pool]}
              for g in CORE_GROUPS}
    return {"version": VERSION, "action_conditioned": mapping.routing.action_conditioned,
        "unique_core_neurons": len(set.union(*groups.values())),
        "core_overlap_neuron_counts": {f"{a}/{b}": len(groups[a] & groups[b])
            for j, a in enumerate(CORE_GROUPS) for b in CORE_GROUPS[j+1:]},
        "play_card_roots": [int(mapping.output_root_ids[i]) for i in sorted(groups["card_slot"])],
        "discard_card_roots": [int(mapping.output_root_ids[i]) for i in sorted(groups["discard_card_slot"])],
        "consumable_policy": "legacy PLAY card/count interpretation retained; not a consumable-targeting redesign"}


def _refine_singletons(pools, source, evidence, ids, *, seed, steps):
    """Exchange/replace exclusive neurons using covariance only, never labels."""
    rng = np.random.default_rng(seed)
    covariance = {g: np.cov(evidence[g]['values'], rowvar=False, bias=True) for g in CORE_GROUPS}
    positions = [(g, pool) for g in CORE_GROUPS for pool in ids[g]]
    def fraction(group):
        selected = np.asarray([pools[pool][0] for pool in ids[group]])
        block = covariance[group][np.ix_(selected, selected)]
        variance = np.diag(block)
        return len(selected)/max(float(np.sum(block**2/np.maximum(np.outer(variance, variance), 1e-12))), 1e-12)
    fractions = {g: fraction(g) for g in CORE_GROUPS}
    loss = lambda result: sum(max(.505-value, 0)**2 for value in result.values())
    current = loss(fractions); best_loss, best = current, list(pools)
    for step in range(steps):
        if min(fractions.values()) >= .505:
            return pools
        group, pool = positions[int(rng.integers(len(positions)))]
        old = pools[pool][0]
        candidate = int(rng.integers(len(source.output_root_ids)))
        if not evidence[group]['eligible'][candidate] or candidate == old:
            continue
        other = next(((g, p) for g, p in positions if pools[p][0] == candidate), None)
        if other is not None and (other[0] == group or not evidence[other[0]]['eligible'][old]):
            continue
        pools[pool] = (candidate,)
        if other is not None:
            pools[other[1]] = (old,)
        proposed_fractions = dict(fractions)
        for changed in ({group, other[0]} if other else {group}):
            proposed_fractions[changed] = fraction(changed)
        proposed = loss(proposed_fractions)
        temperature = .0005*(1-step/max(steps, 1))**3+1e-8
        if proposed < current or rng.random() < np.exp(min(0., (current-proposed)/temperature)):
            current, fractions = proposed, proposed_fractions
            if current < best_loss:
                best_loss, best = current, list(pools)
        else:
            pools[pool] = (old,)
            if other is not None:
                pools[other[1]] = (candidate,)
    return best


def calibrate_action_conditioned(roots, values, *, masks, order=None, metadata=None,
                                 attempts=32, refinement_rounds=4, annealing_steps=3000,
                                 pool_width=2, progress=None, **exploration):
    """Use original response/diversity gates at an explicitly declared pool width."""
    metadata = dict(metadata or {})
    if pool_width not in (1, 2):
        raise MotorCalibrationError('declare a pool width of one or two')
    source = structural_mapping(roots, pool_width=pool_width, **exploration)
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != len(source.output_root_ids)
            or len(values) < 4 or not np.isfinite(values).all() or np.any(values < 0)):
        raise MotorCalibrationError("finite nonnegative sample-by-output calibration required")
    if attempts < 1 or refinement_rounds < 0 or annealing_steps < 0:
        raise MotorCalibrationError("positive attempts and nonnegative search budgets required")
    contexts = motor_context_windows(masks, order=order, routing=source.routing)
    limits = MotorCalibrationThresholds(minimum_pool_width=pool_width)
    evidence = _group_evidence(values, source.routing, contexts, limits, source.output_root_ids)
    ids = {}
    for group in CORE_GROUPS:
        window = contexts[group]
        route = source.routing.head_routes[window.spec.head]
        ids[group] = [route[i] for i in window.active_options if route[i] >= 0]
        if not ids[group]:
            raise MotorCalibrationError(f"no legal calibration options for {group}")
    slots = [(group, pool, member) for group in CORE_GROUPS for pool in ids[group] for member in range(pool_width)]
    failed = []
    for attempt in range(attempts):
        if progress:
            progress({"phase": "reward-free-allocation", "attempt": attempt, "attempts": attempts})
        rng = np.random.default_rng(27_300_000+attempt)
        preferences = {}
        for slot in slots:
            group = slot[0]
            ranked = [int(i) for i in evidence[group]["order"] if evidence[group]["eligible"][i]]
            preferences[slot] = ranked if attempt == 0 else rng.permutation(ranked).tolist()
        try:
            match = _match_slots(slots if attempt == 0 else [slots[i] for i in rng.permutation(len(slots))], preferences)
            pools = list(source.pool_indices)
            for group in CORE_GROUPS:
                for pool in ids[group]:
                    pools[pool] = tuple(match[(group, pool, member)] for member in range(pool_width))
            for _ in range(refinement_rounds if pool_width == 2 else 0):
                pools = _joint_refine(pools, CORE_GROUPS, source, evidence, active_ids=ids)
                for group in CORE_GROUPS:
                    excluded = {i for other in CORE_GROUPS if other != group for pool in ids[other] for i in pools[pool]}
                    ranked = [int(i) for i in evidence[group]["order"] if evidence[group]["eligible"][i] and i not in excluded]
                    pairs = _refine_pairs(evidence[group]["values"], ranked, [pools[i] for i in ids[group]], minimum_effective_fraction=.52)
                    for pool, pair in zip(ids[group], pairs, strict=True):
                        pools[pool] = pair
            if pool_width == 2:
                pools = _anneal_pairs(pools, CORE_GROUPS, source, evidence, 27_350_000+attempt,
                                      steps=annealing_steps, active_ids=ids)
            else:
                pools = _refine_singletons(pools, source, evidence, ids,
                                           seed=27_350_000+attempt, steps=annealing_steps)
            used = {i for group in CORE_GROUPS for pool in ids[group] for i in pools[pool]}
            quiet = sorted(range(len(roots)), key=lambda i: (values[:, i].var(), int(roots[i])))
            for group in CORE_GROUPS:
                for pool in source.routing.group_pool_ids(group):
                    if pool not in ids[group]:
                        pair = tuple(i for i in quiet if i not in used)[:pool_width]
                        if len(pair) != pool_width:
                            raise MotorCalibrationError("no exclusive outputs for inactive contract options")
                        pools[pool] = pair; used.update(pair)
            cards = {i for group in ("card_slot", "discard_card_slot")
                     for pool in source.routing.group_pool_ids(group) for i in pools[pool]}
            contextual = dict(evidence["context_slot"])
            contextual["eligible"] = contextual["eligible"].copy()
            contextual["eligible"][list(cards)] = False
            context_ids = source.routing.group_pool_ids("context_slot")
            routing = SimpleNamespace(pool_count=len(context_ids), group_names=("context_slot",),
                group_pool_counts=(len(context_ids),), group_pool_ids=lambda group: range(len(context_ids)))
            assignment = _assign_pools(routing=routing, pool_width=pool_width,
                maximum_correlation=limits.maximum_within_group_correlation,
                minimum_effective_fraction=limits.minimum_effective_signal_fraction,
                group_evidence={"context_slot": contextual})
            for pool, pair in zip(context_ids, assignment["pool_indices"], strict=True):
                pools[pool] = pair
            candidate = normalize_fixed(replace(source, pool_indices=tuple(pools), selection_method=VERSION),
                values, {**metadata, "action_conditioned_version": VERSION, "attempt": attempt})
            stats = {**candidate.calibration_stats, 'version': VERSION,
                     'declared_pool_width': pool_width}
            candidate = replace(candidate, calibration_stats=stats, calibration_sha256=payload_sha256(stats))
            check = validate_reward_free_motor(candidate, values, thresholds=limits,
                contexts=contexts, calibration_metadata=metadata)
            if check.status == "PASS":
                return candidate, {"version": VERSION, "reward_used": False,
                    "outcome_information_used": False, "selected_attempt": attempt,
                    "thresholds": asdict(limits), "qualification": check.report,
                    "separation": separation_summary(candidate), "failed_attempts": failed,
                    "active_core_pools": ids, "declared_pool_width": pool_width,
                    "width_policy": "distinct MBONs per option; explicit width choice, original response/diversity/coverage limits"}
            failed.append({"attempt": attempt, "gates": check.report["gates"],
                "context_diversity": {name: row.get("effective_signal_fraction")
                    for name, row in check.report["calibration"]["quality"]["by_context"].items()}})
        except MotorCalibrationError as error:
            failed.append({"attempt": attempt, "error": str(error), "report": error.report})
        if progress:
            progress({"phase": "allocation-attempt-failed", "attempt": attempt,
                      "failed": failed[-1].get("gates", failed[-1].get("error"))})
    raise MotorCalibrationError("no action-conditioned mapping passed unchanged quality gates", {"attempts": failed})
