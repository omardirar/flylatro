"""Experimental fixed card isolation, selected exclusively from neural statistics."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

from flylatro.interface.motor import FixedMotorInterface
from flylatro.interface.motor_calibration import (
    MotorCalibrationError, MotorCalibrationThresholds, _assign_pools, _group_evidence,
    _refine_pairs, validate_reward_free_motor,
)
from flylatro.interface.motor import MotorNormalization
from flylatro.learning.config import payload_sha256

VERSION = "reward-free-card-isolated-core-motor-v1"


def separation_summary(mapping):
    groups = {group: set(index for pool in mapping.routing.group_pool_ids(group)
                        for index in mapping.pool_indices[pool]) for group in mapping.routing.group_names}
    overlaps = {f"{left}/{right}": len(groups[left] & groups[right])
                for left in groups for right in groups if left < right}
    passed = (not (groups["card_slot"] & set.union(*(value for key, value in groups.items() if key != "card_slot")))
              and not (groups["action_type"] & groups["card_count"]))
    return {"card_isolated_and_core_disjoint": passed, "overlap_neuron_counts": overlaps,
            "unique_core_neurons": len(groups["card_slot"] | groups["action_type"] | groups["card_count"])}


def normalize_fixed(mapping, values, metadata):
    rates = FixedMotorInterface(mapping).pool_activity(values)
    normalization = MotorNormalization.from_pool_activity(rates, minimum_scale_hz=.5)
    stats = {"version": VERSION, "reward_used": False, "outcome_information_used": False,
             "source_mapping_sha256": mapping.structure_sha256, "samples": len(values),
             "normalization_sha256": normalization.sha256, "metadata": metadata}
    return replace(mapping, normalization=normalization, calibration_sha256=payload_sha256(stats), calibration_stats=stats)


def _match_slots(slots, preferences):
    """Augmenting paths preserve capacity even with overlapping eligible sets."""
    owner = {}
    def place(slot, seen):
        for candidate in preferences[slot]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in owner or place(owner[candidate], seen):
                owner[candidate] = slot
                return True
        return False
    for slot in slots:
        if not place(slot, set()):
            raise MotorCalibrationError("insufficient exclusive eligible neurons for action/count pools")
    return {slot: candidate for candidate, slot in owner.items()}


def _joint_refine(pools, core, source, evidence, iterations=200, active_ids=None):
    """Exchange neurons across groups using only pooled neural covariance."""
    ids = active_ids or {g: list(source.routing.group_pool_ids(g)) for g in core}
    covariances = {g: np.cov(evidence[g]["values"], rowvar=False, bias=True) for g in core}
    def penalty(x):
        return np.maximum(.52-x, 0)**2
    def table(group):
        pairs = np.asarray([pools[i] for i in ids[group]])
        cov = covariances[group]
        block = sum(cov[pairs[:, i, None], pairs[None, :, j]] for i in range(2) for j in range(2))
        variance = np.diag(block)
        total = np.sum(block**2 / np.maximum(np.outer(variance, variance), 1e-12))
        current = len(pairs) / max(total, 1e-12)
        result = []
        for position in range(len(pairs)):
            other = np.delete(pairs, position, axis=0)
            other_var = cov[other[:, 0], other[:, 0]]+cov[other[:, 1], other[:, 1]]+2*cov[other[:, 0], other[:, 1]]
            for member in range(2):
                partner = pairs[position, 1-member]
                variance = np.diag(cov)+cov[partner, partner]+2*cov[:, partner]
                cross = cov[:, other[:, 0]]+cov[:, other[:, 1]]+cov[partner, other[:, 0]]+cov[partner, other[:, 1]]
                row_cost = np.sum(cross**2 / np.maximum(variance[:, None]*other_var[None, :], 1e-12), axis=1)
                old = pairs[position, member]
                fractions = len(pairs) / np.maximum(total+2*(row_cost-row_cost[old]), 1e-12)
                valid = evidence[group]["eligible"].copy()
                valid[pairs.ravel()] = False
                valid[old] = True
                fractions[~valid] = -np.inf
                result.append(fractions)
        return current, np.asarray(result)
    for _ in range(iterations):
        tables = {g: table(g) for g in core}
        if all(value[0] >= .52 for value in tables.values()):
            break
        members = {g: np.asarray([pools[i] for i in ids[g]]).ravel() for g in core}
        used = set(np.concatenate(list(members.values())).tolist())
        unused = np.asarray([i for i in range(len(source.output_root_ids)) if i not in used], dtype=int)
        best = (-1e-12, None)
        for gi, group in enumerate(core):
            current, choices = tables[group]
            if len(unused):
                costs = penalty(choices[:, unused])-penalty(current)
                index = np.unravel_index(np.argmin(costs), costs.shape)
                if costs[index] < best[0]:
                    best = (float(costs[index]), (group, index[0], int(unused[index[1]]), None, None))
            for other in core[gi+1:]:
                other_current, other_choices = tables[other]
                costs = (penalty(choices[:, members[other]])-penalty(current)
                         +(penalty(other_choices[:, members[group]])-penalty(other_current)).T)
                index = np.unravel_index(np.argmin(costs), costs.shape)
                if costs[index] < best[0]:
                    best = (float(costs[index]), (group, index[0], int(members[other][index[1]]), other, index[1]))
        if best[1] is None:
            break
        group, position, replacement, other, other_position = best[1]
        old = int(members[group][position])
        pool, member = ids[group][position//2], position%2
        pair = list(pools[pool]); pair[member] = replacement; pools[pool] = tuple(pair)
        if other is not None:
            pool, member = ids[other][other_position//2], other_position%2
            pair = list(pools[pool]); pair[member] = old; pools[pool] = tuple(pair)
    return pools


def _anneal_pairs(pools, core, source, evidence, seed, steps=3000, active_ids=None):
    """Bounded stochastic membership search; objective contains no game outcomes."""
    rng = np.random.default_rng(seed)
    ids = active_ids or {g: list(source.routing.group_pool_ids(g)) for g in core}
    covs = {g: np.cov(evidence[g]["values"], rowvar=False, bias=True) for g in core}
    positions = [(g, pool, member) for g in core for pool in ids[g] for member in range(2)]
    def fraction(group):
        pairs = np.asarray([pools[i] for i in ids[group]])
        cov = covs[group]
        block = sum(cov[pairs[:, i, None], pairs[None, :, j]] for i in range(2) for j in range(2))
        variance = np.diag(block)
        return len(pairs)/max(np.sum(block**2/np.maximum(np.outer(variance, variance), 1e-12)), 1e-12)
    def loss(fractions):
        return sum(max(.51-v, 0)**2 for v in fractions.values())
    fractions = {g: fraction(g) for g in core}
    current = loss(fractions); best_loss, best = current, list(pools)
    for step in range(steps):
        if min(fractions.values()) >= .505:
            return pools
        a, b = rng.choice(len(positions), size=2, replace=False)
        ga, pa, ma = positions[a]; gb, pb, mb = positions[b]
        if pa == pb:
            continue
        va, vb = pools[pa][ma], pools[pb][mb]
        if not evidence[ga]["eligible"][vb] or not evidence[gb]["eligible"][va]:
            continue
        old_a, old_b = pools[pa], pools[pb]
        pair = list(old_a); pair[ma] = vb; pools[pa] = tuple(pair)
        pair = list(old_b); pair[mb] = va; pools[pb] = tuple(pair)
        updated = dict(fractions)
        for group in {ga, gb}:
            updated[group] = fraction(group)
        proposed = loss(updated)
        temperature = .001*(1-step/steps)**3+1e-7
        if proposed < current or rng.random() < np.exp(min(0, (current-proposed)/temperature)):
            current, fractions = proposed, updated
            if proposed < best_loss:
                best_loss, best = proposed, list(pools)
        else:
            pools[pa], pools[pb] = old_a, old_b
    return best


def calibrate_separated(source, values, *, contexts, metadata, attempts=32):
    """Jointly separate card/action/count; isolate cards from every other group.

    Context-slot pools can reuse action/count neurons, preserving the existing
    sharing between mutually exclusive contextual heads. No reward enters here.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(source.output_root_ids) or not np.isfinite(values).all() or (values < 0).any():
        raise MotorCalibrationError("finite nonnegative sample-by-output calibration required")
    if attempts < 1 or any(len(pool) != 2 for pool in source.pool_indices):
        raise MotorCalibrationError("positive search budget and width-two source pools required")
    limits = MotorCalibrationThresholds()
    baseline = normalize_fixed(source, values, metadata)
    baseline_check = validate_reward_free_motor(baseline, values, thresholds=limits, contexts=contexts, calibration_metadata=metadata)
    if baseline_check.status != "PASS":
        raise MotorCalibrationError("baseline reward-free qualification failed", baseline_check.report)
    evidence = _group_evidence(values, source.routing, contexts, limits, source.output_root_ids)
    card_ids = source.routing.group_pool_ids("card_slot")
    core = ("action_type", "card_count", "card_slot")
    ids = {g: list(source.routing.group_pool_ids(g)) for g in core}
    if contexts is not None:
        for group in core:
            window = contexts[group]
            route = source.routing.head_routes[window.spec.head]
            ids[group] = [int(route[i]) for i in window.active_options if route[i] >= 0]
    slots = [(group, pool, member) for group in core for pool in ids[group] for member in range(2)]
    failed = []
    for attempt in range(attempts):
        rng = np.random.default_rng(27_100_000 + attempt)
        preferences = {}
        for slot in slots:
            group, pool, member = slot
            ranked = [int(i) for i in evidence[group]["order"] if evidence[group]["eligible"][i]]
            if attempt:
                ranked = rng.permutation(ranked).tolist()
            original = list(source.pool_indices[pool])
            preferred = [original[member], original[1-member]]
            preferences[slot] = list(dict.fromkeys([i for i in preferred if i in ranked] + ranked))
        order = slots if not attempt else [slots[i] for i in rng.permutation(len(slots))]
        matched = _match_slots(order, preferences)
        pools = list(source.pool_indices)
        for group in core:
            for pool in ids[group]:
                pools[pool] = tuple(matched[(group, pool, member)] for member in range(2))
        for group in core:
            pool_ids = ids[group]
            excluded = {i for other in core if other != group
                                   for pool in ids[other] for i in pools[pool]}
            ranked = [int(i) for i in evidence[group]["order"] if evidence[group]["eligible"][i] and i not in excluded]
            refined = _refine_pairs(evidence[group]["values"], ranked, [pools[i] for i in pool_ids], minimum_effective_fraction=.5)
            for pool, pair in zip(pool_ids, refined, strict=True):
                pools[pool] = pair
        for _ in range(4):
            pools = _joint_refine(pools, core, source, evidence, active_ids=ids)
            for group in core:
                pool_ids = ids[group]
                excluded = {i for other in core if other != group
                            for pool in ids[other] for i in pools[pool]}
                ranked = [int(i) for i in evidence[group]["order"] if evidence[group]["eligible"][i] and i not in excluded]
                refined = _refine_pairs(evidence[group]["values"], ranked, [pools[i] for i in pool_ids], minimum_effective_fraction=.52)
                for pool, pair in zip(pool_ids, refined, strict=True):
                    pools[pool] = pair
        pools = _anneal_pairs(pools, core, source, evidence, 27_150_000+attempt, active_ids=ids)
        used = {i for group in core for pool in ids[group] for i in pools[pool]}
        # Inactive contract options still own disjoint structural pools. They
        # consume the least variable unused neurons; no claim of responsiveness
        # is made for an option that is illegal throughout the experiment.
        quiet = sorted(range(len(source.output_root_ids)), key=lambda i: (values[:, i].var(), int(source.output_root_ids[i])))
        for group in core:
            for pool in source.routing.group_pool_ids(group):
                if pool not in ids[group]:
                    members = tuple(i for i in quiet if i not in used)[:2]
                    if len(members) != 2:
                        raise MotorCalibrationError("no distinct neurons for inactive contract option")
                    pools[pool] = members; used.update(members)
        reserved = {index for pool in card_ids for index in pools[pool]}
        # Contextual roles retain their six shared slots, but cannot use card neurons.
        contextual = dict(evidence["context_slot"])
        contextual["eligible"] = contextual["eligible"].copy()
        contextual["eligible"][list(reserved)] = False
        context_ids = list(source.routing.group_pool_ids("context_slot"))
        routing = SimpleNamespace(pool_count=len(context_ids), group_names=("context_slot",),
            group_pool_counts=(len(context_ids),), group_pool_ids=lambda group: range(len(context_ids)))
        context_assignment = _assign_pools(routing=routing, pool_width=2, maximum_correlation=.95,
            minimum_effective_fraction=.5, group_evidence={"context_slot": contextual})
        for pool, pair in zip(context_ids, context_assignment["pool_indices"], strict=True):
            pools[pool] = pair
        temporary = replace(source, pool_indices=tuple(pools), selection_method=VERSION)
        candidate = normalize_fixed(temporary, values, {**metadata, "attempt": attempt, "policy": VERSION})
        check = validate_reward_free_motor(candidate, values, thresholds=limits, contexts=contexts, calibration_metadata=metadata)
        if not separation_summary(candidate)["card_isolated_and_core_disjoint"]:
            raise ValueError("separated allocation violated disjointness")
        if check.status == "PASS":
            return baseline, candidate, {"version": VERSION, "thresholds": asdict(limits),
                "reward_used": False, "outcome_information_used": False, "selected_attempt": attempt,
                "baseline": baseline_check.report, "separated": check.report,
                "separation": separation_summary(candidate), "failed_search_attempts": failed,
                "active_core_pools": ids, "inactive_pool_policy": "least variable unused neurons; no response qualification for illegal options",
                "card_pools_changed": sum(candidate.pool_indices[i] != source.pool_indices[i] for i in card_ids)}
        failed.append({"attempt": attempt, "failed": check.report["gates"]["failed"],
            "diversity": {name: entry.get("effective_signal_fraction")
                for name, entry in check.report["calibration"]["quality"]["by_context"].items()}})
    raise MotorCalibrationError("no separated candidate passed unchanged reward-free gates", {"attempts": failed})
