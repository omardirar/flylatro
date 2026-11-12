"""Cached-response audit of anatomical compression and executed-choice credit."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from flylatro.learning.credit import SelectedMotorCredit

VERSION = "offline-card-pathway-credit-audit-v1"
STAGES = ("connected_kc", "clipped_connected_kc", "afferent_all", "afferent_card",
          "mbon_card", "card_pool", "eligible_card_edges", "gated_card_edges")


def root_columns(population, requested):
    population = np.asarray(population)
    if len(np.unique(population)) != len(population):
        raise ValueError("population roots must be unique")
    lookup = {int(root): index for index, root in enumerate(population)}
    if any(int(root) not in lookup for root in requested):
        raise ValueError("topology or mapping root absent from cached population")
    return np.asarray([lookup[int(root)] for root in requested], dtype=int)


class PathwayLayout:
    def __init__(self, topology, motor, kc_roots, mbon_roots):
        self.topology, self.motor = topology, motor
        self.pre = root_columns(kc_roots, topology.pre_root_ids)
        self.post = root_columns(mbon_roots, topology.post_root_ids)
        self.output = root_columns(mbon_roots, motor.mapping.output_root_ids)
        self.card_pools = np.asarray(motor.mapping.routing.head_routes["card"], dtype=int)
        if (self.card_pools < 0).any():
            raise ValueError("all card options need fixed motor pools")
        self.pool_posts = [self.output[np.asarray(pool, dtype=int)] for pool in motor.mapping.pool_indices]
        self.card_posts = np.unique(np.concatenate([self.pool_posts[i] for i in self.card_pools]))
        self.card_edges = np.flatnonzero(np.isin(self.post, self.card_posts))
        self.connected_kc = np.unique(self.pre[self.card_edges])
        self.kc_count, self.mbon_count = len(kc_roots), len(mbon_roots)
        self.credit = SelectedMotorCredit(motor, SimpleNamespace(topology=topology, is_torch=False))

    def structural_summary(self):
        card_set = set(self.card_posts.tolist())
        overlaps = {}
        for head, routes in self.motor.mapping.routing.head_routes.items():
            posts = set(np.concatenate([self.pool_posts[i] for i in routes if i >= 0]).tolist())
            overlaps[head] = {"mbons": len(posts), "shared_with_card": len(card_set & posts)}
        per_pool = []
        for slot, pool in enumerate(self.card_pools):
            edges = np.isin(self.post, self.pool_posts[pool])
            shared = [int(other) for other in range(len(self.pool_posts))
                      if other not in self.card_pools and np.intersect1d(self.pool_posts[pool], self.pool_posts[other]).size]
            per_pool.append({"slot": slot, "pool": int(pool), "edges": int(edges.sum()),
                "connected_kcs": int(len(np.unique(self.pre[edges]))), "sharing_noncard_pools": shared,
                "mbon_root_ids": self.motor.mapping.output_root_ids[list(self.motor.mapping.pool_indices[pool])].tolist()})
        return {"plastic_edges": self.topology.edge_count, "kc_count": self.kc_count,
                "mbon_count": self.mbon_count, "card_mbons": len(self.card_posts),
                "card_edges": len(self.card_edges), "card_connected_kcs": len(self.connected_kc),
                "card_connected_kc_fraction": len(self.connected_kc) / self.kc_count,
                "head_overlap": overlaps, "card_pools": per_pool}

    def projection(self, efficacy):
        """A static weighted KC-rate average, not a simulated MBON response."""
        weights = self.topology.anatomical_weights.astype(np.float64) * np.asarray(efficacy)
        if weights.shape != self.pre.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("finite nonnegative edge-aligned efficacies required")
        result = np.zeros((self.kc_count, self.mbon_count), dtype=np.float64)
        np.add.at(result, (self.pre, self.post), weights)
        total = result.sum(axis=0)
        return np.divide(result, total, out=np.zeros_like(result), where=total > 0)


def feature_matrices(layout, kc, mbon, efficacy, masks, config):
    """Preserve nonlinear operations per repeat before averaging by game run.

    Coactivity vectors are the exact *new* eligibility increments from zero
    traces. They exclude historical eligibility and reinforcement outcomes.
    """
    kc, mbon = np.asarray(kc, dtype=np.float32), np.asarray(mbon, dtype=np.float32)
    if (kc.ndim != 3 or mbon.ndim != 3 or kc.shape[:2] != mbon.shape[:2]
            or kc.shape[2] != layout.kc_count or mbon.shape[2] != layout.mbon_count
            or not np.isfinite(kc).all() or not np.isfinite(mbon).all()
            or (kc < 0).any() or (mbon < 0).any()):
        raise ValueError("finite nonnegative rate arrays must be [run, repeat, neuron]")
    runs, repeats = kc.shape[:2]
    kflat, mflat = kc.reshape(-1, kc.shape[-1]), mbon.reshape(-1, mbon.shape[-1])
    repeated_masks = {key: np.repeat(value, repeats, axis=0) for key, value in masks.items()}
    output = mflat[:, layout.output]
    actions = layout.motor.decode(output, repeated_masks, deterministic=True)
    pools = layout.motor.selected_pool_mask(actions, repeated_masks)
    gate = layout.credit.mask(actions, repeated_masks)
    card_pool_selection = pools[:, layout.card_pools]
    intended_posts = np.zeros((len(output), layout.mbon_count), dtype=bool)
    for slot, pool in enumerate(layout.card_pools):
        intended_posts[:, layout.pool_posts[pool]] |= card_pool_selection[:, slot, None]
    intended_edges = intended_posts[:, layout.post[layout.card_edges]]
    card_gate = gate[:, layout.card_edges]
    aliased = card_gate & ~intended_edges
    # Logical union is the implemented rule: selected action/count pools can
    # admit edges into an unselected card pool when their MBONs are shared.
    if np.any(intended_edges & ~card_gate):
        raise ValueError("actual credit gate omitted intended card edges")
    card_pre = layout.pre[layout.card_edges]
    card_post = layout.post[layout.card_edges]
    pre_scaled = np.clip(kflat / config.kc_reference_hz, 0, 1)
    post_scaled = np.clip(mflat / config.mbon_reference_hz, 0, 1)
    # Bound temporary edge matrices while retaining one feature per edge.
    eligible, gated = [], []
    totals = {key: 0 for key in ("admitted", "nonzero", "aliased_nonzero", "pre_saturated", "post_saturated", "both_saturated", "coactivity_count")}
    coactivity_mass, aliased_mass = 0., 0.
    aliased_nonzero_by_run = []
    for start in range(0, runs, 16):
        stop = min(start + 16, runs)
        sl = slice(start * repeats, stop * repeats)
        pre, post = pre_scaled[sl, card_pre], post_scaled[sl, card_post]
        coincidence = pre * post
        local_gate = card_gate[sl]
        eligible.append(np.minimum(coincidence, config.max_eligibility).reshape(stop - start, repeats, -1).mean(axis=1))
        gated.append(np.minimum(coincidence * local_gate, config.max_eligibility).reshape(stop - start, repeats, -1).mean(axis=1))
        totals["admitted"] += int(local_gate.sum())
        totals["nonzero"] += int(((coincidence > 0) & local_gate).sum())
        active_alias = (coincidence > 0) & aliased[sl]
        totals["aliased_nonzero"] += int(active_alias.sum())
        aliased_nonzero_by_run.append(active_alias.reshape(stop - start, repeats, -1).sum(axis=(1, 2)))
        increment = np.minimum(coincidence, config.max_eligibility)
        coactivity_mass += float(np.sum(increment * local_gate, dtype=np.float64))
        aliased_mass += float(np.sum(increment * aliased[sl], dtype=np.float64))
        totals["pre_saturated"] += int(((pre >= 1) & local_gate).sum())
        totals["post_saturated"] += int(((post >= 1) & local_gate).sum())
        totals["both_saturated"] += int(((coincidence >= 1) & local_gate).sum())
        totals["coactivity_count"] += int(local_gate.size)
    def mean_repeat(values):
        return values.reshape(runs, repeats, -1).mean(axis=1)
    drive = kc.mean(axis=1) @ layout.projection(efficacy)
    pools_hz = layout.motor.pool_activity(output)[:, layout.card_pools]
    features = {"connected_kc": kc[:, :, layout.connected_kc].mean(axis=1),
        "clipped_connected_kc": mean_repeat(pre_scaled[:, layout.connected_kc]),
        "afferent_all": drive, "afferent_card": drive[:, layout.card_posts],
        "mbon_card": mbon[:, :, layout.card_posts].mean(axis=1),
        "card_pool": mean_repeat(pools_hz), "eligible_card_edges": np.concatenate(eligible),
        "gated_card_edges": np.concatenate(gated)}
    card_exposure = np.zeros_like(card_pool_selection, dtype=float)
    for slot, pool in enumerate(layout.card_pools):
        mask = np.isin(layout.post, layout.pool_posts[pool])
        card_exposure[:, slot] = gate[:, mask].mean(axis=1)
    # Per-run diagnostics retain repeats as repeated measurements, not samples.
    by_run = {"selected_card_fraction": mean_repeat(card_pool_selection.astype(float)),
              "card_edge_admitted_fraction": mean_repeat(card_exposure),
              "aliased_edge_count": aliased.reshape(runs, repeats, -1).sum(axis=(1, 2)),
              "aliased_nonzero_edge_count": np.concatenate(aliased_nonzero_by_run),
              "admitted_card_edge_count": card_gate.reshape(runs, repeats, -1).sum(axis=(1, 2))}
    diagnostics = {"action_histogram": np.bincount(actions["action_type"], minlength=13).tolist(),
        "card_count_histogram": np.bincount(actions["n_cards"], minlength=6).tolist(),
        "slot_selected_fraction": card_pool_selection.mean(axis=0).tolist(),
        "slot_edge_admitted_fraction": card_exposure.mean(axis=0).tolist(),
        "aliased_fraction_of_admitted_card_edges": float(aliased.sum() / max(1, card_gate.sum())),
        "trials_with_credit_to_unchosen_card_output_fraction": float(np.mean(aliased.any(axis=1))),
        "aliased_fraction_of_nonzero_card_increments": totals["aliased_nonzero"] / max(1, totals["nonzero"]),
        "aliased_fraction_of_card_increment_mass": aliased_mass / coactivity_mass if coactivity_mass else 0.,
        "card_edge_admission_fraction": totals["admitted"] / max(1, totals["coactivity_count"]),
        "nonzero_coactivity_fraction_among_admitted": totals["nonzero"] / max(1, totals["admitted"]),
        "pre_saturated_fraction_among_admitted": totals["pre_saturated"] / max(1, totals["admitted"]),
        "post_saturated_fraction_among_admitted": totals["post_saturated"] / max(1, totals["admitted"]),
        "both_saturated_fraction_among_admitted": totals["both_saturated"] / max(1, totals["admitted"])}
    return features, diagnostics, by_run, actions


def weight_summary(layout, initial, trained):
    delta = np.asarray(trained) - np.asarray(initial)
    result = {}
    masks = {"all": np.ones(len(delta), dtype=bool), "card": np.isin(np.arange(len(delta)), layout.card_edges)}
    for label, mask in masks.items():
        values = delta[mask]
        result[label] = {"edges": len(values), "changed_fraction": float(np.mean(values != 0)),
                         "mean_delta": float(values.mean()), "mean_absolute_delta": float(np.abs(values).mean()),
                         "minimum_delta": float(values.min()), "maximum_delta": float(values.max())}
    return result
