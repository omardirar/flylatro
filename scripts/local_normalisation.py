"""Experimental local competition over anatomical KC->MBON inputs.

After the unchanged three-factor update, each affected postsynaptic neuron
redistributes its own incoming strengths within its initial weighted budget.
No gradient, teacher, fitted decoder or cross-neuron budget is used.
"""
from dataclasses import asdict, dataclass, replace
import hashlib

import numpy as np

from flylatro.fly.mushroom_body.plasticity import (
    PlasticityConfig, PlasticityEventDetail, ThreeFactorPlasticity,
)
from flylatro.fly.mushroom_body.state import state_numpy


VERSION = 'local-postsynaptic-strength-budget-v1'


@dataclass(frozen=True, slots=True)
class NormalisedPlasticityConfig(PlasticityConfig):
    normalisation_version: str = VERSION

    def __post_init__(self):
        PlasticityConfig.__post_init__(self)
        if self.normalisation_version != VERSION or self.min_efficacy <= 0:
            raise ValueError('supported normalisation and strictly positive efficacy floor required')
        if not np.isfinite([v for v in asdict(self).values() if isinstance(v, (float, int))]).all():
            raise ValueError('finite plasticity parameters required')


class NormalisedThreeFactorPlasticity(ThreeFactorPlasticity):
    """Per-MBON sum(anatomical_weight * efficacy) stays at its initial value.

    For provisional efficacies u, choose a positive multiplier s per learner
    and MBON such that sum(a * clip(s*u, low, high)) equals the initial budget.
    Untouched groups and no-update controls remain bit-identical. Uniform gain
    within a group is removed; relative input preferences can still change.
    """
    def __init__(self, topology, state, config=None, *, normalised_learners=None):
        base = config or PlasticityConfig()
        if not isinstance(base, NormalisedPlasticityConfig):
            base = NormalisedPlasticityConfig(**asdict(base))
        super().__init__(topology, state, base)
        initial = state_numpy(state.initial_efficacy)
        if (not np.isfinite(initial).all() or np.any(initial < base.min_efficacy)
                or np.any(initial > base.max_efficacy)):
            raise ValueError('initial budgets must come from finite bounded initial efficacies')
        self.posts, groups = np.unique(topology.post_root_ids, return_inverse=True)
        order = np.argsort(groups, kind='stable')
        self.counts = np.bincount(groups)
        self.starts = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.order, self.inverse = order, np.argsort(order)
        self.groups = groups[order]
        self.weights = topology.anatomical_weights[order].astype(np.float64)
        self.normalised_learners = np.ones(state.learners, dtype=bool) if normalised_learners is None else np.asarray(normalised_learners)
        if self.normalised_learners.shape != (state.learners,) or self.normalised_learners.dtype != np.bool_:
            raise ValueError('one immutable boolean normalisation assignment per learner required')
        self.normalised_learners = self.normalised_learners.copy()
        if self.is_torch:
            import torch
            device = state.device
            self.order = torch.as_tensor(self.order, device=device)
            self.inverse = torch.as_tensor(self.inverse, device=device)
            self.groups = torch.as_tensor(self.groups, device=device)
            self.weights = torch.as_tensor(self.weights, device=device)
            self.counts = torch.as_tensor(self.counts, device=device)
            self.normalised_learners = torch.as_tensor(self.normalised_learners, device=device)
            self.budgets = self._sum(state.initial_efficacy[:, self.order].to(torch.float64) * self.weights)
        else:
            self.budgets = self._sum(initial[:, self.order].astype(np.float64) * self.weights)

    def _sum(self, values):
        if self.is_torch:
            import torch
            return torch.segment_reduce(values, 'sum',
                lengths=self.counts[None, :].expand(values.shape[0], -1), axis=1)
        return np.add.reduceat(values, self.starts, axis=1)

    def incoming_strengths(self):
        values = self.state.efficacy[:, self.order]
        if self.is_torch:
            import torch
            values = values.to(torch.float64)
        else:
            values = values.astype(np.float64)
        return self._sum(values * self.weights)

    def _project(self, before):
        low, high = self.config.min_efficacy, self.config.max_efficacy
        current = self.state.efficacy[:, self.order]
        previous = before[:, self.order]
        if self.is_torch:
            import torch
            values = current.to(torch.float64)
            touched = (self._sum((current != previous).to(torch.float64)) > 0) & self.normalised_learners[:, None]
            total = self._sum(values * self.weights)
            scales = self.budgets / total
            candidate = values * scales[:, self.groups]
            needs_projection = (self._sum(((candidate < low) | (candidate > high)).to(torch.float64)) > 0) & touched
            bounded = not bool(needs_projection.any())
            clip = lambda x: x.clamp(low, high)
            where = torch.where
            zeros = torch.zeros_like(self.budgets)
            upper = torch.full_like(self.budgets, high / low)
        else:
            values = current.astype(np.float64)
            touched = (self._sum((current != previous).astype(np.float64)) > 0) & self.normalised_learners[:, None]
            scales = self.budgets / self._sum(values * self.weights)
            candidate = values * scales[:, self.groups]
            needs_projection = (self._sum(((candidate < low) | (candidate > high)).astype(np.float64)) > 0) & touched
            bounded = not bool(needs_projection.any())
            clip = lambda x: np.clip(x, low, high)
            where = np.where
            zeros = np.zeros_like(self.budgets)
            upper = np.full_like(self.budgets, high / low)
        if not bounded:
            # Monotone bounded projection; deterministic grouped reductions,
            # not unordered scatter atomics. No learner/neuron mixes budgets.
            lower = zeros
            for _ in range(36):
                middle = (lower + upper) / 2
                total = self._sum(clip(values * middle[:, self.groups]) * self.weights)
                below = total < self.budgets
                lower = where(below, middle, lower)
                upper = where(below, upper, middle)
            projected = clip(values * ((lower + upper) / 2)[:, self.groups])
            candidate = where(needs_projection[:, self.groups], projected, candidate)
        # Control lanes keep the unnormalised three-factor proposal exactly.
        candidate = where(touched[:, self.groups], candidate, values)
        if self.is_torch:
            self.state.efficacy.copy_(candidate[:, self.inverse].to(self.state.efficacy.dtype))
        else:
            self.state.efficacy[...] = candidate[:, self.inverse].astype(np.float32)

    def apply_dopamine(self, appetitive, aversive, *, plasticity_enabled=True, detail=False):
        if self.is_torch:
            import torch
            with torch.no_grad():
                return self._apply_normalised(appetitive, aversive, plasticity_enabled, detail)
        return self._apply_normalised(appetitive, aversive, plasticity_enabled, detail)

    def _apply_normalised(self, appetitive, aversive, enabled, detail):
        before = self.state.efficacy.clone() if self.is_torch else self.state.efficacy.copy()
        events = super().apply_dopamine(appetitive, aversive, plasticity_enabled=enabled, detail=detail)
        if not any(e.changed_synapses for e in events):
            return events
        self._project(before)
        delta = self.state.efficacy - before
        if self.is_torch:
            import torch
            changed = (delta != 0).sum(dim=1)
            summary = torch.stack((changed.to(torch.float64), delta.abs().sum(dim=1).to(torch.float64),
                delta.abs().amax(dim=1).to(torch.float64)), dim=1).cpu().numpy()
            old_changed = torch.as_tensor([e.changed_synapses > 0 for e in events], device=self.state.device)
            self.state.update_count.add_((changed > 0).to(torch.int64) - old_changed.to(torch.int64))
        else:
            changed = np.count_nonzero(delta, axis=1)
            summary = np.stack((changed, abs(delta).sum(axis=1), abs(delta).max(axis=1)), axis=1)
            self.state.update_count += (changed > 0) - np.asarray([e.changed_synapses > 0 for e in events], dtype=np.int64)
        details = [None] * len(events)
        if detail:
            a, b, differences = state_numpy(before), state_numpy(self.state.efficacy), state_numpy(delta)
            for row, difference in enumerate(differences):
                ids = np.flatnonzero(difference)
                digest = lambda value: hashlib.sha256(value.astype('<f4').tobytes()).hexdigest()
                details[row] = PlasticityEventDetail(digest(a[row]), digest(b[row]),
                    tuple(int(i) for i in ids), tuple(float(v) for v in difference[ids]))
        return tuple(replace(event, changed_synapses=int(row[0]), absolute_change=float(row[1]),
            maximum_absolute_change=float(row[2]), detail=extra)
            for event, row, extra in zip(events, summary, details, strict=True))


def install_normalisation(stack):
    """Research-only factory hook; checkpoint rule hash rejects old rules."""
    from flylatro.learning.credit import SelectedMotorCredit
    original = stack.agent.plasticity
    rule = NormalisedThreeFactorPlasticity(original.topology, original.state, original.config)
    stack.agent.plasticity = rule
    stack.agent.credit = SelectedMotorCredit(stack.agent.motor, rule)
    stack.components['plasticity_rule_sha256'] = rule.config.sha256
    stack.components['local_normalisation'] = {
        'version': VERSION, 'budget': 'initial sum(anatomical synapse count * efficacy) per learner and postsynaptic root',
        'competition_scope': 'only incoming edges of postsynaptic neurons affected by a native update',
        'external_trainable_parameter_count': 0,
        'initial_budgets_sha256': hashlib.sha256(state_numpy(rule.budgets).astype('<f8').tobytes()).hexdigest(),
    }
    return stack
