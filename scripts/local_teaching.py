"""Experimental synapse-local covariance plasticity with output teaching pulses.

No derivatives, optimizer, fitted decoder, critic or feedback-weight matrix.
Only existing anatomical KC->MBON efficacy values are persistent parameters.
Presynaptic means are local activity traces, calibrated without teaching labels.
This exact equation is an engineering model, not an established fly mechanism.
"""
from dataclasses import asdict, dataclass

import numpy as np

from flylatro.fly.mushroom_body.state import state_numpy


@dataclass(frozen=True)
class LocalTeachingConfig:
    version: str = 'local-covariance-teaching-v1'
    learning_rate: float = .05
    kc_reference_hz: float = 20.
    mbon_reference_hz: float = 20.
    centering_strength: float = .99
    mean_update_rate: float = .01
    eligibility_decay: float = 0.
    min_efficacy: float = .2
    max_efficacy: float = 2.
    maximum_step: float = .01

    def __post_init__(self):
        if self.version != 'local-covariance-teaching-v1':
            raise ValueError('unsupported local teaching rule')
        if not np.isfinite(list(asdict(self).values())[1:]).all():
            raise ValueError('finite local rule parameters required')
        if self.learning_rate <= 0 or min(self.kc_reference_hz, self.mbon_reference_hz, self.maximum_step) <= 0:
            raise ValueError('positive local rate, references and step bound required')
        if not (0 <= self.centering_strength <= 1 and 0 < self.mean_update_rate <= 1 and 0 <= self.eligibility_decay <= 1):
            raise ValueError('invalid activity trace parameters')
        if not 0 <= self.min_efficacy < self.max_efficacy:
            raise ValueError('invalid efficacy bounds')


class LocalTeachingPlasticity:
    def __init__(self, topology, state, config=None):
        self.topology, self.state = topology, state
        self.config = config or LocalTeachingConfig()
        if topology.edge_count != state.edge_count:
            raise ValueError('local rule topology and state differ')
        efficacy = state_numpy(state.efficacy)
        if not np.isfinite(efficacy).all() or np.any(efficacy < self.config.min_efficacy) or np.any(efficacy > self.config.max_efficacy):
            raise ValueError('initial local efficacies outside finite bounds')
        self.torch = bool(getattr(state, 'is_torch', False))
        if self.torch:
            import torch
            self.mean = torch.zeros_like(state.efficacy)
            self.calibration_count = torch.zeros_like(state.decision_count)
        else:
            self.mean = np.zeros_like(state.efficacy)
            self.calibration_count = np.zeros_like(state.decision_count)

    def _array(self, values, *, integer=False):
        if self.torch:
            import torch
            return torch.as_tensor(values, device=self.state.device, dtype=torch.int64 if integer else torch.float32)
        return np.asarray(values, dtype=np.int64 if integer else np.float32)

    def _finite(self, value):
        if self.torch:
            import torch
            return bool(torch.isfinite(value).all())
        return bool(np.isfinite(value).all())

    def _clip(self, value, low, high):
        return value.clamp(low, high) if self.torch else np.clip(value, low, high)

    def observe(self, pre_hz, post_hz, *, calibrate=False):
        if self.torch:
            import torch
            with torch.no_grad():
                return self._observe(pre_hz, post_hz, calibrate=calibrate)
        return self._observe(pre_hz, post_hz, calibrate=calibrate)

    def _observe(self, pre_hz, post_hz, *, calibrate):
        pre, post = self._array(pre_hz), self._array(post_hz)
        if pre.shape != self.state.efficacy.shape or post.shape != pre.shape or not self._finite(pre) or not self._finite(post):
            raise ValueError('finite edge-aligned local activities required')
        if bool((pre < 0).any()) or bool((post < 0).any()):
            raise ValueError('nonnegative firing rates required')
        # Smooth bounded compression retains distinctions above the old hard
        # reference cutoff. Every operation uses this synapse's two endpoints.
        x = pre/(pre+self.config.kc_reference_hz)
        y = post/(post+self.config.mbon_reference_hz)
        if calibrate:
            self.calibration_count += 1
            self.mean += (x-self.mean)/self.calibration_count[:, None]
            self.state.eligibility[...] = 0
            return
        if bool((self.calibration_count == 0).any()):
            raise ValueError('label-free activity calibration required before teaching')
        self.state.eligibility[...] = self._clip(self.config.eligibility_decay*self.state.eligibility
            + (x-self.config.centering_strength*self.mean)*y, -1., 1.)
        self.mean += self.config.mean_update_rate*(x-self.mean)
        self.state.decision_count += 1

    def teach(self, pool_signals, edge_pool):
        if self.torch:
            import torch
            with torch.no_grad():
                return self._teach(pool_signals, edge_pool)
        return self._teach(pool_signals, edge_pool)

    def _teach(self, pool_signals, edge_pool):
        signals, ids = self._array(pool_signals), self._array(edge_pool, integer=True)
        if (signals.ndim != 2 or signals.shape[0] != self.state.learners or ids.shape != (self.topology.edge_count,)
                or not self._finite(signals) or bool((abs(signals) > 1).any())
                or bool((ids < -1).any()) or bool((ids >= signals.shape[1]).any())):
            raise ValueError('bounded neuron teaching pulses and aligned edge routing required')
        routed = signals[:, self._clip(ids, 0, signals.shape[1]-1)]*(ids >= 0)[None, :]
        change = self._clip(self.config.learning_rate*self.state.eligibility*routed,
                            -self.config.maximum_step, self.config.maximum_step)
        proposed = self.state.efficacy+change
        low, high = proposed < self.config.min_efficacy, proposed > self.config.max_efficacy
        updated = self._clip(proposed, self.config.min_efficacy, self.config.max_efficacy)
        delta = updated-self.state.efficacy
        self.state.efficacy[...] = updated
        self.state.lower_bound_hits += low.sum(axis=1)
        self.state.upper_bound_hits += high.sum(axis=1)
        self.state.update_count += (delta != 0).sum(axis=1)
        self.state.eligibility[...] = 0
        self.state.dopamine[...] = 0
        return {'update_l1': state_numpy(abs(delta).sum(axis=1)).tolist(),
            'potentiated': state_numpy((delta > 0).sum(axis=1)).tolist(),
            'depressed': state_numpy((delta < 0).sum(axis=1)).tolist(),
            'bound_hits': state_numpy((low | high).sum(axis=1)).tolist()}

    def state_dict(self):
        return {**{k: v.copy() for k, v in self.state.state_dict().items()},
                'presynaptic_mean': state_numpy(self.mean).copy(),
                'calibration_count': state_numpy(self.calibration_count).copy()}

    def reset_activity_calibration(self):
        self.mean[...] = 0; self.calibration_count[...] = 0
        self.state.eligibility[...] = 0; self.state.dopamine[...] = 0

    def load_traces(self, values):
        mean, count = self._array(values['presynaptic_mean']), self._array(values['calibration_count'], integer=True)
        if mean.shape != self.mean.shape or count.shape != self.calibration_count.shape or not self._finite(mean):
            raise ValueError('checkpoint local traces differ')
        if bool((mean < 0).any()) or bool((mean > 1).any()) or bool((count < 0).any()):
            raise ValueError('invalid checkpoint local traces')
        self.mean[...] = mean; self.calibration_count[...] = count


def edge_pool_indices(mapping, topology, heads=('card', 'card_count')):
    result = np.full(topology.edge_count, -1, dtype=np.int64)
    for head in heads:
        for pool in mapping.routing.head_routes[head]:
            if pool < 0:
                continue
            roots = mapping.output_root_ids[list(mapping.pool_indices[pool])]
            edges = np.isin(topology.post_root_ids, roots)
            if np.any(result[edges] >= 0):
                raise ValueError('teaching motor populations must not overlap')
            result[edges] = pool
    return result


def teaching_signals(mapping, target, actual, *, teach_count=False):
    """Neuron-specific teaching pulses; labels never enter sensory processing."""
    n = len(actual['action_type']); signals = np.zeros((n, mapping.routing.pool_count), dtype=np.float32)
    for actions, sign in ((target, 1.), (actual, -1.)):
        if np.shape(actions['action_type']) != (n,) or np.any(actions['action_type'] != 0):
            raise ValueError('local card-play curriculum requires PLAY actions')
        for row in range(n):
            count = int(actions['n_cards'][row]); cards = actions['cards'][row, :count]
            if not 1 <= count <= 5 or len(set(cards.tolist())) != count or np.any(cards < 0) or np.any(cards >= 10):
                raise ValueError('valid distinct teacher and actual card choices required')
            signals[row, np.asarray(mapping.routing.head_routes['card'])[cards]] += sign
            if teach_count:
                signals[row, mapping.routing.head_routes['card_count'][count]] += sign
    return signals
