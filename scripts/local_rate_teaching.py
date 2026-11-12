"""Neuron-local rate-target teaching; no network derivatives or optimizer.

The output neuron receives a synthetic desired firing level during training.
The instruction minus its own measured activity sets the third-factor pulse.
Inference has no instruction and uses the unchanged fixed neural readout.
"""
from dataclasses import asdict, dataclass

import numpy as np

from local_teaching import LocalTeachingConfig
from flylatro.interface.motor import FixedMotorInterface


@dataclass(frozen=True)
class LocalRateTeachingConfig(LocalTeachingConfig):
    version: str = 'local-rate-target-teaching-v1'

    def __post_init__(self):
        if self.version != 'local-rate-target-teaching-v1':
            raise ValueError('unsupported local rate teaching rule')
        LocalTeachingConfig(**{**asdict(self), 'version': 'local-covariance-teaching-v1'})


def rate_teaching_signals(mapping, target, actual, *, activity, masks, teach_count=False, decision_ms=100.):
    """Each output compares its own rate with a fixed teacher-specified level.

    Calibration baseline/scales are fixed before learning. Two scale units or
    one spike per decision, whichever is larger, supply the target margin;
    no learned decoder or gradient
    conveys the instruction. Illegal/uninterpreted output options receive zero.
    """
    rates = FixedMotorInterface(mapping).pool_activity(activity)
    if not np.isfinite(decision_ms) or decision_ms <= 0:
        raise ValueError('positive decision window required')
    n = len(target['action_type']); result = np.zeros_like(rates, dtype=np.float32)
    if rates.shape != (n, mapping.routing.pool_count) or np.any(target['action_type'] != 0) or np.any(actual['action_type'] != 0):
        raise ValueError('aligned PLAY activity and teaching actions required')
    for row in range(n):
        cards = target['cards'][row, :int(target['n_cards'][row])]
        if not 1 <= int(target['n_cards'][row]) <= 5 or len(cards) < 1 or len(set(cards.tolist())) != len(cards) or np.any(cards < 0) or np.any(cards >= 10):
            raise ValueError('valid visible teaching cards required')
        if not masks['card_select_mask'][row, cards].all():
            raise ValueError('teacher chose an illegal card')
        selected = set(cards.tolist())
        for option, pool in enumerate(mapping.routing.head_routes['card']):
            if pool < 0 or not masks['card_select_mask'][row, option]:
                continue
            scale = float(mapping.normalization.scale[pool]); margin = max(2*scale, 1000/decision_ms)
            desired = np.clip(mapping.normalization.baseline[pool]+margin*(1 if option in selected else -1), 0., 200.)
            result[row, pool] = np.clip((desired-rates[row, pool])/max(margin, .5), -1., 1.)
        if teach_count:
            legal_count = min(5, int(masks['card_select_mask'][row].sum()))
            for option, pool in enumerate(mapping.routing.head_routes['card_count']):
                if pool < 0 or not 1 <= option <= legal_count:
                    continue
                scale = float(mapping.normalization.scale[pool]); margin = max(2*scale, 1000/decision_ms)
                desired = np.clip(mapping.normalization.baseline[pool]+margin*(1 if option == int(target['n_cards'][row]) else -1), 0., 200.)
                result[row, pool] = np.clip((desired-rates[row, pool])/max(margin, .5), -1., 1.)
    return result
