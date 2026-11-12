"""Versioned reward-free repair of width-one action-conditioned motor layouts.

Kept outside production source so existing hash-bound experiments still resume.
Only failing pools are replaced; candidate ranking uses neural response, never
reward or diagnostic learning results. Every replacement passes all old gates.
"""
from dataclasses import asdict, replace
from itertools import product

import numpy as np

from flylatro.interface.motor_calibration import (
    MotorCalibrationError, MotorCalibrationThresholds, _group_evidence,
    validate_reward_free_motor,
)
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.interface.separated_motor import normalize_fixed
from flylatro.learning.config import payload_sha256

VERSION = 'reward-free-active-plastic-input-motor-v1'
MINIMUM_COACTIVE_FRACTION = .5


def learning_support(mapping, coactivity, *, masks, order=None,
                     minimum_fraction=MINIMUM_COACTIVE_FRACTION):
    """Require fresh KC/post coactivity for each member of each competing option.

    coactivity is sample-by-canonical-output, summed over incoming plastic edges
    using the actual source rule's reference rates and eligibility clipping.
    Padding, disabled actions and forced-only options are explicitly unexercised.
    """
    mass = np.asarray(coactivity, dtype=np.float64)
    if (mass.ndim != 2 or mass.shape[1] != len(mapping.output_root_ids)
            or not np.isfinite(mass).all() or np.any(mass < 0)):
        raise ValueError('finite nonnegative sample-by-canonical-output coactivity required')
    if not np.isfinite(minimum_fraction) or not 0 < minimum_fraction <= 1:
        raise ValueError('minimum coactive fraction must be in (0, 1]')
    contexts = motor_context_windows(masks, order=order, routing=mapping.routing)
    eligible = {g: np.ones(mass.shape[1], dtype=bool) for g in mapping.routing.group_names}
    roles = []
    for window in contexts.values():
        if len(window.relevant) != len(mass):
            raise ValueError('coactivity samples and mask order differ')
        head = window.spec.head
        for option, pool in enumerate(mapping.routing.head_routes[head]):
            if pool < 0:
                continue
            selected = window.relevant & (window.option_legal.sum(axis=1) > 1) & window.option_legal[:, option]
            members = list(mapping.pool_indices[pool])
            fraction = (mass[selected] > 0).mean(axis=0) if selected.any() else None
            if fraction is not None:
                eligible[window.spec.group] &= fraction >= minimum_fraction
            passed = fraction is None or bool(np.all(fraction[members] >= minimum_fraction))
            roles.append({'context': window.name, 'head': head, 'option': option, 'pool': pool,
                'post_root_ids': mapping.output_root_ids[members].tolist(),
                'competing_samples': int(selected.sum()),
                'coactive_fractions': fraction[members].tolist() if fraction is not None else None,
                'status': ('PASS' if passed else 'FAIL') if fraction is not None else 'UNEXERCISED'})
    return {'version': VERSION, 'status': 'PASS' if all(r['status'] != 'FAIL' for r in roles) else 'FAIL',
        'minimum_coactive_fraction': minimum_fraction, 'roles': roles,
        'scope': 'fresh positive eligibility opportunity; does not prove a useful learned strategy'}, eligible


def repair_mapping(source, values, coactivity, *, masks, activity_order=None,
                   coactivity_order=None, metadata=None, maximum_candidates=256):
    """Allocate only failed pools, subject to old gates plus active-input gates.

    This migration deliberately supports the qualified width-one architecture.
    No automatic widening, threshold relaxation, or checkpoint weight migration.
    """
    if not source.routing.action_conditioned or any(len(p) != 1 for p in source.pool_indices):
        raise MotorCalibrationError('repair requires a declared width-one action-conditioned mapping')
    if maximum_candidates < 1:
        raise ValueError('positive candidate budget required')
    limits = MotorCalibrationThresholds(minimum_pool_width=1)
    contexts = motor_context_windows(masks, order=activity_order, routing=source.routing)
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or values.shape != (len(next(iter(contexts.values())).relevant), len(source.output_root_ids))
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('finite nonnegative aligned response activity required')
    before, support = learning_support(source, coactivity, masks=masks, order=coactivity_order)
    failed = sorted({r['pool'] for r in before['roles'] if r['status'] == 'FAIL'})
    evidence = _group_evidence(values, source.routing, contexts, limits, source.output_root_ids)
    used = {i for p in source.pool_indices for i in p}
    choices = []
    for pool in failed:
        group = source.routing.pool_groups[pool]
        # A shared context output must work in every exercised interpretation.
        allowed = support[group] & evidence[group]['eligible']
        choices.append([int(i) for i in evidence[group]['order'] if allowed[i] and int(i) not in used])
    attempts = []
    for trial, replacements in enumerate(product(*choices)):
        if trial >= maximum_candidates:
            break
        if len(set(replacements)) != len(replacements):
            continue
        pools = list(source.pool_indices)
        for pool, index in zip(failed, replacements, strict=True):
            pools[pool] = (index,)
        stats = {'version': VERSION, 'source_mapping_sha256': source.sha256,
            'minimum_coactive_fraction': MINIMUM_COACTIVE_FRACTION,
            'candidate_rank': trial, 'metadata': dict(metadata or {})}
        candidate = normalize_fixed(replace(source, pool_indices=tuple(pools), selection_method=VERSION), values, stats)
        stats = {**candidate.calibration_stats, **stats}
        candidate = replace(candidate, calibration_stats=stats, calibration_sha256=payload_sha256(stats))
        check = validate_reward_free_motor(candidate, values, thresholds=limits, contexts=contexts,
                                          calibration_metadata=metadata)
        after, _ = learning_support(candidate, coactivity, masks=masks, order=coactivity_order)
        if check.status == 'PASS' and after['status'] == 'PASS':
            return candidate, {'version': VERSION, 'reward_used': False, 'outcome_information_used': False,
                'thresholds': asdict(limits), 'qualification': check.report,
                'before': before, 'after': after, 'failed_candidates': attempts,
                'replacements': [{'pool': p, 'group': source.routing.pool_groups[p],
                    'old_root': int(source.output_root_ids[source.pool_indices[p][0]]),
                    'new_root': int(source.output_root_ids[i])} for p, i in zip(failed, replacements, strict=True)]}
        attempts.append({'rank': trial, 'gates': check.report['gates'], 'learning_support': after['status']})
    raise MotorCalibrationError('no repair passed original quality and active plastic input gates',
                                {'before': before, 'attempts': attempts, 'choices_per_failed_pool': list(map(len, choices))})
