"""Visible-state demonstrations and native Small-Blind card-play evaluation."""
import numpy as np

from card_learning.pairs import pair_slots
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch, validate_strict_action_batch
from flylatro.seeds import derive_seed
from run_card_membership import fixed_motor
from run_training_bottlenecks import best_visible_play
import train_discard_play_curriculum as delayed

STAGES = ('higher_card', 'pair_member', 'pair', 'best_hand', 'small_blind')


def visible_teacher(observations):
    actions = empty_action_batch(len(observations['hand_len']))
    actions['action_type'][:] = 0
    for row, size in enumerate(observations['hand_len']):
        _, cards = best_visible_play(observations['hand'][row, :int(size)])
        actions['n_cards'][row] = len(cards); actions['cards'][row, :len(cards)] = cards
    return actions


def opening_batch(corpus, indices, stage, *, key, correct_side=None):
    if stage not in STAGES[:-1]:
        raise ValueError('opening curriculum stage required')
    n = len(indices); env = BalatroSimAdapter(n, win_ante=1, allow_skip_blinds=False); env.reset([0]*n)
    orders = []
    for row, index in enumerate(indices):
        corpus.restore(env, row, index)
        orders.append(np.r_[np.random.default_rng(derive_seed('local-teacher-display-v1', key, index)).permutation(8), 8, 9])
    obs, masks = delayed.permute_rows(*env.observe(), orders); masks = delayed.forced(masks, 0)
    target = empty_action_batch(n); target['action_type'][:] = 0
    if stage in ('higher_card', 'pair_member'):
        masks['card_select_mask'][:] = False
        # Put the two candidates in slots 0/1, counterbalancing their correct
        # side across examples. The remaining six visible cards supply context.
        for row, index in enumerate(indices):
            rng = np.random.default_rng(derive_seed('local-teacher-candidates-v1', key, index))
            if stage == 'higher_card':
                ranks = obs['hand'][row, :8, :13].argmax(axis=1)+2
                points = np.where(ranks == 14, 11, np.minimum(ranks, 10))
                candidates = [(i, j) for i in range(8) for j in range(i+1, 8) if points[i] != points[j]]
                if not candidates:
                    raise ValueError('higher-card curriculum needs unequal visible chip values')
                i, j = candidates[int(rng.integers(len(candidates)))]; good, bad = (i, j) if points[i] > points[j] else (j, i)
            else:
                pair = pair_slots(obs, row); good = int(rng.choice(pair))
                bad = int(rng.choice([i for i in range(8) if i not in pair]))
            correct = int(derive_seed('local-teacher-correct-side-v1', key, index) % 2) if correct_side is None else int(correct_side)
            if correct not in (0, 1):
                raise ValueError('binary teacher side must be zero or one')
            first = [good, bad] if correct == 0 else [bad, good]
            rem = [i for i in range(8) if i not in first]; rng.shuffle(rem)
            permutation = np.r_[first, rem, 8, 9]
            obs['hand'][row] = obs['hand'][row, permutation]
            orders[row] = orders[row][permutation]
            masks['card_select_mask'][row, :2] = True
            target['n_cards'][row] = 1; target['cards'][row, 0] = correct
    elif stage == 'pair':
        target['n_cards'][:] = 2
        for row in range(n):
            target['cards'][row, :2] = pair_slots(obs, row)
    else:
        target = visible_teacher(obs)
    validate_strict_action_batch(target, masks)
    return obs, masks, target, orders


def decode_stage(agent, activity, masks, stage):
    # No teacher/target argument exists in this inference function.
    if stage in ('higher_card', 'pair_member', 'pair'):
        return fixed_motor(agent, 0, 2 if stage == 'pair' else 1).decode(activity, masks, deterministic=True)
    return agent.motor.decode(activity, delayed.forced(masks, 0), deterministic=True)


def shuffle_instruction(target, stage, ids, *, key):
    result = {k: v.copy() for k, v in target.items()}
    for row, replicate in enumerate(ids):
        rng = np.random.default_rng(derive_seed('local-teacher-shuffled-instruction-v1', stage, key, replicate))
        count = int(target['n_cards'][row]); result['cards'][row] = -1
        legal = 2 if stage in ('higher_card', 'pair_member') else 8
        result['cards'][row, :count] = np.sort(rng.choice(legal, count, replace=False))
    return result


def collect_blind_demonstrations(seeds):
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    observations, masks, actions, source_seeds = [], [], [], []
    clears = 0
    for seed in seeds:
        env.reset([int(seed)])
        start = empty_action_batch(1); start['action_type'][:] = 2; start['n_cards'][:] = 0; env.step(start)
        for _ in range(8):
            obs, legal = env.observe()
            if not legal['action_type_mask'][0, 0]:
                raise ValueError('native demonstration left first-blind PLAY context')
            teacher = visible_teacher(obs)
            observations.append(obs); masks.append(delayed.forced(legal, 0)); actions.append(teacher); source_seeds.append(int(seed))
            outcome = env.step(teacher)
            if outcome.infos[0].get('blind_cleared', False) or outcome.infos[0]['reward_components'].get('blind_clear', 0):
                clears += 1; break
            if outcome.dones[0]:
                break
        else:
            raise ValueError('native first blind exceeded demonstration decision cap')
    return ({k: np.concatenate([o[k] for o in observations]) for k in observations[0]},
            {k: np.concatenate([m[k] for m in masks]) for k in masks[0]},
            {k: np.concatenate([a[k] for a in actions]) for k in actions[0]},
            {'seeds': list(map(int, seeds)), 'source_seed': source_seeds, 'teacher_clears': clears, 'games': len(seeds)})


def evaluate_small_blind(agent, rule, seeds, ids, *, key, progress=None):
    """Fly alone chooses cards/count; PLAY-only curriculum, no teacher inference."""
    n = len(ids); env = BalatroSimAdapter(n, win_ante=1, allow_skip_blinds=False)
    helper = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False)
    rows = []
    for number, seed in enumerate(seeds):
        helper.reset([int(seed)]); start = empty_action_batch(1); start['action_type'][:] = 2; start['n_cards'][:] = 0; helper.step(start)
        snapshot = helper.snapshot(0); env.reset([0]*n)
        for lane in range(n):
            env.restore(lane, snapshot)
        active = np.ones(n, dtype=bool); cleared = np.zeros(n, dtype=bool); scores = np.zeros(n); counts = np.zeros(n, dtype=int)
        trajectory = []
        for step in range(8):
            for lane in np.flatnonzero(~active):
                env.restore(int(lane), snapshot)
            obs, masks = env.observe(); masks = delayed.forced(masks, 0)
            neural = agent.processor.process(obs, fly_seeds=[derive_seed('local-teacher-blind-eval-v1', key, seed, r, step) for r in ids],
                                             efficacy=rule.state.efficacy)
            action = decode_stage(agent, neural.output_activity, masks, 'small_blind')
            validate_strict_action_batch(action, masks)
            outcome = env.step(action)
            trajectory.append({'active': active.tolist(), 'cards': action['cards'].tolist(), 'counts': action['n_cards'].tolist(),
                'progress': [info['reward_components']['blind_progress'] for info in outcome.infos],
                'cleared': [bool(info.get('blind_cleared', False) or info['reward_components'].get('blind_clear', 0)) for info in outcome.infos],
                'dones': outcome.dones.tolist()})
            for lane in np.flatnonzero(active):
                info = outcome.infos[int(lane)]; component = info['reward_components']
                counts[lane] += 1; scores[lane] += component['blind_progress']
                if info.get('blind_cleared', False) or component.get('blind_clear', 0):
                    cleared[lane] = True; active[lane] = False
                elif outcome.dones[lane]:
                    active[lane] = False
            if not active.any():
                break
        if active.any():
            raise ValueError('native first blind exceeded evaluation decision cap')
        rows.append({'seed': int(seed), 'cleared': cleared.tolist(), 'progress': scores.tolist(), 'plays': counts.tolist(), 'trajectory': trajectory})
        if progress and (number+1) % 8 == 0:
            progress({'phase': 'teacher-free-small-blind-evaluation', 'games': number+1})
    return {'scope': 'native first Small Blind; agent controls card count/membership; PLAY forced, no discards or shop decisions',
        'games': len(rows), 'rows': rows,
        'clear_rate_per_lane': np.asarray([r['cleared'] for r in rows]).mean(axis=0).tolist(),
        'mean_progress_per_lane': np.asarray([r['progress'] for r in rows]).mean(axis=0).tolist()}
