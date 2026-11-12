"""Fixed-five-card native factorial pilot; anatomical local efficacy only."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import time
import numpy as np

from card_learning.common import write_json, identity
from card_learning.motor import install_motor, CardExploration
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.credit import SelectedMotorCredit
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.fly.flywire_artifact import sha256_file
from local_normalisation import NormalisedThreeFactorPlasticity
from run_card_membership import fixed_motor
from strategy_puzzles import action
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.seeds import derive_seed

VERSION = 'exchange-local-credit-v1'
ARMS = ['subset-selected', 'swap-selected', 'subset-signed-centred', 'swap-signed-centred', 'frozen']


def explore(cards, rng, swap):
    cards = np.asarray(cards, dtype=int).copy()
    if rng.random() >= .05: return cards
    if swap:
        index = int(rng.integers(5))
        cards[index] = rng.choice(np.setdiff1d(np.arange(8), cards))
        return np.sort(cards)
    return np.sort(rng.choice(8, 5, replace=False))


def signed_tags(chosen, greedy):
    tags = np.zeros((len(chosen), 10), dtype=np.float32)
    for row, (c, g) in enumerate(zip(chosen, greedy, strict=True)):
        tags[row, c] += 1
        tags[row, g] -= 1
    return tags


def run(output, steps=2000, replicates=3, repeats=16):
    if min(steps, replicates, repeats) < 1: raise ValueError('positive counts required')
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path('artefacts/learnable-motor-20261028-v2/config.toml')
    config = PlasticExperimentConfig.load(config_path)
    n = len(ARMS) * replicates
    config = replace(config, environment=replace(config.environment, num_envs=n))
    declaration = {'version': VERSION, 'steps': steps, 'replicates': replicates, 'arms': ARMS,
        'evaluation_repeats': repeats, 'implementation': identity(),
        'script_sha256': sha256_file(Path(__file__)), 'config_sha256': sha256_file(config_path),
        'fixed_count': 5, 'training_seed': 4000, 'fresh_unit_weights': True,
        'exploration_probability': .05, 'baseline_rate': .01,
        'signed_rule': 'clipped pre/post coactivity times entering-minus-leaving card-post tag; outcome minus prior local reward EMA',
        'limits': 'one fixed opening/display, fresh weights; signed credit and centring are one compound treatment, not separately attributed; no policy gradient claim; no terminal or strategy training'}
    write_json(output/'declaration.json', declaration)
    stack = install_motor(build_plastic_stack(config), CardExploration(0))
    agent = stack.agent
    local = NormalisedThreeFactorPlasticity(agent.plasticity.topology, agent.plasticity.state, agent.plasticity.config)
    motor = fixed_motor(agent, 0, 5)
    credit = SelectedMotorCredit(motor, local)
    roles = np.tile(np.arange(5), replicates); ids = np.repeat(np.arange(replicates), 5)
    env = BalatroSimAdapter(n, win_ante=1, allow_skip_blinds=False)
    env.reset([4000]*n); env.step(action(2, batch=n)); obs, masks = env.observe()
    opening = env.snapshot_all()
    baseline = np.zeros(n); baseline_ready = False
    update_total = np.zeros(n); discoveries = np.zeros(n, dtype=int)
    card_routes = motor.mapping.routing.head_routes['card']
    # Each card slot's tag is broadcast only to its own anatomical post edges.
    slot_edges = np.stack([credit.pool_posts[p][credit.edge_posts] for p in card_routes])
    if np.any(slot_edges.sum(axis=0) > 1): raise ValueError('signed slot credit requires distinct card posts')
    def responses(observations, key, initial=False):
        seeds = [derive_seed(VERSION, key, int(rep)) for rep in ids]
        neural = agent.processor.process(observations, fly_seeds=seeds,
            efficacy=local.state.initial_efficacy if initial else local.state.efficacy)
        return neural, motor.decode(neural.output_activity, masks, deterministic=True)
    def evaluate(initial=False):
        import torch
        before = local.state.weight_sha256
        rows = []
        for label, seeds in [('training', [4000]*repeats), ('unseen', list(range(20920000,20920000+repeats)))]:
            chips = []
            for repeat, seed in enumerate(seeds):
                env.reset([seed]*n); env.step(action(2, batch=n)); o,_ = env.observe()
                _, a = responses(o, ('eval', label, repeat), initial)
                result = env.step(a)
                if np.any(result.dones): raise ValueError('opening PLAY unexpectedly terminated')
                chips.append([info['reward_components']['blind_progress']*300 for info in result.infos])
            values = np.array(chips)
            rows.append({'split': label, 'mean_capped_chips': values.mean(axis=0).tolist(),
                'scores': values.tolist(), 'scope': 'single opening five-card PLAY; chips capped at remaining target'})
        assert local.state.weight_sha256 == before
        return rows
    try:
        initial = evaluate(initial=True)
        env.restore_all(opening); obs, masks = env.observe()
        started = time.monotonic()
        with (output/'training.jsonl').open('w') as stream:
            for step in range(steps):
                neural, greedy = responses(obs, ('train', step))
                chosen = greedy['cards'][:, :5].copy()
                for lane in range(n):
                    rng = np.random.default_rng(derive_seed(VERSION, step, int(ids[lane]), 'explore'))
                    chosen[lane] = explore(chosen[lane], rng, roles[lane] in (1,3))
                actual = {k:v.copy() for k,v in greedy.items()}; actual['cards'][:,:5] = chosen
                env.restore_all(opening); result = env.step(actual)
                pulse = np.array([agent.reinforcement.map(i['reward_components'],i).appetitive for i in result.infos])
                if np.any(result.dones): raise ValueError('training opening unexpectedly terminated')
                tags = signed_tags(chosen, greedy['cards'][:,:5])
                gate = credit.mask(actual, masks)
                local.record_activity(neural.edge_pre_activity, neural.edge_post_activity, edge_credit=gate)
                signed = np.isin(roles, [2,3])
                # Recompute the same zero-history coactivity for signed lanes,
                # including the output of the card that was removed.
                import torch
                pre = torch.clamp(neural.edge_pre_activity/local.config.kc_reference_hz,0,1)
                post = torch.clamp(neural.edge_post_activity/local.config.mbon_reference_hz,0,1)
                edge_tags = torch.as_tensor(tags @ slot_edges.astype(np.float32),device=local.state.device)
                local.state.eligibility[signed] = (pre*post*edge_tags)[signed]
                delta = pulse.copy()
                if not baseline_ready: baseline[:] = pulse; baseline_ready = True
                delta[signed] -= baseline[signed]
                delta[roles == 4] = 0
                events = local.apply_dopamine(np.maximum(delta,0), np.maximum(-delta,0))
                baseline += .01 * (pulse-baseline)
                local.state.reset_fast_traces(eligibility=True)
                update_total += [e.absolute_change for e in events]
                discoveries += (pulse*300/config.reinforcement.progress_scale >= 91.99)
                stream.write(json.dumps({'step':step,'cards':chosen.tolist(),'greedy':greedy['cards'][:,:5].tolist(),
                    'native_pulses':pulse.tolist(),'modulators':delta.tolist(),'baseline_after':baseline.tolist()})+'\n')
                if (step+1)%250 == 0:
                    w=state_numpy(local.state.efficacy)
                    bound=np.mean((w<=local.config.min_efficacy)|(w>=local.config.max_efficacy),axis=1)
                    drift=np.mean(abs(w-state_numpy(local.state.initial_efficacy)),axis=1)
                    if np.max(bound)>.1 or np.max(drift)>.5: raise ValueError('safety limit')
                    print(f'{step+1}/{steps} trials per learner',flush=True)
        final = evaluate()
        state = local.state.state_dict()
        np.savez_compressed(output/'checkpoint.npz',**state,baseline=baseline)
        frozen=roles==4
        assert np.array_equal(state['efficacy'][frozen],state['initial_efficacy'][frozen])
        strengths=state_numpy(local.incoming_strengths()); budgets=state_numpy(local.budgets)
        error=np.max(abs(strengths-budgets)/budgets,axis=1)
        assert np.max(error)<1e-6
        report={'status':'completed','declaration':declaration,'initial':initial,'final':final,
            'roles':roles.tolist(),'replicate_ids':ids.tolist(),'update_l1':update_total.tolist(),
            'budget_error':error.tolist(),'training_best_hand_discoveries':discoveries.tolist(),
            'seconds':time.monotonic()-started,'checkpoint_sha256':sha256_file(output/'checkpoint.npz')}
        write_json(output/'report.json',report)
        return report
    except BaseException as error:
        write_json(output/'status.json',{'status':'failed','error':str(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--heavy',action='store_true');a=p.parse_args()
    if not a.heavy:p.error('requires --heavy')
    run(a.output_dir,a.steps)
