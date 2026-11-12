"""Summarize paired screens and independently replay recorded native choices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from card_learning.common import write_json
from card_learning.pairs import PairCorpus
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import derive_seed
from run_training_bottlenecks import VERSION
import train_discard_play_curriculum as delayed


def summarize(report):
    n=len(report['plastic']);plastic=np.asarray(report['plastic'])
    treatments={}
    policies=('binary',) if report['study']=='binary' else ('greedy','training-sampled')
    for policy in policies:
        before=report['before'] if policy=='binary' else report['before'][policy]
        after=report['after'] if policy=='binary' else report['after'][policy]
        metrics=('rewarded_probability','greedy_correct','sampled_correct') if policy=='binary' else (
            ('pair_correct','progress') if report['study']=='exploration' else
            ('progress','clears','pair_kept','discard_counts','random_same_count_progress'))
        treatments[policy]={}
        for treatment in range(2):
            lanes=np.arange(n)[(np.arange(n)%4==2*treatment)&plastic]
            values={}
            for metric in metrics:
                a=np.asarray([r[metric] for r in before],dtype=float)[:,lanes]
                b=np.asarray([r[metric] for r in after],dtype=float)[:,lanes]
                if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite outcomes')
                values[metric]={'before_per_learner':a.mean(axis=0).tolist(),
                    'after_per_learner':b.mean(axis=0).tolist(),
                    'change_per_learner':(b-a).mean(axis=0).tolist(),
                    'mean_before':float(a.mean()),'mean_after':float(b.mean()),
                    'mean_change':float((b-a).mean())}
            if report['study']=='followup':
                p=np.asarray([r['progress'] for r in after])[:,lanes]
                q=np.asarray([r['random_same_count_progress'] for r in after])[:,lanes]
                values['post_training_progress_minus_random_same_count']=(p-q).mean(axis=0).tolist()
            treatments[policy][str(treatment)]=values
    return treatments


def replay(report):
    if report['study']=='binary':return {'native_actions':0,'scope':'no native transitions in binary diagnostic'}
    corpus=PairCorpus(Path(report['corpus']))
    if corpus.sha256!=report['corpus_sha256']:raise ValueError('corpus changed')
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([0])
    verified=0;ids=report['replicate_ids']
    for stage in ('before','after'):
        for rows in report[stage].values():
            for row in rows:
                for lane,index in enumerate(row['case_indices']):
                    key=row['key'];replicate=ids[lane]
                    if report['study']=='exploration':
                        seed=derive_seed(VERSION,'pair',key,replicate,index,'display')
                    else:
                        seed=derive_seed(delayed.VERSION,(VERSION,'followup',key),replicate,index,'display')
                    rng=np.random.default_rng(seed)
                    order=np.r_[rng.permutation(8),8,9]
                    corpus.restore(env,0,index)
                    if report['study']=='exploration':
                        action=empty_action_batch(1);action['action_type'][:]=0;action['n_cards'][:]=2
                        action['cards'][0,:2]=order[row['cards'][lane][:2]]
                    else:
                        next_order=np.r_[rng.permutation(8),8,9]
                        action=empty_action_batch(1);action['action_type'][:]=1
                        count=row['discard_counts'][lane];action['n_cards'][:]=count
                        action['cards'][0,:count]=order[row['discard_cards'][lane][:count]]
                        discarded=env.step(action);verified+=1
                        if discarded.dones[0] or discarded.rewards[0]!=0:raise ValueError('discard replay differs')
                        cards=[c for c in row['play_cards'][lane] if c>=0]
                        action=empty_action_batch(1);action['action_type'][:]=0;action['n_cards'][:]=len(cards)
                        action['cards'][0,:len(cards)]=next_order[cards]
                    result=env.step(action);verified+=1
                    native=result.infos[0]['reward_components']['blind_progress']
                    if result.dones[0] or abs(native-row['progress'][lane])>2e-6:
                        raise ValueError('native action/progress replay differs')
    return {'native_actions':verified,'state_restored_from_hash_verified_corpus':True,
            'recorded_action_progress_and_nonterminal_verified':True}


def verify_state(report,path):
    with np.load(path,allow_pickle=False) as data:
        efficacy,initial=data['efficacy'],data['initial_efficacy']
        plastic=np.asarray(report['plastic'],dtype=bool)
        if efficacy.shape!=initial.shape or efficacy.shape[0]!=len(plastic):
            raise ValueError('diagnostic state shape differs')
        if not np.isfinite(efficacy).all() or np.any(initial!=1):
            raise ValueError('diagnostic state is nonfinite or did not start at unit efficacy')
        rule=report['learning_rule']
        if np.any(efficacy<rule['min_efficacy']) or np.any(efficacy>rule['max_efficacy']):
            raise ValueError('diagnostic efficacy out of bounds')
        np.testing.assert_array_equal(efficacy[~plastic],initial[~plastic])
        delta=np.abs(efficacy-initial)
        np.testing.assert_allclose(delta.mean(axis=1),report['mean_absolute_drift'],atol=1e-9)
        np.testing.assert_allclose(delta.max(axis=1),report['max_absolute_drift'],atol=1e-9)
        for key in ('eligibility','dopamine','lower_bound_hits','upper_bound_hits'):
            if data[key].any():raise ValueError(f'nonzero final {key}')
        expected=report['steps']*(2 if report['study']=='followup' else 1)
        if np.any(data['decision_count']!=expected):raise ValueError('decision count differs')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args(argv);path=args.run_dir/'report.json';report=json.loads(path.read_text())
    if report['status']!='completed':raise ValueError('completed run required')
    for source,digest in report['source_hashes'].items():
        if sha256_file(Path(source))!=digest:raise ValueError(f'source changed: {source}')
    verify_state(report,args.run_dir/'diagnostic-final-state.npz')
    result={'study':report['study'],'status':'completed','source_report_sha256':sha256_file(path),
        'assessment_source_sha256':sha256_file(Path(__file__)),'diagnostic_state_verified':True,
        'final_state_sha256':sha256_file(args.run_dir/'diagnostic-final-state.npz'),
        'summary':summarize(report),'replay':replay(report),
        'limits':'descriptive screen with three learner seeds on one mapping; shared hands/repeats are not independent training replicates; binary evaluates fixed training states; reference PLAY is diagnostic, not a deployable learned policy'}
    write_json(args.run_dir/'assessment.json',result)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
