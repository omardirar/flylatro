"""Independent replay, saved-state checks and paired fixed-count outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assess_training_bottlenecks import verify_state
from card_learning.common import write_json
from card_learning.pairs import PairCorpus
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import derive_seed
from run_card_membership import VERSION,matched_gain


def summarize(report):
    corpus=PairCorpus(Path(report['corpus']));cases=len(corpus.indices('validation'))
    role=np.asarray(report['roles']);replicates=len(role)//4
    summary={}
    for policy in ('greedy','training-sampled'):
        summary[policy]={};a=report['before'][policy];b=report['after'][policy]
        repeats=len(a)//cases
        if len(a)!=len(b) or repeats*cases!=len(a):raise ValueError('case/repeat alignment differs')
        for r,label in enumerate(('neural-native','reference-native','reference-matched')):
            lanes=np.flatnonzero(role==r);result={}
            for followup in ('weak','reference'):
                before=np.asarray([row[followup+'_progress'] for row in a])[:,lanes]
                after=np.asarray([row[followup+'_progress'] for row in b])[:,lanes]
                frozen_before=np.asarray([row[followup+'_progress'] for row in a])[:,3::4]
                frozen_after=np.asarray([row[followup+'_progress'] for row in b])[:,3::4]
                np.testing.assert_array_equal(before,frozen_before)
                np.testing.assert_array_equal(frozen_before,frozen_after)
                difference=after-before
                per_case=difference.reshape(cases,repeats,replicates).mean(axis=(1,2))
                rng=np.random.default_rng(derive_seed(VERSION,'assessment',policy,r,followup))
                boot=per_case[rng.integers(0,cases,size=(2000,cases))].mean(axis=1)
                result[followup]={'mean_before':float(before.mean()),'mean_after':float(after.mean()),
                    'before_per_learner':before.mean(axis=0).tolist(),'after_per_learner':after.mean(axis=0).tolist(),
                    'change_per_learner':difference.mean(axis=0).tolist(),'mean_change':float(difference.mean()),
                    'case_bootstrap_95_interval_conditional_on_these_learners':np.quantile(boot,[.025,.975]).tolist()}
            uniform=np.asarray([row['uniform_reference_progress'] for row in b])[:,lanes]
            actual=np.asarray([row['reference_progress'] for row in b])[:,lanes]
            result['reference_post_training_minus_uniform_per_learner']=(actual-uniform).mean(axis=0).tolist()
            result['uniform_reference_mean']=float(uniform.mean())
            summary[policy][label]=result
    return summary


def replay(report):
    corpus=PairCorpus(Path(report['corpus']))
    if corpus.sha256!=report['corpus_sha256']:raise ValueError('corpus identity differs')
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([0])
    ids=report['replicate_ids'];verified=0
    rows=list(report['calibration_rows'])
    for stage in ('before','after'):
        for policy_rows in report[stage].values():rows.extend(policy_rows)
    for row in rows:
        for lane,(r,index) in enumerate(zip(ids,row['case_indices'],strict=True)):
            if row['discard_counts'][lane]!=report['count']:raise ValueError('fixed count changed')
            rng=np.random.default_rng(derive_seed(VERSION,'native',row['key'],r,index,'display'))
            order=np.r_[rng.permutation(8),8,9];next_order=np.r_[rng.permutation(8),8,9]
            corpus.restore(env,0,index)
            action=empty_action_batch(1);action['action_type'][:]=1;action['n_cards'][:]=report['count']
            action['cards'][0,:report['count']]=order[row['discard_cards'][lane][:report['count']]]
            discarded=env.step(action);verified+=1
            if discarded.dones[0] or discarded.rewards[0]!=0:raise ValueError('discard replay differs')
            snapshot=env.snapshot(0)
            for followup in ('weak','reference'):
                env.restore(0,snapshot)
                cards=[x for x in row[followup+'_play_cards'][lane] if x>=0]
                action=empty_action_batch(1);action['action_type'][:]=0;action['n_cards'][:]=len(cards)
                action['cards'][0,:len(cards)]=next_order[cards]
                result=env.step(action);verified+=1
                components=result.infos[0]['reward_components']
                if (result.dones[0] or abs(components['blind_progress']-row[followup+'_progress'][lane])>2e-6
                    or bool(components.get('blind_clear',0))!=row[followup+'_clears'][lane]):
                    raise ValueError('native PLAY replay differs')
    return {'native_actions':verified,'calibration_and_evaluation_actions_verified':True}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args(argv);root=args.run_dir;path=root/'report.json';report=json.loads(path.read_text())
    if report.get('status')!='completed' or report['study']!='fixed-membership':raise ValueError('completed native membership study required')
    for source,digest in report['source_hashes'].items():
        if sha256_file(Path(source))!=digest:raise ValueError(f'source changed: {source}')
    verify_state({**report,'study':'followup'},root/'diagnostic-final-state.npz')
    match=matched_gain(report['calibration_rows'],report['learning_rule']['learning_rate'])
    if match!=report['calibration']:raise ValueError('update calibration differs')
    for policy in report['before']:
        for a,b in zip(report['before'][policy],report['after'][policy],strict=True):
            np.testing.assert_array_equal(a['uniform_reference_progress'],b['uniform_reference_progress'])
    result={'status':'completed','source_report_sha256':sha256_file(path),
        'final_state_sha256':sha256_file(root/'diagnostic-final-state.npz'),
        'assessment_source_sha256':sha256_file(Path(__file__)),
        'summary':summarize(report),'replay':replay(report),
        'limits':'52 shared validation cases and three learner seeds on one mapping; bootstrap conditional on these learners; count/action timing forced; reference is test-only; initial expected update match is not realized trajectory matching'}
    write_json(root/'assessment.json',result)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
