"""Post-run paired outcomes, native replay and observable hand-potential audit."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from card_learning.audit import score
from card_learning.common import write_json
from card_learning.pairs import PairCorpus
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import derive_seed
from train_discard_play_curriculum import VERSION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.curriculum_dir
    d = json.loads((root/'report.json').read_text())
    corpus = PairCorpus(root/'pairs.json')
    if d['status'] != 'completed' or d['corpus_sha256'] != corpus.sha256:
        raise ValueError('completed curriculum with exact corpus required')
    env = BalatroSimAdapter(1, win_ante=1, allow_skip_blinds=False); env.reset([0])
    cache = {}; verified = 0; summaries = {}
    def discard_hand(index, physical):
        key = (index, tuple(physical))
        if key not in cache:
            corpus.restore(env,0,index)
            action = empty_action_batch(1); action['action_type'][:] = 1; action['n_cards'][:] = len(physical)
            action['cards'][0,:len(physical)] = physical
            env.step(action)
            obs,_ = env.observe(); n=int(obs['hand_len'][0]); features=obs['hand'][0,:n]
            hand=[(int(card[:13].argmax())+2,int(card[13:17].argmax())) for card in features]
            best=max(score(hand, subset)[0] for count in range(1,min(5,n)+1) for subset in combinations(range(n),count))
            cache[key] = (best, env.snapshot(0))
        return cache[key]
    for policy in ('greedy','training-sampled'):
        summaries[policy] = {}
        for stage in ('before','after'):
            report=d[stage][policy]; rows=report['trials']
            repeats=len(rows)//len(corpus.indices('validation'))
            actual, random, captured = [], [], []
            for row_index,row in enumerate(rows):
                potentials, random_potentials = [], []
                for lane,index in enumerate(row['case_indices']):
                    replicate=lane//2; key=('evaluation',row_index%repeats,policy)
                    rng=np.random.default_rng(derive_seed(VERSION,key,replicate,index,'display'))
                    order=np.r_[rng.permutation(8),8,9]; next_order=np.r_[rng.permutation(8),8,9]
                    count=row['discard_counts'][lane]
                    physical=order[np.asarray(row['discard_cards'][lane][:count],dtype=int)].tolist()
                    best,snapshot=discard_hand(index,physical); potentials.append(best)
                    env.restore(0,snapshot)
                    cards=[slot for slot in row['play_cards'][lane] if slot>=0]
                    action=empty_action_batch(1); action['action_type'][:]=0;action['n_cards'][:]=len(cards)
                    action['cards'][0,:len(cards)]=next_order[cards]
                    native=env.step(action).infos[0]['reward_components']['blind_progress']
                    if abs(native-row['progress'][lane])>2e-6:
                        raise ValueError('curriculum action/reward replay differs')
                    verified+=1
                    rng=np.random.default_rng(derive_seed(VERSION,key,replicate,index,'random-same-count'))
                    physical=order[np.sort(rng.choice(8,count,replace=False))].tolist()
                    random_potentials.append(discard_hand(index,physical)[0])
                actual.append(potentials);random.append(random_potentials)
                captured.append(np.asarray(row['progress'])*300/np.minimum(potentials,300))
            summaries[policy][stage]={'mean_best_available_chips_after_discard':np.mean(actual,axis=0).tolist(),
                'mean_best_available_chips_random_same_count':np.mean(random,axis=0).tolist(),
                'mean_fraction_available_progress_captured_by_fixed_play':np.mean(captured,axis=0).tolist()}
        before=np.asarray([row['progress'] for row in d['before'][policy]['trials']])
        after=np.asarray([row['progress'] for row in d['after'][policy]['trials']])
        np.testing.assert_array_equal(before[:,1::2],after[:,1::2])
        deltas=after[:,::2]-before[:,::2]
        cases=len(corpus.indices('validation'));repeats=len(after)//cases
        per_case=deltas.reshape(cases,repeats,d['replicates']).mean(axis=(1,2))
        rng=np.random.default_rng(derive_seed(VERSION,policy,'case-bootstrap'))
        boot=per_case[rng.integers(0,cases,size=(2000,cases))].mean(axis=1)
        summaries[policy]['paired']={'per_learner_progress_delta':deltas.mean(axis=0).tolist(),
            'mean_progress_delta':float(deltas.mean()),
            'case_bootstrap_95_interval_conditional_on_these_learners':np.quantile(boot,[.025,.975]).tolist(),
            'post_training_minus_random_same_count_progress':(after[:,::2]-np.asarray([row['random_same_count_progress'] for row in d['after'][policy]['trials']])[:,::2]).mean(axis=0).tolist(),
            'changed_discard_memberships':[(sum(set(a['discard_cards'][2*i][:a['discard_counts'][2*i]])!=set(b['discard_cards'][2*i][:b['discard_counts'][2*i]]) for a,b in zip(d['before'][policy]['trials'],d['after'][policy]['trials'],strict=True))) for i in range(d['replicates'])]}
    result={'version':'posthoc-discard-curriculum-assessment-v1','status':'completed',
        'source_report_sha256':sha256_file(root/'report.json'),'final_state_sha256':sha256_file(root/'diagnostic-final-state.npz'),
        'verified_native_followup_plays':verified,'unique_discard_outcomes_checked':len(cache),'policies':summaries,
        'limits':'observable hand-potential analysis is post-hoc and never training feedback; bootstrap resamples whole validation cases, conditional on only three trained learners; no population-level significance claim'}
    write_json(root/'assessment.json',result)
    print(json.dumps({'verified_native_plays':verified,'paired':{p:v['paired'] for p,v in summaries.items()}},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
