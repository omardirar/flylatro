"""Independent native reward/EMA accounting for the exchange-credit pilot."""
from pathlib import Path
from itertools import combinations
import json
import sys
import numpy as np
from card_learning.common import write_json
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from strategy_puzzles import action


def assess(output):
    report=json.loads((output/'report.json').read_text())
    declaration=json.loads((output/'declaration.json').read_text())
    assert report['status']=='completed' and report['declaration']==declaration
    assert declaration['script_sha256']==sha256_file(Path(__file__).with_name('run_exchange_credit.py'))
    assert report['checkpoint_sha256']==sha256_file(output/'checkpoint.npz')
    roles=np.array(report['roles']);n=len(roles)
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False)
    env.reset([4000]);env.step(action(2));snapshot=env.snapshot(0)
    rewards={}
    for cards in combinations(range(8),5):
        env.restore(0,snapshot);result=env.step(action(0,list(cards)))
        assert not result.dones[0]
        rewards[cards]=float(result.rewards[0])*.05
    baseline=np.zeros(n); records=0; changes=np.zeros(n,dtype=int)
    with (output/'training.jsonl').open() as stream:
        for step,line in enumerate(stream):
            row=json.loads(line);assert row['step']==step
            pulses=np.array([rewards[tuple(sorted(c))] for c in row['cards']])
            np.testing.assert_allclose(pulses,row['native_pulses'],atol=2e-9,rtol=0)
            # Native float32 rewards differ slightly from adapter's decomposed
            # progress; the declared local EMA uses the recorded native pulses.
            pulses=np.array(row['native_pulses'])
            if step==0:baseline[:]=pulses
            delta=pulses.copy();delta[np.isin(roles,[2,3])]-=baseline[np.isin(roles,[2,3])]
            delta[roles==4]=0
            np.testing.assert_allclose(delta,row['modulators'],atol=1e-14,rtol=0)
            baseline+=.01*(pulses-baseline)
            np.testing.assert_allclose(baseline,row['baseline_after'],atol=1e-14,rtol=0)
            for lane,(chosen,greedy) in enumerate(zip(row['cards'],row['greedy'])):
                difference=len(set(chosen)^set(greedy))
                assert len(set(chosen))==len(set(greedy))==5
                assert set(chosen)<=set(range(8))
                if roles[lane] in (1,3):assert difference in (0,2)
                changes[lane]+=difference>0
            records+=n
    assert records==declaration['steps']*n
    with np.load(output/'checkpoint.npz') as z:
        np.testing.assert_allclose(z['baseline'],baseline,atol=1e-14,rtol=0)
        np.testing.assert_array_equal(z['efficacy'][roles==4],z['initial_efficacy'][roles==4])
        assert np.isfinite(z['efficacy']).all()
        assert np.all((z['efficacy']>=.2)&(z['efficacy']<=2))
    for before,after in zip(report['initial'],report['final']):
        np.testing.assert_array_equal(np.array(before['scores'])[:,roles==4],np.array(after['scores'])[:,roles==4])
    result={'status':'completed','training_records_verified':records,'unique_native_opening_actions_checked':56,
        'changed_sets_per_lane':changes.tolist(),'frozen_choices_exact':True,
        'report_sha256':sha256_file(output/'report.json'),'assessor_sha256':sha256_file(Path(__file__)),
        'scope':'native rewards, sampler exchange size, baseline/modulator accounting and frozen evaluations; no replay of neural weight updates',
        'research_sources':{p:sha256_file(Path(__file__).with_name(p)) for p in ['local_normalisation.py','run_card_membership.py']}}
    write_json(output/'assessment.json',result)
    print(json.dumps(result))


if __name__=='__main__':assess(Path(sys.argv[1]))
