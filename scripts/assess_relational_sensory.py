"""Independent native replay of relational-input pair pilot decisions."""
from pathlib import Path
import json
import sys
import numpy as np
from card_learning.common import write_json
from card_learning.pairs import PairCorpus,pair_slots
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import sha256_file
from strategy_puzzles import action
from relational_sensory import companion_counts


def assess(root):
    d=json.loads((root/'declaration.json').read_text());report=json.loads((root/'report.json').read_text())
    for path,digest in d['input_sha256'].items():assert sha256_file(Path(path))==digest
    assert report['checkpoint_sha256']==sha256_file(root/'checkpoint.npz')
    corpus=PairCorpus(Path('artefacts/local-normalisation-20261031-v1/pairs.json'))
    env=BalatroSimAdapter(1,win_ante=1,allow_skip_blinds=False);env.reset([0])
    cache={};counts={'training':0,'evaluation':0}
    def check(row,label):
        for lane,index in enumerate(row['case_indices']):
            order=np.array(row['orders'][lane]);assert np.array_equal(np.sort(order[:8]),np.arange(8))
            cards=tuple(sorted(int(order[c]) for c in row['cards'][lane]));assert len(set(cards))==2
            key=(index,cards)
            if key not in cache:
                env.restore(0,corpus.cases[index]['_snapshot']);obs,_=env.observe()
                pair=set(pair_slots(obs))
                assert set(np.flatnonzero(companion_counts(obs)[0]>0))==pair
                result=env.step(action(0,list(cards)))
                assert not result.dones[0]
                cache[key]=(result.infos[0]['reward_components']['blind_progress'],pair==set(cards))
            progress,hit=cache[key]
            assert abs(progress-row['native_progress'][lane])<1e-7 and hit==row['pair'][lane]
            counts[label]+=1
    with (root/'training.jsonl').open() as stream:
        for step,line in enumerate(stream):
            row=json.loads(line);assert row['step']==step
            assert all(corpus.cases[i]['split']=='training' for i in row['case_indices'])
            check(row,'training')
    assert counts['training']==18*d['steps']
    for condition in ['initial','final']:
        for split,rows in report[condition].items():
            assert len(rows)==52
            for row in rows:
                assert all(corpus.cases[i]['split']==split for i in row['case_indices'])
                check(row,'evaluation')
    frozen=np.array(report['roles'])>=3
    with np.load(root/'checkpoint.npz') as z:
        np.testing.assert_array_equal(z['efficacy'][frozen],z['initial_efficacy'][frozen])
        assert np.isfinite(z['efficacy']).all() and np.all((z['efficacy']>=.2)&(z['efficacy']<=2))
    for split in report['initial']:
        for a,b in zip(report['initial'][split],report['final'][split]):
            np.testing.assert_array_equal(np.array(a['cards'])[frozen],np.array(b['cards'])[frozen])
    result={'status':'completed','records_verified':counts,'unique_native_actions':len(cache),
        'frozen_choices_and_weights_exact':True,'report_sha256':sha256_file(root/'report.json'),
        'assessor_sha256':sha256_file(Path(__file__)),'limits':'native decisions and outcomes checked; neural training updates not replayed'}
    write_json(root/'assessment.json',result);print(json.dumps(result))


if __name__=='__main__':assess(Path(sys.argv[1]))
