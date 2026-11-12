from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from relational_sensory import companion_counts,RelationalEncoder
from flylatro.fly.encoder import Stimulus


def observations(ranks):
    hand=np.zeros((len(ranks),10,38),dtype=np.float32)
    for row,cards in enumerate(ranks):
        for slot,rank in enumerate(cards):hand[row,slot,rank-2]=1
    return {'hand':hand,'hand_len':np.array([len(r) for r in ranks])}


def test_companion_counts_visibility_and_multiplicity():
    obs=observations([[2,2,2,2,7,7,8,9]])
    np.testing.assert_array_equal(companion_counts(obs)[0,:8],[3,3,3,3,1,1,0,0])
    obs['hand'][0,1,37]=1
    np.testing.assert_array_equal(companion_counts(obs)[0,:4],[2,0,2,2])
    obs['hand'][0,2,23]=1
    np.testing.assert_array_equal(companion_counts(obs)[0,:4],[1,0,0,1])


def test_features_follow_card_permutations():
    obs=observations([[2,3,4,5,6,7,8,2]])
    before=companion_counts(obs)
    order=np.array([7,5,3,1,6,4,2,0,8,9]);obs['hand']=obs['hand'][:,order]
    np.testing.assert_array_equal(companion_counts(obs),before[:,order])


def test_true_wrong_energy_and_headroom_are_exactly_matched():
    class Base:
        mapping=SimpleNamespace(available_alpn_indices=np.arange(60),max_rate_hz=150)
        def encode(self,obs):return Stimulus(np.tile(np.linspace(0,150,60,dtype=np.float32),(3,1)),'base','hash')
    obs=observations([[2,3,4,5,6,7,8,2]]*3)
    encoder=RelationalEncoder(Base(),['off','true','wrong'],[0,0,0])
    rates=encoder.encode(obs).rates_hz
    np.testing.assert_allclose((rates[1]-rates[0]).sum(),(rates[2]-rates[0]).sum())
    assert (rates[1]-rates[0]).sum()==150
    assert rates.max()<=150 and not np.array_equal(rates[1],rates[2])
    assert np.count_nonzero(rates[1]-rates[0])==6
    np.testing.assert_array_equal(rates,encoder.encode(obs).rates_hz)
