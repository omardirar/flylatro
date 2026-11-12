from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_sensory_control import route_scores,CueEncoder
from flylatro.fly.encoder import Stimulus


def test_anatomical_scores_choose_specific_two_hop_pathways():
    artifact=SimpleNamespace(root_ids=np.arange(6),pre_indices=np.array([0,1]),
        post_indices=np.array([2,3]),signed_synapse_counts=np.array([4.,4.]))
    topology=SimpleNamespace(pre_indices=np.array([2,3]),post_root_ids=np.array([4,5]),
        anatomical_weights=np.array([5.,5.]))
    scores=route_scores(artifact,topology,np.array([0,1]),[4],[5])
    np.testing.assert_array_equal(scores,[1,-1])


def test_cues_are_equal_energy_and_do_not_clip():
    class Base:
        def encode(self,obs):return Stimulus(np.full((4,12),150.),'base','hash')
    encoder=CueEncoder(Base(),{'target':np.arange(3),'reference':np.arange(3,6),'random':np.arange(6,9)})
    encoder.cues=['off','target','reference','random']
    rates=encoder.encode({}).rates_hz
    for row in range(1,4):assert (rates[row]-rates[0]).sum()==225
    assert rates.max()==150 and rates.min()==75
    np.testing.assert_array_equal(rates[:,9:],150)
