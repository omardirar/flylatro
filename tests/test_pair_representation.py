from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_pair_representation import contrast,rank_case


def test_matching_contrast_cancels_additive_rank_effects():
    values=np.array([3,5,7,9])[:,None,None]*np.ones((1,8,2))
    assert contrast(values)['mean_contrast_rms']==0
    values[[0,3]]+=4
    assert contrast(values)['matching_minus_nonmatching_mean']==4
    assert contrast(values)['cross_repeat_contrast_power']==16


def test_cross_repeat_power_retains_noise_sign():
    x=np.zeros((4,2,1));x[0,:,0]=[2,-2]
    assert contrast(x)['cross_repeat_contrast_power']==-1


def test_rank_edit_preserves_other_observations():
    hand=np.zeros((1,10,38),dtype=np.float32);hand[0,:,1]=1;hand[0,:,13]=1
    obs={'hand':hand,'global':np.arange(4)}
    edited=rank_case(obs,7,8)
    assert edited['hand'][0,7,5]==1 and edited['hand'][0,6,6]==1
    np.testing.assert_array_equal(edited['global'],obs['global'])
    np.testing.assert_array_equal(edited['hand'][0,:6],obs['hand'][0,:6])
    assert obs['hand'][0,7,1]==1
