from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_exchange_credit import explore, signed_tags


def test_single_exchange_keeps_count_and_changes_exactly_two_memberships():
    original=np.arange(5)
    changed=0
    for seed in range(1000):
        choice=explore(original,np.random.default_rng(seed),True)
        assert len(set(choice))==5 and min(choice)>=0 and max(choice)<8
        difference=len(set(choice)^set(original))
        assert difference in (0,2)
        changed+=difference==2
    assert 25<changed<80


def test_signed_credit_only_tags_entering_and_leaving_cards():
    greedy=np.array([[0,1,2,3,4],[0,1,2,3,4]])
    chosen=np.array([[0,1,2,3,7],[4,3,2,1,0]])
    tags=signed_tags(chosen,greedy)
    assert tags[0,7]==1 and tags[0,4]==-1
    assert np.count_nonzero(tags[0])==2 and np.count_nonzero(tags[1])==0
    assert np.all(tags.sum(axis=1)==0)


def test_signed_eligibility_reverses_update_and_preserves_local_budget():
    from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
    from flylatro.fly.mushroom_body.state import PlasticEdgeState
    from flylatro.fly.mushroom_body.plasticity import PlasticityConfig
    from local_normalisation import NormalisedThreeFactorPlasticity
    topology=PlasticEdgeTopology.synthetic(np.array([0,1,0,1]),np.array([2,2,3,3]),np.ones(4,dtype=np.float32))
    rule=NormalisedThreeFactorPlasticity(topology,PlasticEdgeState.initialize(4),PlasticityConfig())
    rule.state.eligibility[:]=[[1,0,-1,0]]
    rule.apply_dopamine(.1,0)
    assert rule.state.efficacy[0,0]>1 and rule.state.efficacy[0,2]<1
    np.testing.assert_allclose(rule.incoming_strengths(),rule.budgets,atol=1e-6)
