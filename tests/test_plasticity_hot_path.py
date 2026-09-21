from __future__ import annotations

import numpy as np
import pytest

from flylatro.fly.mushroom_body.plasticity import (
    PlasticityConfig,
    PlasticityEvent,
    ThreeFactorPlasticity,
)
from flylatro.fly.mushroom_body.state import (
    PlasticEdgeState,
    TorchPlasticEdgeState,
    state_numpy,
)
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology

torch = pytest.importorskip("torch")


def _topology(edges: int = 6) -> PlasticEdgeTopology:
    return PlasticEdgeTopology.synthetic(
        np.arange(edges, dtype=np.int64),
        np.arange(edges, dtype=np.int64) + edges,
        np.linspace(0.5, 2.0, edges).astype(np.float32),
    )


def _rule(state) -> ThreeFactorPlasticity:
    return ThreeFactorPlasticity(
        _topology(state.edge_count),
        state,
        PlasticityConfig(learning_rate=0.1, kc_reference_hz=20.0, mbon_reference_hz=10.0),
    )


def test_routine_events_are_lightweight_and_detail_is_opt_in() -> None:
    rule = _rule(PlasticEdgeState.initialize(6, learners=2))
    rule.record_activity(np.full((2, 6), 20.0), np.full((2, 6), 10.0))
    routine = rule.apply_dopamine(0.5, 0.0)[0]
    assert isinstance(routine, PlasticityEvent)
    assert routine.detail is None and routine.detailed is False
    assert routine.before_hash is None and routine.after_hash is None
    assert routine.changed_edge_indices == () and routine.efficacy_changes == ()
    assert routine.changed_synapses == 6
    assert routine.absolute_change > 0
    assert routine.maximum_absolute_change > 0
    assert routine.eligible_synapses == 6
    assert routine.mean_absolute_eligibility > 0

    rule.record_activity(np.full((2, 6), 20.0), np.full((2, 6), 10.0))
    detailed = rule.apply_dopamine(0.5, 0.0, detail=True)[0]
    assert detailed.detailed is True
    assert len(detailed.changed_edge_indices) == detailed.changed_synapses
    assert len(detailed.efficacy_changes) == detailed.changed_synapses
    assert detailed.before_hash != detailed.after_hash


def test_torch_and_numpy_paths_agree_exactly() -> None:
    numpy_rule = _rule(PlasticEdgeState.initialize(6, learners=2))
    torch_rule = _rule(TorchPlasticEdgeState.initialize(6, learners=2))
    pre = np.tile(np.linspace(0.0, 40.0, 6, dtype=np.float32), (2, 1))
    post = np.tile(np.linspace(20.0, 0.0, 6, dtype=np.float32), (2, 1))
    for appetitive, aversive in ((0.4, 0.0), (0.0, 0.3), (0.2, 0.1)):
        numpy_rule.record_activity(pre, post)
        torch_rule.record_activity(pre, post)
        numpy_event = numpy_rule.apply_dopamine(appetitive, aversive)[0]
        torch_event = torch_rule.apply_dopamine(appetitive, aversive)[0]
        assert numpy_event.changed_synapses == torch_event.changed_synapses
        assert numpy_event.absolute_change == pytest.approx(
            torch_event.absolute_change, rel=1e-5, abs=1e-7
        )
        assert numpy_event.eligible_synapses == torch_event.eligible_synapses
    np.testing.assert_allclose(
        state_numpy(torch_rule.state.efficacy), numpy_rule.state.efficacy, rtol=1e-6
    )


def test_routine_torch_step_never_materialises_a_full_host_vector(monkeypatch) -> None:
    rule = _rule(TorchPlasticEdgeState.initialize(64, learners=2))
    transfers: list[tuple[int, ...]] = []
    original = torch.Tensor.cpu

    def tracked(self, *args, **kwargs):  # noqa: ANN001 - torch method shim
        transfers.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", tracked)
    rule.record_activity(np.full((2, 64), 20.0), np.full((2, 64), 10.0))
    rule.apply_dopamine(0.5, 0.0)
    rule.routine_metrics()
    # Only small scalar blocks cross the device boundary: one [learners, 10]
    # summary and one nine-element metric block.
    assert transfers
    assert all(int(np.prod(shape or (1,))) <= 32 for shape in transfers), transfers

    transfers.clear()
    rule.record_activity(np.full((2, 64), 20.0), np.full((2, 64), 10.0))
    rule.apply_dopamine(0.5, 0.0, detail=True)
    assert any(64 in shape for shape in transfers), "detail mode must copy edges"


def test_routine_metrics_match_explicit_host_computation() -> None:
    for state in (
        PlasticEdgeState.initialize(16, learners=3),
        TorchPlasticEdgeState.initialize(16, learners=3),
    ):
        rule = _rule(state)
        rule.record_activity(np.full((3, 16), 30.0), np.full((3, 16), 5.0))
        rule.apply_dopamine(0.7, 0.1)
        metrics = rule.routine_metrics()
        efficacy = state_numpy(rule.state.efficacy).astype(np.float64)
        initial = state_numpy(rule.state.initial_efficacy).astype(np.float64)
        eligibility = np.abs(state_numpy(rule.state.eligibility).astype(np.float64))
        assert metrics["mean_efficacy"] == pytest.approx(efficacy.mean(), rel=1e-6)
        assert metrics["absolute_change"] == pytest.approx(
            np.abs(efficacy - initial).sum(), rel=1e-5
        )
        assert metrics["eligibility_mean_absolute"] == pytest.approx(
            eligibility.mean(), rel=1e-6
        )
        assert metrics["lower_bound_fraction"] == pytest.approx(
            float(np.mean(efficacy <= rule.config.min_efficacy))
        )


def test_effective_anatomical_weights_work_for_both_state_implementations() -> None:
    expected = None
    for state in (
        PlasticEdgeState.initialize(6, learners=2),
        TorchPlasticEdgeState.initialize(6, learners=2),
    ):
        rule = _rule(state)
        weights = rule.effective_anatomical_weights()
        assert weights.dtype == np.float32
        assert weights.shape == (2, 6)
        np.testing.assert_allclose(weights[0], rule.topology.anatomical_weights)
        if expected is None:
            expected = weights
        else:
            np.testing.assert_allclose(weights, expected)


def test_weight_audit_is_explicit_and_not_part_of_the_hot_path() -> None:
    rule = _rule(TorchPlasticEdgeState.initialize(8, learners=2))
    audit = rule.weight_audit()
    assert len(audit["plastic_weight_sha256"]) == 64
    assert len(audit["per_learner_weight_sha256"]) == 2
    assert audit["plastic_topology_sha256"] == rule.topology.sha256
    assert audit["plasticity_rule_sha256"] == rule.config.sha256
    rule.record_activity(np.full((2, 8), 20.0), np.full((2, 8), 10.0))
    event = rule.apply_dopamine(0.5, 0.0)[0]
    assert event.before_hash is None, "routine events must not hash weights"


def test_disabled_plasticity_reports_zero_change_without_allocating_deltas() -> None:
    for state in (
        PlasticEdgeState.initialize(8, learners=2),
        TorchPlasticEdgeState.initialize(8, learners=2),
    ):
        rule = _rule(state)
        rule.record_activity(np.full((2, 8), 20.0), np.full((2, 8), 10.0))
        event = rule.apply_dopamine(0.9, 0.0, plasticity_enabled=False)[0]
        assert event.changed_synapses == 0
        assert event.absolute_change == 0.0
        assert event.maximum_absolute_change == 0.0
        assert event.eligible_synapses == 8
        np.testing.assert_allclose(
            state_numpy(rule.state.efficacy), state_numpy(rule.state.initial_efficacy)
        )
