from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import subprocess
import sys

from flylatro.learning.checkpoints import (
    load_plastic_checkpoint,
    save_plastic_checkpoint,
)
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.evaluation.plastic import evaluate_plastic_fly
from flylatro.learning.reward_schedule import MatchedActionStep


def smoke_config() -> PlasticExperimentConfig:
    return PlasticExperimentConfig.load(Path("configs/plastic-smoke.toml"))


def test_primary_training_path_changes_only_internal_plastic_state() -> None:
    stack = build_plastic_stack(smoke_config())
    initial = stack.agent.plasticity.state.efficacy.copy()

    stack.trainer.train()

    assert stack.agent.trainable_parameter_count == 0
    assert stack.agent.plastic_parameter_count == initial.size
    assert np.any(stack.agent.plasticity.state.efficacy != initial)
    assert stack.components["learned_state"] == "kc-mbon-efficacy"
    assert stack.components["external_trainable_parameter_count"] == 0
    assert stack.components["sensory_mapping"]["version"]
    assert stack.components["motor_mapping"]["structure_sha256"] == stack.components[
        "motor_mapping_sha256"
    ]
    assert stack.components["motor_mapping"]["sha256"] == stack.components[
        "motor_artifact_sha256"
    ]
    assert stack.components["reinforcement_mapping"]["version"]


def test_checkpoint_resume_reproduces_future_plasticity_exactly(
    tmp_path: Path,
) -> None:
    config = smoke_config()
    uninterrupted = build_plastic_stack(config)
    for _ in range(5):
        uninterrupted.trainer.step()
    checkpoint = tmp_path / "plastic.pkl"
    save_plastic_checkpoint(
        checkpoint,
        uninterrupted.trainer,
        experiment_config=config.to_dict(),
        component_metadata=uninterrupted.components,
    )

    resumed = build_plastic_stack(config)
    load_plastic_checkpoint(
        checkpoint,
        resumed.trainer,
        expected_components=resumed.components,
    )
    for _ in range(4):
        uninterrupted.trainer.step()
        resumed.trainer.step()

    np.testing.assert_array_equal(
        uninterrupted.agent.plasticity.state.efficacy,
        resumed.agent.plasticity.state.efficacy,
    )
    np.testing.assert_array_equal(
        uninterrupted.agent.plasticity.state.eligibility,
        resumed.agent.plasticity.state.eligibility,
    )
    assert (
        uninterrupted.agent.plasticity.state.weight_sha256
        == resumed.agent.plasticity.state.weight_sha256
    )
    assert uninterrupted.trainer.state == resumed.trainer.state


def test_same_experience_and_rng_plan_reproduce_learned_weights() -> None:
    first = build_plastic_stack(smoke_config())
    second = build_plastic_stack(smoke_config())

    first.trainer.train()
    second.trainer.train()

    np.testing.assert_array_equal(
        first.agent.plasticity.state.efficacy,
        second.agent.plasticity.state.efficacy,
    )
    assert (
        first.agent.plasticity.state.weight_sha256
        == second.agent.plasticity.state.weight_sha256
    )


def test_no_plasticity_control_keeps_initial_weights() -> None:
    stack = build_plastic_stack(smoke_config())
    stack.trainer.config = type(stack.trainer.config)(
        max_environment_decisions=8,
        deterministic_motor=stack.trainer.config.deterministic_motor,
        plasticity_enabled=False,
        checkpoint_every_decisions=8,
        base_fly_seed=stack.trainer.config.base_fly_seed,
    )
    initial_hash = stack.agent.plasticity.state.weight_sha256
    stack.trainer.train()
    assert stack.agent.plasticity.state.weight_sha256 == initial_hash


def test_frozen_evaluation_changes_no_plastic_learning_state() -> None:
    stack = build_plastic_stack(smoke_config())
    for _ in range(4):
        stack.trainer.step()
    before_weights = stack.agent.plasticity.state.weight_sha256
    before_eligibility = stack.agent.plasticity.state.eligibility.copy()

    result = evaluate_plastic_fly(stack.env, stack.agent, (30_000_000,))

    assert result.frozen_weight_sha256 == before_weights
    assert stack.agent.plasticity.state.weight_sha256 == before_weights
    np.testing.assert_array_equal(
        stack.agent.plasticity.state.eligibility, before_eligibility
    )
    assert len(result.episodes) == 1


def test_changing_internal_synapses_can_change_fixed_motor_behaviour() -> None:
    stack = build_plastic_stack(smoke_config())
    observations, masks = stack.env.reset((123,))
    baseline = stack.agent.act(
        observations,
        masks,
        fly_seeds=(1,),
        deterministic_motor=True,
        record_eligibility=False,
    )
    original_action = int(baseline.actions["action_type"][0])
    alternative = 1 - original_action  # mock exposes only PLAY=0 and DISCARD=1
    topology = stack.agent.plasticity.topology
    state = stack.agent.plasticity.state
    mapping = stack.agent.motor.mapping
    target_outputs = mapping.pools["action_type"][alternative]
    original_outputs = mapping.pools["action_type"][original_action]
    for output in target_outputs:
        target_post = stack.agent.processor.spec.kenyon_count + output
        state.efficacy[0, topology.post_indices == target_post] = 2.0
    for output in original_outputs:
        original_post = stack.agent.processor.spec.kenyon_count + output
        state.efficacy[0, topology.post_indices == original_post] = 0.2

    changed = stack.agent.act(
        observations,
        masks,
        fly_seeds=(1,),
        deterministic_motor=True,
        record_eligibility=False,
    )

    assert int(changed.actions["action_type"][0]) == alternative


def test_whole_brain_output_mode_uses_same_fixed_motor_contract() -> None:
    config = smoke_config()
    config = replace(config, fly=replace(config.fly, mode="whole_brain"))
    stack = build_plastic_stack(config)

    metrics = stack.trainer.step()

    assert stack.components["output_mode"] == "whole_brain"
    assert stack.agent.motor.trainable_parameter_count == 0
    assert np.isfinite(metrics["neural/descending_mean_activity"])


def test_primary_cli_import_graph_does_not_load_legacy_policy_or_ppo() -> None:
    code = """
import sys
import flylatro.learning.cli
import flylatro.evaluation.plastic_cli
import flylatro.replay.cli
bad = [name for name in sys.modules if name.startswith(
    ('flylatro.policy', 'flylatro.training.ppo', 'flylatro.training.rollout')
)]
raise SystemExit(1 if bad else 0)
"""
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_shuffled_reward_control_replays_the_same_experience_actions() -> None:
    source = build_plastic_stack(smoke_config())
    scheduled_steps = []
    pulses = []
    for _ in range(8):
        source.trainer.step()
        scheduled_steps.append(
            MatchedActionStep(
                actions={
                key: value.copy()
                for key, value in source.trainer.last_executed_actions.items()
                },
                state_hashes_before=source.trainer.last_state_hashes_before,
                state_hashes_after=source.trainer.last_state_hashes_after,
                curriculum_ante=8,
            )
        )
        pulses.extend(source.trainer.last_learning.pulses)

    control = build_plastic_stack(smoke_config())
    control.trainer.action_schedule = tuple(scheduled_steps)
    control.trainer.dopamine_schedule = tuple(reversed(pulses))
    for expected in scheduled_steps:
        control.trainer.step()
        assert control.trainer.last_state_hashes_before == expected.state_hashes_before
        assert control.trainer.last_state_hashes_after == expected.state_hashes_after


def test_matched_experience_schedule_fails_fast_on_state_divergence() -> None:
    stack = build_plastic_stack(smoke_config())
    stack.trainer.action_schedule = (
        MatchedActionStep(
            actions=stack.agent.act(
                stack.trainer.observations,
                stack.trainer.masks,
                fly_seeds=(1,),
                deterministic_motor=True,
                record_eligibility=False,
            ).actions,
            state_hashes_before=("not-the-current-state",),
        ),
    )

    import pytest

    with pytest.raises(RuntimeError, match="diverged before"):
        stack.trainer.step()


def test_training_reports_behaviour_diversity_and_can_emit_sparse_changes() -> None:
    stack = build_plastic_stack(smoke_config())
    stack.trainer.record_detailed_plasticity = True

    metrics = stack.trainer.step()
    event = stack.trainer.last_learning.events[0]

    assert metrics["behaviour/action_type_entropy"] >= 0
    assert metrics["behaviour/action_types_observed"] >= 1
    assert 0 < metrics["behaviour/dominant_action_fraction"] <= 1
    assert len(event.changed_edge_indices) == event.changed_synapses
    assert len(event.efficacy_changes) == event.changed_synapses
    for edge in event.changed_edge_indices:
        assert 0 <= edge < stack.agent.plasticity.topology.edge_count
