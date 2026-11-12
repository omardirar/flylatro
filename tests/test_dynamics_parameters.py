from dataclasses import replace
from pathlib import Path

import pytest

from flylatro.learning.config import PlasticExperimentConfig, _lif_parameters, fly_dynamics_payload


def test_fixed_gain_changes_dynamics_identity_without_changing_learning_architecture():
    c = PlasticExperimentConfig.load(Path("configs/plastic-v783-source-complete-calibration.toml"))
    adjusted = replace(c, fly=replace(c.fly, synapse_scale_mv=0.18))
    assert _lif_parameters(c).synapse_scale_mv == 0.275
    assert _lif_parameters(adjusted).synapse_scale_mv == 0.18
    assert fly_dynamics_payload(c) != fly_dynamics_payload(adjusted)
    assert c.plasticity.sha256 == adjusted.plasticity.sha256
    assert c.reinforcement.sha256 == adjusted.reinforcement.sha256
    assert c.fly.sensory_mapping_seed == adjusted.fly.sensory_mapping_seed


@pytest.mark.parametrize("gain", [0, -1, float("inf"), float("nan")])
def test_config_refuses_invalid_fixed_synaptic_gain(gain):
    c = PlasticExperimentConfig.load(Path("configs/plastic-smoke.toml"))
    with pytest.raises(ValueError, match="synapse_scale_mv"):
        replace(c, fly=replace(c.fly, synapse_scale_mv=gain)).validate()
