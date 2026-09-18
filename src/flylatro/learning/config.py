"""Strict configuration and stack construction for plastic-brain experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping, TypeVar

import numpy as np

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.mushroom_body.plasticity import PlasticityConfig, ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.fly.plastic_backend import PlasticFlyProcessor, PlasticTorchFlyWireBackend
from flylatro.fly.synthetic_plastic import (
    SyntheticPlasticCircuitSpec,
    SyntheticPlasticFlyProcessor,
)
from flylatro.fly.upstream_encoder import full_feature_names
from flylatro.interface.motor import FixedMotorInterface, MotorMapping
from flylatro.interface.sensory import FixedPlasticSensoryEncoder, SensoryMapping
from flylatro.learning.agent import PlasticFlyAgent
from flylatro.learning.reinforcement import ReinforcementConfig, ReinforcementMapper
from flylatro.learning.reward_schedule import DopamineSchedule
from flylatro.learning.reward_schedule import load_action_schedule
from flylatro.learning.trainer import PlasticTrainer, PlasticTrainingConfig
from flylatro.seeds import SeedPlan


@dataclass(frozen=True, slots=True)
class EnvironmentSettings:
    backend: str = "mock"
    num_envs: int = 1
    blind_target: float = 25.0
    initial_hands: int = 3
    initial_discards: int = 2
    strict_actions: bool = True


@dataclass(frozen=True, slots=True)
class FlySettings:
    backend: str = "synthetic"
    mode: str = "mbon_direct"
    artifact_path: str = "data/flywire/flywire_fafb_v783.npz"
    device: str = "cpu"
    duration_ms: float = 50.0
    reset_fast_state_each_decision: bool = True
    sensory_mapping_seed: int = 0
    sensory_population_width: int = 3
    max_rate_hz: float = 150.0
    motor_pool_width: int = 1
    synthetic_seed: int = 1701
    synthetic_kenyon_count: int = 48
    topology: str = "real"
    shuffle_seed: int = 1701


@dataclass(frozen=True, slots=True)
class MotorSettings:
    deterministic_training: bool = False
    exploration_epsilon: float = 0.05
    exploration_temperature: float = 1.0
    exploration_seed: int = 2203


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    max_environment_decisions: int = 100
    checkpoint_every_decisions: int = 100
    base_fly_seed: int = 2001
    plasticity_enabled: bool = True
    training_seed_offset: int = 0
    reinforcement_mode: str = "outcome"
    dopamine_schedule_path: str = ""
    action_schedule_path: str = ""
    condition: str = "plastic_real"


@dataclass(frozen=True, slots=True)
class CurriculumSettings:
    enabled: bool = False
    ladder: tuple[int, ...] = (1, 2, 3, 5, 8)
    promotion_win_rate: float = 0.70
    evaluation_every_decisions: int = 10_000
    evaluation_episodes: int = 256


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    output_root: str = "runs"
    tensorboard: bool = False


@dataclass(frozen=True, slots=True)
class PlasticExperimentConfig:
    name: str
    source_path: Path
    environment: EnvironmentSettings
    fly: FlySettings
    motor: MotorSettings
    plasticity: PlasticityConfig
    reinforcement: ReinforcementConfig
    training: TrainingSettings
    curriculum: CurriculumSettings
    runtime: RuntimeSettings

    @classmethod
    def load(cls, path: Path) -> "PlasticExperimentConfig":
        path = path.resolve()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        allowed = {
            "experiment",
            "environment",
            "fly",
            "motor",
            "plasticity",
            "reinforcement",
            "training",
            "curriculum",
            "runtime",
        }
        unknown = raw.keys() - allowed
        if unknown:
            raise ValueError(f"unknown plastic config sections: {sorted(unknown)}")
        experiment = raw.get("experiment", {})
        if set(experiment) != {"name"}:
            raise ValueError("[experiment] must contain exactly name")
        curriculum = dict(raw.get("curriculum", {}))
        if "ladder" in curriculum:
            curriculum["ladder"] = tuple(int(value) for value in curriculum["ladder"])
        config = cls(
            name=str(experiment["name"]),
            source_path=path,
            environment=_section(EnvironmentSettings, raw.get("environment", {})),
            fly=_section(FlySettings, raw.get("fly", {})),
            motor=_section(MotorSettings, raw.get("motor", {})),
            plasticity=_section(PlasticityConfig, raw.get("plasticity", {})),
            reinforcement=_section(
                ReinforcementConfig, raw.get("reinforcement", {})
            ),
            training=_section(TrainingSettings, raw.get("training", {})),
            curriculum=_section(CurriculumSettings, curriculum),
            runtime=_section(RuntimeSettings, raw.get("runtime", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.environment.backend not in {"mock", "balatro_sim"}:
            raise ValueError("environment backend must be mock or balatro_sim")
        if self.environment.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.fly.backend not in {"synthetic", "flywire"}:
            raise ValueError("fly backend must be synthetic or flywire")
        if self.fly.mode not in {"mbon_direct", "whole_brain"}:
            raise ValueError("fly mode must be mbon_direct or whole_brain")
        if self.fly.topology not in {"real", "shuffled"}:
            raise ValueError("fly topology must be real or shuffled")
        if self.fly.backend == "synthetic" and self.fly.topology != "real":
            raise ValueError("synthetic topology control uses its own fixed test graph")
        if self.curriculum.enabled:
            if tuple(sorted(set(self.curriculum.ladder))) != self.curriculum.ladder:
                raise ValueError("curriculum ladder must be strictly increasing")
            if self.curriculum.ladder[-1] != 8:
                raise ValueError("curriculum must end at Ante 8")
            if self.curriculum.evaluation_every_decisions < 1:
                raise ValueError("curriculum evaluation cadence must be positive")
            if self.curriculum.evaluation_episodes < 1:
                raise ValueError("curriculum evaluation episodes must be positive")
        if self.training.reinforcement_mode not in {"outcome", "shuffled_schedule"}:
            raise ValueError("unknown reinforcement_mode")
        if (
            self.training.reinforcement_mode == "shuffled_schedule"
            and not self.training.dopamine_schedule_path
        ):
            raise ValueError("shuffled_schedule requires dopamine_schedule_path")
        conditions = {
            "plastic_real",
            "no_plasticity",
            "shuffled_topology",
            "shuffled_reward",
        }
        if self.training.condition not in conditions:
            raise ValueError("unknown experimental condition")
        if self.training.condition == "no_plasticity" and self.training.plasticity_enabled:
            raise ValueError("no_plasticity condition requires plasticity_enabled=false")
        if self.training.condition == "shuffled_topology" and self.fly.topology != "shuffled":
            raise ValueError("shuffled_topology condition requires fly.topology=shuffled")
        if (
            self.training.condition == "shuffled_reward"
            and self.training.reinforcement_mode != "shuffled_schedule"
        ):
            raise ValueError("shuffled_reward condition requires shuffled_schedule")
        if self.training.condition == "shuffled_reward" and not self.training.action_schedule_path:
            raise ValueError("shuffled_reward condition requires action_schedule_path")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "environment": asdict(self.environment),
            "fly": asdict(self.fly),
            "motor": asdict(self.motor),
            "plasticity": asdict(self.plasticity),
            "reinforcement": asdict(self.reinforcement),
            "training": asdict(self.training),
            "curriculum": asdict(self.curriculum),
            "runtime": asdict(self.runtime),
        }

    def require_heavy_opt_in(self, heavy: bool) -> None:
        reasons = []
        if self.environment.backend == "balatro_sim":
            reasons.append("real Balatro simulator")
        if self.fly.backend == "flywire":
            reasons.append("full FlyWire backend")
        if self.environment.num_envs > 8:
            reasons.append("more than eight independent flies")
        if self.training.max_environment_decisions > 1_000:
            reasons.append("more than 1,000 environment decisions")
        if reasons and not heavy:
            raise ValueError("--heavy is required for " + ", ".join(reasons))


@dataclass(slots=True)
class PlasticTrainingStack:
    env: Any
    agent: PlasticFlyAgent
    trainer: PlasticTrainer
    components: dict[str, Any]


def build_plastic_stack(config: PlasticExperimentConfig) -> PlasticTrainingStack:
    env = build_plastic_environment(config)
    learners = config.environment.num_envs
    if config.fly.backend == "synthetic":
        processor = SyntheticPlasticFlyProcessor(
            SyntheticPlasticCircuitSpec(
                seed=config.fly.synthetic_seed,
                kenyon_count=config.fly.synthetic_kenyon_count,
            ),
            mode=config.fly.mode,
        )
        topology = processor.topology
        artifact_hash = "synthetic-development-only"
        population_hash = "synthetic-development-only"
        population_manifest: dict[str, Any] = {
            "development_only": True,
            "counts": {
                "kenyon": config.fly.synthetic_kenyon_count,
                "mbon": len(processor.mbon_root_ids),
                "dan": len(processor.dan_root_ids),
                "descending": len(processor.descending_root_ids),
                "kc_mbon_edges": topology.edge_count,
            },
        }
        sensory_hash = "synthetic-fixed-projection-v1"
        sensory_manifest: dict[str, Any] = {
            "version": "synthetic-fixed-projection-v1",
            "seed": config.fly.synthetic_seed,
            "feature_names": list(full_feature_names()),
            "note": "development-only fixed dense projection; matrix is reproduced from seed",
        }
        backend_version = processor.version
        fly_connectivity_hash = topology.sha256
        dynamics_payload: dict[str, Any] = {
            "version": processor.version,
            "mode": config.fly.mode,
            "seed": config.fly.synthetic_seed,
            "kenyon_count": config.fly.synthetic_kenyon_count,
            "decision_duration_ms": 1.0,
            "fast_state_policy": "stateless-synthetic-decision",
        }
        topology_procedure = "synthetic-complete-kc-mbon-test-graph"
    else:
        artifact_path = Path(config.fly.artifact_path)
        if not artifact_path.is_absolute():
            artifact_path = config.source_path.parent.parent / artifact_path
        artifact = FlyWireArtifact.load(artifact_path)
        shuffle_seed = (
            config.fly.shuffle_seed if config.fly.topology == "shuffled" else None
        )
        shuffled_post = None
        if shuffle_seed is not None:
            _, shuffled_post, _, _ = artifact.edge_arrays(
                shuffle_seed=shuffle_seed, preserve_populations=True
            )
        topology = PlasticEdgeTopology.from_artifact(
            artifact,
            post_indices=shuffled_post,
            version=(
                f"flywire-v783-kc-mbon-population-shuffle-v1-seed-{shuffle_seed}"
                if shuffle_seed is not None
                else "flywire-v783-kc-mbon-neuron-pairs-v1"
            ),
        )
        sensory_mapping = SensoryMapping.from_artifact(
            artifact,
            mapping_seed=config.fly.sensory_mapping_seed,
            population_width=config.fly.sensory_population_width,
            max_rate_hz=config.fly.max_rate_hz,
        )
        encoder = FixedPlasticSensoryEncoder(sensory_mapping)
        output_indices = (
            np.unique(topology.post_indices)
            if config.fly.mode == "mbon_direct"
            else artifact.descending_indices
        )
        readout = PlasticFlyProcessor.required_readout_indices(
            artifact, topology, output_indices
        )
        backend = PlasticTorchFlyWireBackend(
            artifact,
            topology,
            device=config.fly.device,
            readout_indices=readout,
            shuffle_seed=shuffle_seed,
        )
        processor = PlasticFlyProcessor(
            encoder,
            backend,
            topology,
            output_indices=output_indices,
            mode=config.fly.mode,
            duration_ms=config.fly.duration_ms,
            reset_fast_state_each_decision=config.fly.reset_fast_state_each_decision,
        )
        artifact_hash = artifact.manifest["artifact_sha256"]
        population_hash = artifact.population_hash
        population_manifest = {
            "rules": artifact.manifest["mushroom_body_population_rules"],
            "counts": {
                name.removeprefix("n_"): int(value)
                for name, value in artifact.manifest.items()
                if name.startswith("n_")
                and isinstance(value, (int, np.integer))
            },
        }
        sensory_hash = sensory_mapping.sha256
        sensory_manifest = sensory_mapping.to_manifest()
        backend_version = backend.backend_version
        fly_connectivity_hash = backend.connectivity_hash
        dynamics_payload = {
            "backend_dynamics_sha256": backend.dynamics_hash,
            "decision_duration_ms": config.fly.duration_ms,
            "reset_fast_state_each_decision": (
                config.fly.reset_fast_state_each_decision
            ),
        }
        topology_procedure = backend.topology_procedure
    state = PlasticEdgeState.initialize(topology.edge_count, learners=learners)
    plasticity = ThreeFactorPlasticity(topology, state, config.plasticity)
    motor_mapping = MotorMapping.round_robin(
        processor.output_root_ids,
        mode=config.fly.mode,
        pool_width=config.fly.motor_pool_width,
        exploration_epsilon=config.motor.exploration_epsilon,
        exploration_temperature=config.motor.exploration_temperature,
        exploration_seed=config.motor.exploration_seed,
    )
    motor = FixedMotorInterface(motor_mapping)
    reinforcement = ReinforcementMapper(config.reinforcement)
    agent = PlasticFlyAgent(processor, plasticity, motor, reinforcement)
    dopamine_schedule = None
    schedule_hash = None
    action_schedule = None
    action_schedule_hash = None
    if config.training.reinforcement_mode == "shuffled_schedule":
        schedule_path = Path(config.training.dopamine_schedule_path)
        if not schedule_path.is_absolute():
            schedule_path = config.source_path.parent.parent / schedule_path
        schedule = DopamineSchedule.load(schedule_path)
        dopamine_schedule = schedule.pulses
        schedule_hash = schedule.sha256
    if config.training.action_schedule_path:
        action_path = Path(config.training.action_schedule_path)
        if not action_path.is_absolute():
            action_path = config.source_path.parent.parent / action_path
        action_schedule, action_schedule_hash = load_action_schedule(action_path)
    trainer = PlasticTrainer(
        env,
        agent,
        PlasticTrainingConfig(
            max_environment_decisions=config.training.max_environment_decisions,
            deterministic_motor=config.motor.deterministic_training,
            plasticity_enabled=config.training.plasticity_enabled,
            checkpoint_every_decisions=config.training.checkpoint_every_decisions,
            base_fly_seed=config.training.base_fly_seed,
        ),
        training_seeds=SeedPlan().seeds(
            "training",
            learners,
            offset=config.training.training_seed_offset,
        ),
        dopamine_schedule=dopamine_schedule,
        action_schedule=action_schedule,
    )
    components = {
        "architecture": "plastic-brain-v1",
        "condition": config.training.condition,
        "learned_state": "kc-mbon-efficacy",
        "external_trainable_parameter_count": 0,
        "plastic_parameter_count": agent.plastic_parameter_count,
        "fly_backend": backend_version,
        "fly_connectivity_sha256": fly_connectivity_hash,
        "fly_dynamics": dynamics_payload,
        "fly_dynamics_sha256": hashlib.sha256(
            json.dumps(
                dynamics_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "output_mode": config.fly.mode,
        "topology_condition": config.fly.topology,
        "topology_procedure": topology_procedure,
        "topology_shuffle_seed": (
            config.fly.shuffle_seed if config.fly.topology == "shuffled" else None
        ),
        "artifact_sha256": artifact_hash,
        "population_sha256": population_hash,
        "population_manifest": population_manifest,
        "plastic_topology_sha256": topology.sha256,
        "plasticity_rule_sha256": config.plasticity.sha256,
        "sensory_mapping_sha256": sensory_hash,
        "sensory_mapping": sensory_manifest,
        "motor_mapping_sha256": motor_mapping.sha256,
        "motor_mapping": motor_mapping.to_manifest(),
        "reinforcement_mapping_sha256": config.reinforcement.sha256,
        "reinforcement_mapping": asdict(config.reinforcement),
        "plasticity_rule": asdict(config.plasticity),
        "reinforcement_mode": config.training.reinforcement_mode,
        "dopamine_schedule_sha256": schedule_hash,
        "action_schedule_sha256": action_schedule_hash,
        "learner_semantics": (
            "one-sequential-fly"
            if learners == 1
            else "independent-parallel-flies-no-weight-sharing"
        ),
    }
    return PlasticTrainingStack(env, agent, trainer, components)


def build_plastic_environment(config: PlasticExperimentConfig) -> Any:
    settings = config.environment
    if settings.backend == "mock":
        env = MockArrayBalatroEnv(
            settings.num_envs,
            blind_target=settings.blind_target,
            initial_hands=settings.initial_hands,
            initial_discards=settings.initial_discards,
        )
        env.set_win_ante(
            config.curriculum.ladder[0] if config.curriculum.enabled else 8
        )
        return env
    return BalatroSimAdapter(
        settings.num_envs,
        strict=settings.strict_actions,
        win_ante=config.curriculum.ladder[0] if config.curriculum.enabled else 8,
    )


T = TypeVar("T")


def _section(cls: type[T], values: Mapping[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)
