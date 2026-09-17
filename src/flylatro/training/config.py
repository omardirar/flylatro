"""Complete V1 experiment configuration and stack construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping, TypeVar

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.fly.processors import (
    DirectObservationProcessor,
    FixedReservoirProcessor,
    RealFlyProcessor,
)
from flylatro.fly.upstream_encoder import (
    FixedUpstreamBalatroEncoder,
    FlyWireEncoderSpec,
    full_feature_names,
)
from flylatro.seeds import SeedPlan, seed_everything


@dataclass(frozen=True, slots=True)
class EnvironmentSettings:
    backend: str = "mock"
    num_envs: int = 4
    blind_target: float = 25.0
    initial_hands: int = 3
    initial_discards: int = 2
    strict_actions: bool = True


@dataclass(frozen=True, slots=True)
class ProcessorSettings:
    kind: str = "synthetic"
    artifact_path: str = "data/flywire/flywire_fafb_v783.npz"
    duration_ms: float = 50.0
    microbatch_size: int = 2
    population_width: int = 3
    max_rate_hz: float = 150.0
    readout_count: int = 1305
    synthetic_reservoir_size: int = 64
    synthetic_output_size: int = 32
    synthetic_steps: int = 3
    shuffle_seed: int = 1701
    record_events: bool = False


@dataclass(frozen=True, slots=True)
class PolicySettings:
    hidden_size: int = 0


@dataclass(frozen=True, slots=True)
class PPOSettings:
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    seed: int = 1
    deterministic_torch: bool = False
    max_updates: int = 2
    checkpoint_every_updates: int = 1


@dataclass(frozen=True, slots=True)
class RewardSettings:
    shaping_beta: float = 1.0
    progress_component: str = "delta_capped_blind_progress"
    blind_clear_base: float = 0.5
    final_win_bonus: float = 15.0


@dataclass(frozen=True, slots=True)
class CurriculumSettings:
    enabled: bool = False
    ladder: tuple[int, ...] = (1, 2, 3, 5, 8)
    promotion_win_rate: float = 0.7
    evaluation_frequency_updates: int = 100
    evaluation_episodes: int = 256


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    stream: str = "validation"
    episodes: int = 8
    seed_offset: int = 0
    max_vector_steps: int = 10_000
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    device: str = "cpu"
    output_root: str = "runs"
    tensorboard: bool = True


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    environment: EnvironmentSettings
    processor: ProcessorSettings
    policy: PolicySettings
    ppo: PPOSettings
    reward: RewardSettings
    curriculum: CurriculumSettings
    evaluation: EvaluationSettings
    runtime: RuntimeSettings
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        path = path.resolve()
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        allowed = {
            "experiment", "environment", "processor", "policy", "ppo",
            "reward", "curriculum", "evaluation", "runtime",
        }
        unknown = raw.keys() - allowed
        if unknown:
            raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")
        try:
            config = cls(
                name=str(raw["experiment"]["name"]),
                environment=_section(EnvironmentSettings, raw.get("environment", {})),
                processor=_section(ProcessorSettings, raw.get("processor", {})),
                policy=_section(PolicySettings, raw.get("policy", {})),
                ppo=_section(PPOSettings, raw.get("ppo", {})),
                reward=_section(RewardSettings, raw.get("reward", {})),
                curriculum=_section(CurriculumSettings, raw.get("curriculum", {})),
                evaluation=_section(EvaluationSettings, raw.get("evaluation", {})),
                runtime=_section(RuntimeSettings, raw.get("runtime", {})),
                source_path=path,
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid experiment config {path}: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.environment.backend not in {"mock", "balatro_sim"}:
            raise ValueError("environment.backend must be mock or balatro_sim")
        if self.processor.kind not in {"synthetic", "real", "shuffled", "conventional"}:
            raise ValueError("unknown processor kind")
        if self.environment.num_envs < 1 or self.processor.microbatch_size < 1:
            raise ValueError("environment and microbatch sizes must be positive")
        if self.processor.duration_ms <= 0 or self.processor.population_width < 1:
            raise ValueError("fly timing/population settings must be positive")
        if not 0 <= self.reward.shaping_beta <= 1:
            raise ValueError("reward shaping_beta must be in [0, 1]")
        if self.reward.progress_component != "delta_capped_blind_progress":
            raise ValueError("V1 only supports strategy-neutral blind progress")
        if self.reward.blind_clear_base != 0.5 or self.reward.final_win_bonus != 15.0:
            raise ValueError(
                "the pinned simulator fixes blind_clear_base=0.5 and "
                "final_win_bonus=15.0; shaping_beta is the configurable V1 scale"
            )
        if self.ppo.minibatch_size > self.ppo.rollout_steps * self.environment.num_envs:
            raise ValueError("PPO minibatch exceeds rollout sample count")
        if min(self.ppo.max_updates, self.ppo.checkpoint_every_updates) < 1:
            raise ValueError("training update/checkpoint counts must be positive")
        if (
            self.evaluation.episodes < 1
            or self.evaluation.max_vector_steps < 1
            or self.evaluation.seed_offset < 0
        ):
            raise ValueError("evaluation limits must be positive")
        if self.curriculum.enabled:
            if tuple(sorted(set(self.curriculum.ladder))) != self.curriculum.ladder:
                raise ValueError("curriculum ladder must be strictly increasing")
            if self.curriculum.ladder[-1] != 8:
                raise ValueError("curriculum must end at Ante 8")
            if self.curriculum.evaluation_episodes % self.environment.num_envs:
                raise ValueError(
                    "curriculum evaluation episodes must divide by num_envs"
                )

    def require_heavy_opt_in(self, heavy: bool) -> None:
        reasons = []
        if self.environment.backend == "balatro_sim":
            reasons.append("real Balatro simulator")
        if self.processor.kind in {"real", "shuffled"}:
            reasons.append("adult connectome")
        if self.environment.num_envs > 64:
            reasons.append("more than 64 environments")
        if self.ppo.max_updates > 10:
            reasons.append("more than 10 PPO updates")
        if reasons and not heavy:
            raise ValueError(
                "heavy profile requires explicit --heavy on the dedicated machine: "
                + ", ".join(reasons)
            )

    def payload(self) -> dict[str, Any]:
        result = {
            "experiment": {"name": self.name},
            "environment": asdict(self.environment),
            "processor": asdict(self.processor),
            "policy": asdict(self.policy),
            "ppo": asdict(self.ppo),
            "reward": asdict(self.reward),
            "curriculum": asdict(self.curriculum),
            "evaluation": asdict(self.evaluation),
            "runtime": asdict(self.runtime),
        }
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(slots=True)
class TrainingStack:
    env: Any
    processor: Any
    policy: Any
    trainer: Any
    components: dict[str, Any]


def build_training_stack(config: ExperimentConfig) -> TrainingStack:
    import torch

    from flylatro.fly.flywire_artifact import FlyWireArtifact
    from flylatro.fly.torch_backend import TorchFlyWireBackend
    from flylatro.policy.torch_structured import TorchStructuredPolicy
    from flylatro.training.ppo import PPOConfig, PPOTrainer

    seed_everything(config.ppo.seed, deterministic_torch=config.ppo.deterministic_torch)
    first_ante = config.curriculum.ladder[0] if config.curriculum.enabled else 8
    env = build_environment(config, win_ante=first_ante)
    settings = config.processor
    components: dict[str, Any] = {"condition": settings.kind}
    if settings.kind == "synthetic":
        processor = FixedReservoirProcessor(
            len(full_feature_names()),
            reservoir_size=settings.synthetic_reservoir_size,
            output_size=settings.synthetic_output_size,
            steps=settings.synthetic_steps,
            seed=config.ppo.seed,
            device=config.runtime.device,
        )
        components.update(
            processor_version=processor.version,
            connectome="synthetic-development-only",
            encoder="direct-complete-observation-development-only",
            backend_version="synthetic-development-only",
        )
    elif settings.kind == "conventional":
        processor = DirectObservationProcessor(device=config.runtime.device)
        components.update(
            processor_version=processor.version,
            connectome=None,
            encoder="complete-observation-control",
            backend_version=None,
        )
    else:
        artifact = FlyWireArtifact.load(_resolve(config, settings.artifact_path))
        encoder_spec = FlyWireEncoderSpec.from_artifact(
            artifact,
            population_width=settings.population_width,
            max_rate_hz=settings.max_rate_hz,
        )
        encoder = FixedUpstreamBalatroEncoder(encoder_spec)
        readout = artifact.descending_indices[: settings.readout_count]
        backend = TorchFlyWireBackend(
            artifact,
            device=config.runtime.device,
            readout_indices=readout,
            shuffle_seed=(settings.shuffle_seed if settings.kind == "shuffled" else None),
            record_events=settings.record_events,
        )
        processor = RealFlyProcessor(
            encoder,
            backend,
            duration_ms=settings.duration_ms,
            microbatch_size=settings.microbatch_size,
        )
        components.update(
            processor_version=processor.version,
            backend_version=backend.backend_version,
            fly_dynamics_hash=backend.dynamics_hash,
            fly_dynamics_parameters=asdict(backend.parameters),
            connectome_hash=backend.connectivity_hash,
            topology_procedure=backend.topology_procedure,
            encoder_hash=encoder_spec.sha256,
            readout_neurons=len(readout),
            feature_extractor_hash=processor.feature_extractor_hash,
        )
    hidden_size = config.policy.hidden_size
    if settings.kind == "conventional" and hidden_size == 0:
        raise ValueError("conventional control must explicitly configure modest hidden_size")
    if settings.kind in {"real", "shuffled"} and hidden_size != 0:
        raise ValueError("fly conditions use a linear readout in V1")
    policy = TorchStructuredPolicy(processor.output_size, hidden_size=hidden_size)
    ppo_values = asdict(config.ppo)
    ppo_values.pop("max_updates")
    ppo_values.pop("checkpoint_every_updates")
    trainer = PPOTrainer(
        env,
        processor,
        policy,
        PPOConfig(**ppo_values),
        training_seeds=SeedPlan().seeds("training", env.num_envs),
        device=config.runtime.device,
    )
    env.set_shaping_beta(config.reward.shaping_beta)
    components.update(
        simulator_version=env.simulator_version,
        policy_version=policy.policy_version,
        policy_hidden_size=hidden_size,
        trainable_parameter_count=policy.trainable_parameter_count,
        torch_version=torch.__version__,
    )
    return TrainingStack(env, processor, policy, trainer, components)


def build_environment(config: ExperimentConfig, *, win_ante: int) -> Any:
    settings = config.environment
    if settings.backend == "mock":
        env = MockArrayBalatroEnv(
            settings.num_envs,
            blind_target=settings.blind_target,
            initial_hands=settings.initial_hands,
            initial_discards=settings.initial_discards,
        )
        env.set_win_ante(win_ante)
        return env
    return BalatroSimAdapter(
        settings.num_envs,
        strict=settings.strict_actions,
        win_ante=win_ante,
    )


T = TypeVar("T")


def _section(cls: type[T], values: Mapping[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    converted = dict(values)
    if cls is CurriculumSettings and "ladder" in converted:
        converted["ladder"] = tuple(int(value) for value in converted["ladder"])
    return cls(**converted)


def _resolve(config: ExperimentConfig, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config.source_path.parent.parent / path)
