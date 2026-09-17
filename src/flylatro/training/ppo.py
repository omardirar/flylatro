"""Compact PPO trainer over fixed fly features and structured actions."""

from __future__ import annotations

from dataclasses import dataclass
import time
import resource
from typing import Any, Sequence

import numpy as np
import torch

from flylatro.env.balatro_sim import ArrayBalatroEnv
from flylatro.fly.processors import ObservationProcessor
from flylatro.policy.torch_structured import (
    TorchStructuredPolicy,
    actions_to_numpy,
    masks_to_torch,
)
from flylatro.seeds import derive_seed, seed_everything
from flylatro.training.rollout import FlatRollout, RolloutBuffer


@dataclass(frozen=True, slots=True)
class PPOConfig:
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
    target_kl: float | None = None
    seed: int = 1
    deterministic_torch: bool = False

    def __post_init__(self) -> None:
        if min(self.rollout_steps, self.update_epochs, self.minibatch_size) < 1:
            raise ValueError("PPO rollout/update sizes must be positive")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("invalid PPO learning rate or gamma")
        if not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_coefficient < 1:
            raise ValueError("invalid PPO GAE or clip coefficient")


class PPOTrainer:
    episode_history_limit = 10_000

    def __init__(
        self,
        env: ArrayBalatroEnv,
        processor: ObservationProcessor,
        policy: TorchStructuredPolicy,
        config: PPOConfig,
        *,
        training_seeds: Sequence[int],
        device: str = "cpu",
    ) -> None:
        if len(training_seeds) != env.num_envs:
            raise ValueError("one initial training seed is required per env")
        if policy.feature_size != processor.output_size:
            raise ValueError("processor output and policy feature sizes differ")
        self.env = env
        self.processor = processor
        self.policy = policy.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.training_seeds = tuple(int(seed) for seed in training_seeds)
        self.global_step = 0
        self.update_index = 0
        self.decision_index = 0
        self.observations = None
        self.masks = None
        self.episode_metrics: list[dict[str, Any]] = []
        seed_everything(config.seed, deterministic_torch=config.deterministic_torch)

    def reset(self) -> None:
        self.observations, self.masks = self.env.reset(self.training_seeds)

    def collect_rollout(self) -> tuple[FlatRollout, dict[str, float]]:
        if self.observations is None or self.masks is None:
            self.reset()
        assert self.observations is not None and self.masks is not None
        buffer = RolloutBuffer()
        reward_components: dict[str, float] = {}
        completed: list[dict[str, Any]] = []
        env_seconds = 0.0
        processor_seconds = 0.0
        start = time.perf_counter()
        for _ in range(self.config.rollout_steps):
            fly_seeds = self._fly_seeds()
            processor_start = time.perf_counter()
            features = self.processor.process(
                self.observations, fly_seeds=fly_seeds
            ).to(self.device)
            processor_seconds += time.perf_counter() - processor_start
            tensor_masks = masks_to_torch(self.masks, self.device)
            with torch.no_grad():
                policy_output = self.policy(features, tensor_masks)
            env_start = time.perf_counter()
            step = self.env.step(actions_to_numpy(policy_output.actions))
            env_seconds += time.perf_counter() - env_start
            reward = torch.as_tensor(
                step.rewards, dtype=torch.float32, device=self.device
            )
            done = torch.as_tensor(step.dones, dtype=torch.bool, device=self.device)
            buffer.add(
                features=features,
                masks=tensor_masks,
                actions=policy_output.actions,
                log_probability=policy_output.log_probability,
                value=policy_output.value,
                reward=reward,
                done=done,
            )
            for info in step.infos:
                for key, value in info.get("reward_components", {}).items():
                    reward_components[key] = reward_components.get(key, 0.0) + float(
                        value
                    )
                if "episode" in info:
                    episode = dict(info["episode"])
                    completed.append(episode)
                    self.episode_metrics.append(episode)
                    excess = len(self.episode_metrics) - self.episode_history_limit
                    if excess > 0:
                        del self.episode_metrics[:excess]
            self.observations, self.masks = step.observations, step.masks
            self.global_step += self.env.num_envs
            self.decision_index += 1
        with torch.no_grad():
            final_features = self.processor.process(
                self.observations, fly_seeds=self._fly_seeds()
            ).to(self.device)
            final_value = self.policy(
                final_features, masks_to_torch(self.masks, self.device)
            ).value
        rollout = buffer.finish(
            final_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        wall_seconds = time.perf_counter() - start
        decisions = self.config.rollout_steps * self.env.num_envs
        metrics = {
            "rollout/decisions": float(decisions),
            "rollout/wall_seconds": wall_seconds,
            "throughput/end_to_end_decisions_per_second": decisions / wall_seconds,
            "throughput/environment_steps_per_second": (
                decisions / env_seconds if env_seconds else float("inf")
            ),
            "throughput/fly_decisions_per_second": (
                decisions / processor_seconds if processor_seconds else float("inf")
            ),
            "rollout/mean_reward": float(
                torch.stack(buffer.rewards).mean().cpu()
            ),
            "system/process_peak_rss_mb": _peak_rss_mb(),
        }
        if completed:
            returns = [float(episode.get("r", 0.0)) for episode in completed]
            antes = [float(episode.get("ante", 0.0)) for episode in completed]
            lengths = [float(episode.get("l", 0.0)) for episode in completed]
            metrics.update(
                {
                    "episodes/completed": float(len(completed)),
                    "episodes/mean_return": float(np.mean(returns)),
                    "episodes/mean_ante": float(np.mean(antes)),
                    "episodes/median_ante": float(np.median(antes)),
                    "episodes/win_rate": float(
                        np.mean([bool(item.get("won", False)) for item in completed])
                    ),
                    "episodes/mean_length": float(np.mean(lengths)),
                    "throughput/episodes_per_hour": (
                        len(completed) / wall_seconds * 3600.0
                    ),
                }
            )
        else:
            metrics["episodes/completed"] = 0.0
        if self.device.type == "cuda":
            metrics["system/gpu_allocated_mb"] = (
                torch.cuda.memory_allocated(self.device) / 1024**2
            )
            metrics["system/gpu_peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
        for key, value in reward_components.items():
            metrics[f"reward/{key}"] = value / decisions
        return rollout, metrics

    def update(self, rollout: FlatRollout) -> dict[str, float]:
        sample_count = rollout.features.shape[0]
        if self.config.minibatch_size > sample_count:
            raise ValueError("minibatch_size exceeds rollout sample count")
        advantages = rollout.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        rollout = FlatRollout(
            features=rollout.features,
            masks=rollout.masks,
            actions=rollout.actions,
            old_log_probability=rollout.old_log_probability,
            old_value=rollout.old_value,
            advantages=advantages,
            returns=rollout.returns,
        )
        metric_lists: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "gradient_norm": [],
        }
        stop_for_kl = False
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                batch = rollout.select(indices)
                output = self.policy(
                    batch.features, batch.masks, actions=batch.actions
                )
                log_ratio = output.log_probability - batch.old_log_probability
                ratio = log_ratio.exp()
                unclipped = -batch.advantages * ratio
                clipped = -batch.advantages * ratio.clamp(
                    1 - self.config.clip_coefficient,
                    1 + self.config.clip_coefficient,
                )
                policy_loss = torch.maximum(unclipped, clipped).mean()
                value_loss = 0.5 * (output.value - batch.returns).square().mean()
                entropy = output.entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_gradient_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_coefficient
                    ).float().mean()
                for key, value in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                    ("gradient_norm", gradient_norm),
                ):
                    metric_lists[key].append(float(value.detach().cpu()))
                if (
                    self.config.target_kl is not None
                    and float(approx_kl) > self.config.target_kl
                ):
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break
        self.update_index += 1
        predicted = rollout.old_value
        target = rollout.returns
        variance = torch.var(target)
        explained_variance = (
            float(1 - torch.var(target - predicted) / variance)
            if float(variance) > 0
            else float("nan")
        )
        return {
            f"train/{key}": float(np.mean(values))
            for key, values in metric_lists.items()
        } | {
            "train/explained_variance": explained_variance,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            "train/update": float(self.update_index),
            "train/global_step": float(self.global_step),
        }

    def train_update(self) -> dict[str, float]:
        rollout, rollout_metrics = self.collect_rollout()
        return rollout_metrics | self.update(rollout)

    def _fly_seeds(self) -> tuple[int, ...]:
        return tuple(
            derive_seed("fly", self.config.seed, self.decision_index, env_index)
            for env_index in range(self.env.num_envs)
        )


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value / (1024.0 if value < 10**10 else 1024.0**2)
