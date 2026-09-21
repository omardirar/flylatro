"""Benchmark entry points. Real connectome and large runs require --heavy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import numpy as np

from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.upstream_contract import MASK_SPEC, UpstreamActionType
from flylatro.evaluation.baselines import RandomLegalPolicy
from flylatro.fly.backend import TinyGraphFlyBackend
from flylatro.fly.encoder import Stimulus


def balatro_main(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Balatro environment throughput")
    parser.add_argument("--backend", choices=("mock", "real"), default="mock")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    _guard(args, real=args.backend == "real", work=args.num_envs * args.steps)
    env = (
        MockArrayBalatroEnv(args.num_envs)
        if args.backend == "mock"
        else BalatroSimAdapter(args.num_envs)
    )
    _, masks = env.reset(tuple(range(args.num_envs)))
    policy = RandomLegalPolicy(19)
    start = time.perf_counter()
    for _ in range(args.steps):
        output = policy.act(masks)
        step = env.step(output.actions)
        masks = step.masks
    seconds = time.perf_counter() - start
    decisions = args.num_envs * args.steps
    details = {
        "num_envs": args.num_envs,
        "vector_steps": args.steps,
        "environment_steps_per_second": decisions / seconds,
    }
    _emit("balatro", args.backend, decisions, seconds, details)
    return 0


def fly_main(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Fly simulation throughput")
    parser.add_argument("--backend", choices=("tiny", "flywire"), default="tiny")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--durations-ms", default="1,5,10")
    parser.add_argument("--readout-size", type=int, default=32)
    args = parser.parse_args(argv)
    batches = _csv_ints(args.batch_sizes)
    durations = _csv_floats(args.durations_ms)
    _guard(
        args,
        real=args.backend == "flywire" or args.device.startswith("cuda"),
        work=max(batches) * max(durations),
    )
    rows = []
    for batch in batches:
        for duration in durations:
            if args.backend == "tiny":
                backend: Any = TinyGraphFlyBackend(96)
                neuron_count = backend.neuron_count
            else:
                if args.artifact is None:
                    parser.error("--artifact is required for FlyWire")
                from flylatro.fly.flywire_artifact import FlyWireArtifact
                from flylatro.fly.torch_backend import TorchFlyWireBackend

                artifact = FlyWireArtifact.load(args.artifact)
                readout = artifact.descending_indices[: args.readout_size]
                backend = TorchFlyWireBackend(
                    artifact, device=args.device, readout_indices=readout
                )
                neuron_count = artifact.neuron_count
            rates = np.zeros((batch, neuron_count), dtype=np.float32)
            rates[:, : min(8, neuron_count)] = 50.0
            stimulus = Stimulus(rates, "benchmark-v1", "benchmark")
            backend.reset(batch, tuple(range(batch)))
            if args.device.startswith("cuda"):
                import torch

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            backend.simulate(stimulus, duration)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            row = _metrics(batch, seconds)
            row.update(batch_size=batch, duration_ms=duration)
            rows.append(row)
    print(json.dumps({"benchmark": "fly", "backend": args.backend, "results": rows}))
    return 0


def policy_main(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Structured policy throughput")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=0)
    args = parser.parse_args(argv)
    _guard(
        args,
        real=args.device.startswith("cuda"),
        work=args.batch_size * args.iterations,
    )
    import torch

    from flylatro.policy.torch_structured import TorchStructuredPolicy, masks_to_torch

    policy = TorchStructuredPolicy(
        args.feature_size, hidden_size=args.hidden_size
    ).to(args.device)
    features = torch.zeros(
        (args.batch_size, args.feature_size), device=args.device
    )
    masks = _play_masks(args.batch_size)
    tensor_masks = masks_to_torch(masks, args.device)
    with torch.no_grad():
        policy(features, tensor_masks, deterministic=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            policy(features, tensor_masks, deterministic=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    _emit(
        "policy", args.device, args.batch_size * args.iterations, seconds,
        {
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "feature_size": args.feature_size,
            "trainable_parameters": policy.trainable_parameter_count,
        },
    )
    return 0


def end_to_end_main(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Environment-to-policy rollout throughput")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.toml"))
    args = parser.parse_args(argv)
    from flylatro.training.config import ExperimentConfig, build_training_stack

    config = ExperimentConfig.load(args.config)
    config.require_heavy_opt_in(args.heavy)
    stack = build_training_stack(config)
    _, metrics = stack.trainer.collect_rollout()
    print(json.dumps({"benchmark": "end_to_end", "metrics": metrics}, sort_keys=True))
    return 0


def plasticity_main(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Sparse KC->MBON plasticity update throughput")
    parser.add_argument("--edges", type=int, default=10_000)
    parser.add_argument("--learners", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(argv)
    work = args.edges * args.learners * args.iterations
    _guard(args, real=False, work=work)
    from flylatro.fly.mushroom_body.plasticity import ThreeFactorPlasticity
    from flylatro.fly.mushroom_body.state import PlasticEdgeState
    from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology

    pre = np.arange(args.edges, dtype=np.int64)
    topology = PlasticEdgeTopology.synthetic(
        pre,
        pre + args.edges,
        np.ones(args.edges, dtype=np.float32),
    )
    state = PlasticEdgeState.initialize(args.edges, learners=args.learners)
    rule = ThreeFactorPlasticity(topology, state)
    activity = np.ones((args.learners, args.edges), dtype=np.float32)
    start = time.perf_counter()
    for _ in range(args.iterations):
        rule.record_activity(activity, activity)
        rule.apply_dopamine(0.1, 0.0)
    seconds = time.perf_counter() - start
    _emit(
        "plasticity",
        "numpy-sparse-edge-vector",
        args.learners * args.iterations,
        seconds,
        {
            "plastic_edges": args.edges,
            "learners": args.learners,
            "iterations": args.iterations,
            "edge_updates_per_second": work / seconds,
        },
    )
    return 0


def plastic_end_to_end_main(argv: Sequence[str] | None = None) -> int:
    parser = _base_parser("Plastic fly environment-decision throughput")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/plastic-smoke.toml")
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-mode", choices=("mbon_direct", "whole_brain"))
    parser.add_argument(
        "--compare-sequential",
        action="store_true",
        help="measure one shared batched simulation against row-by-row execution",
    )
    args = parser.parse_args(argv)
    from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack

    config = PlasticExperimentConfig.load(args.config)
    if args.output_mode is not None:
        from dataclasses import replace

        config = replace(config, fly=replace(config.fly, mode=args.output_mode))
    config.require_heavy_opt_in(args.heavy)
    stack = build_plastic_stack(config)
    if config.fly.device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    last = {}
    for _ in range(args.steps):
        last = stack.trainer.step()
    if config.fly.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    decisions = args.steps * stack.env.num_envs
    comparison = None
    if args.compare_sequential:
        from flylatro.fly.mushroom_body.state import state_numpy

        observations = stack.trainer.observations
        efficacy = state_numpy(stack.agent.plasticity.state.efficacy)
        fly_seeds = tuple(91_000_000 + row for row in range(stack.env.num_envs))
        if config.fly.device.startswith("cuda"):
            torch.cuda.synchronize()
        batch_started = time.perf_counter()
        stack.agent.processor.process(observations, fly_seeds=fly_seeds, efficacy=efficacy)
        if config.fly.device.startswith("cuda"):
            torch.cuda.synchronize()
        batch_seconds = time.perf_counter() - batch_started
        if config.fly.device.startswith("cuda"):
            torch.cuda.synchronize()
        sequential_started = time.perf_counter()
        for row in range(stack.env.num_envs):
            stack.agent.processor.process(
                {key: value[row:row + 1] for key, value in observations.items()},
                fly_seeds=(fly_seeds[row],),
                efficacy=efficacy[row:row + 1],
            )
        if config.fly.device.startswith("cuda"):
            torch.cuda.synchronize()
        sequential_seconds = time.perf_counter() - sequential_started
        comparison = {
            "batch_size": stack.env.num_envs,
            "batched_seconds": batch_seconds,
            "sequential_seconds": sequential_seconds,
            "measured_speedup": sequential_seconds / max(batch_seconds, 1e-12),
            "memory_model": "one shared fixed sparse graph plus batched plastic KC-MBON edge contribution",
        }
    _emit(
        "plastic_end_to_end",
        config.fly.backend,
        decisions,
        seconds,
        {
            "output_mode": config.fly.mode,
            "plastic_edges": stack.agent.plasticity.topology.edge_count,
            "learners": stack.agent.plasticity.state.learners,
            "batch_size": stack.env.num_envs,
            "execution_mode": "shared-fixed-sparse-plus-batched-plastic",
            "neural_execution_comparison": comparison,
            "external_trainable_parameters": 0,
            "last_metrics": last,
        },
    )
    return 0


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--heavy", action="store_true")
    return parser


def _guard(args: argparse.Namespace, *, real: bool, work: float) -> None:
    if (real or work > 10_000) and not args.heavy:
        raise ValueError("real or large benchmark requires explicit --heavy")


def _play_masks(batch: int) -> dict[str, np.ndarray]:
    masks = {
        key: np.zeros((batch, *shape), dtype=dtype)
        for key, (shape, dtype) in MASK_SPEC.items()
    }
    masks["action_type_mask"][:, int(UpstreamActionType.PLAY_HAND)] = True
    masks["card_select_mask"][:, :5] = True
    return masks


def _metrics(decisions: int, seconds: float) -> dict[str, float]:
    result = {
        "wall_seconds": seconds,
        "decisions_per_second": decisions / seconds,
        "process_peak_rss_mb": _peak_rss_mb(),
        "peak_memory_bytes": int(_peak_rss_mb() * 1024**2),
    }
    try:
        import torch
    except ImportError:
        return result
    if torch.cuda.is_available():
        result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        result["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    return result


def _emit(
    benchmark: str, backend: str, decisions: int, seconds: float,
    details: dict[str, Any],
) -> None:
    print(
        json.dumps(
            {"benchmark": benchmark, "backend": backend, **details,
             **_metrics(decisions, seconds)},
            sort_keys=True,
        )
    )


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 if value < 10**10 else 1024.0**2)


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or min(result) < 1:
        raise ValueError("list values must be positive integers")
    return result


def _csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or min(result) <= 0:
        raise ValueError("list values must be positive")
    return result
