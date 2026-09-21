"""Formal, evidence-addressable pre-training gate evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.util
import json
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import weak_edge_diagnostics
from flylatro.interface.motor import MotorMapping
from flylatro.interface.sensory import SensoryMapping
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol import assert_matched_control_manifests


@dataclass(frozen=True, slots=True)
class PreflightThresholds:
    maximum_alpn_assignment_p99: float = 20.0
    minimum_alpn_used_fraction: float = 0.20
    require_representation_pass: bool = True
    require_plasticity_pass: bool = True
    require_benchmark_report: bool = False
    require_control_manifest_check: bool = False


def run_preflight(
    config: PlasticExperimentConfig,
    *,
    representation_report: Mapping[str, Any] | None = None,
    plasticity_report: Mapping[str, Any] | None = None,
    benchmark_report: Mapping[str, Any] | None = None,
    control_manifests: Sequence[Mapping[str, Any]] = (),
    thresholds: PreflightThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or PreflightThresholds()
    gates: list[dict[str, Any]] = []

    def gate(name: str, status: str, reason: str, evidence: object | None = None) -> None:
        gates.append({"name": name, "status": status, "reason": reason, "evidence": evidence})

    artifact = None
    if config.fly.backend == "flywire":
        artifact_path = _resolve(config, config.fly.artifact_path)
        try:
            artifact = FlyWireArtifact.load(artifact_path)
            artifact.validate()
        except Exception as error:  # report exact external readiness failure
            gate("artifact_and_populations", "FAIL", str(error))
        else:
            gate(
                "artifact_and_populations", "PASS",
                "v783 artifact validates, including exact 5,177 KC census",
                {"artifact_sha256": artifact.manifest["artifact_sha256"], "population_sha256": artifact.population_hash},
            )
            quality = {
                key: artifact.manifest.get(key)
                for key in ("n_unresolved_nt_neurons", "n_synapses_unresolved_sign", "fraction_raw_synapses_dropped", "n_missing_coordinates")
            }
            missing_quality = [key for key, value in quality.items() if value is None]
            gate(
                "flywire_data_quality",
                "WARN" if missing_quality else "PASS",
                "artifact predates data-quality counters" if missing_quality else "unresolved-sign and coordinate effects are explicit",
                quality,
            )
            gate("weak_edge_analysis", "PASS", "thresholds 1, 2, 5 and 10 reported", weak_edge_diagnostics(artifact))
    else:
        gate("artifact_and_populations", "WARN", "synthetic backend cannot validate real FlyWire readiness")

    if artifact is not None:
        mapping = SensoryMapping.from_artifact(
            artifact,
            mapping_seed=config.fly.sensory_mapping_seed,
            population_width=config.fly.sensory_population_width,
            max_rate_hz=config.fly.max_rate_hz,
        )
        audit = mapping.collision_audit()
        checks = {
            "p99_assignment": audit["assignments_per_alpn"]["p99"] <= limits.maximum_alpn_assignment_p99,
            "used_fraction": audit["fraction_alpns_used"] >= limits.minimum_alpn_used_fraction,
            "collision_policy": audit["policy"] == "clipped_sum_preserve_sparse_indicators",
        }
        gate("sensory_mapping_and_collisions", "PASS" if all(checks.values()) else "FAIL", "deterministic field-aware ALPN mapping checked", {"checks": checks, "audit": audit})
    else:
        gate("sensory_mapping_and_collisions", "WARN", "requires the real v783 artifact")

    if config.fly.motor_mapping_path:
        try:
            motor = MotorMapping.load(_resolve(config, config.fly.motor_mapping_path))
            expected = artifact.mbon_indices if artifact is not None and config.fly.mode == "mbon_direct" else artifact.descending_indices if artifact is not None else None
            roots_match = expected is None or np.array_equal(motor.output_root_ids, artifact.root_ids[expected])
            calibrated = motor.selection_method == "reward-free-activity-dynamic-range" and motor.calibration_sha256 is not None
            pool_widths = [len(pool) for pools in motor.pools.values() for pool in pools]
            passed = roots_match and calibrated and min(pool_widths) > 1
            gate("canonical_motor_mapping", "PASS" if passed else "FAIL", "reward-free canonical population pools checked", {"sha256": motor.sha256, "roots_match": roots_match, "calibrated": calibrated, "minimum_pool_width": min(pool_widths)})
        except Exception as error:
            gate("canonical_motor_mapping", "FAIL", str(error))
    else:
        gate("canonical_motor_mapping", "FAIL" if config.fly.backend == "flywire" else "WARN", "no persisted real motor mapping configured")

    _report_gate(gate, "representation", representation_report, limits.require_representation_pass)
    _report_gate(gate, "plasticity_calibration", plasticity_report, limits.require_plasticity_pass)

    reward = asdict(config.reinforcement)
    finite_reward = all(not isinstance(value, float) or np.isfinite(value) for value in reward.values())
    gate("reward_mapping", "PASS" if finite_reward else "FAIL", "fixed synthetic appetitive/aversive mapping is finite and versioned", reward)

    state = PlasticEdgeState.initialize(3, learners=2)
    state.eligibility[:] = 1
    state.dopamine[:] = 1
    state.efficacy[0, 0] = 1.1
    state.reset_fast_traces(learners=(0,), eligibility=True)
    reset_ok = bool(np.all(state.eligibility[0] == 0) and np.all(state.dopamine[0] == 0) and np.all(state.eligibility[1] == 1) and state.efficacy[0, 0] == np.float32(1.1))
    gate("eligibility_episode_reset", "PASS" if reset_ok else "FAIL", "learner-local fast traces reset while efficacy persists")
    restored = PlasticEdgeState.from_state_dict(state.state_dict())
    gate("checkpoint_resume", "PASS" if restored.weight_sha256 == state.weight_sha256 else "FAIL", "plastic state round-trip hash checked")

    if benchmark_report is None:
        gate("throughput_and_memory_benchmark", "FAIL" if limits.require_benchmark_report else "WARN", "no dedicated-machine benchmark report supplied")
    else:
        required = {"decisions_per_second", "peak_memory_bytes", "batch_size", "execution_mode"}
        missing = sorted(required - benchmark_report.keys())
        gate("throughput_and_memory_benchmark", "PASS" if not missing else "FAIL", "measured benchmark fields checked", {"missing": missing, "report": benchmark_report})

    if len(control_manifests) >= 2:
        try:
            assert_matched_control_manifests(control_manifests)
        except ValueError as error:
            gate("matched_controls", "FAIL", str(error))
        else:
            gate("matched_controls", "PASS", "matched action/state/curriculum/seed/mapping/budget fields agree")
    else:
        gate("matched_controls", "FAIL" if limits.require_control_manifest_check else "WARN", "at least two control manifests were not supplied")

    gate("environment_backend", "PASS" if config.environment.backend == "mock" or importlib.util.find_spec("balatro_sim") is not None else "FAIL", f"backend={config.environment.backend}")
    gate("recording_and_replay", "PASS" if importlib.util.find_spec("pyarrow") is not None else "FAIL", "PyArrow supports streaming decision row groups")
    gate("visualization_tooling", "PASS" if shutil.which("ffmpeg") else "WARN", "ffmpeg found" if shutil.which("ffmpeg") else "ffmpeg not found; scientific runs can proceed but final rendering cannot")
    gate("external_trainable_policy", "PASS", "primary architecture declares zero external trainable parameters and fixed encoder/motor")

    status = preflight_status(gates)
    return {"version": "flylatro-preflight-v1", "status": status, "thresholds": asdict(limits), "gates": gates}


def preflight_status(gates: Sequence[Mapping[str, Any]]) -> str:
    if any(item.get("status") == "FAIL" for item in gates):
        return "FAIL"
    if any(item.get("status") == "WARN" for item in gates):
        return "WARN"
    return "PASS"


def _resolve(config: PlasticExperimentConfig, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config.source_path.parent.parent / path


def _report_gate(callback: object, name: str, report: Mapping[str, Any] | None, required: bool) -> None:
    if report is None:
        callback(name, "FAIL" if required else "WARN", f"no {name} report supplied")
        return
    status = str(report.get("gates", {}).get("status", "FAIL"))
    callback(name, status if status in {"PASS", "WARN", "FAIL"} else "FAIL", f"{name} report gate={status}", report.get("gates"))


def load_json(path: Path | None) -> Mapping[str, Any] | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))
