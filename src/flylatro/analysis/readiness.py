"""What actually remains before a real Ante-1 run.

This orchestrates the existing reports rather than re-deriving their logic, and
it never claims Ante-1 readiness from synthetic evidence: a synthetic circuit
or mock Balatro can prove software semantics, never that the real experiment is
ready to start.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from flylatro.analysis.preflight import PreflightThresholds, load_json, run_preflight
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.learning.config import (
    PlasticExperimentConfig,
    expected_corpus_environment,
    resolve_path,
)


READINESS_VERSION = "flylatro-run-readiness-v2"

READINESS_STATES = (
    "NEEDS_CONFIGURATION",
    "READY_FOR_ARTIFACT_BUILD",
    "READY_FOR_CALIBRATION",
    "READY_FOR_TINY_REAL_RUN",
    "READY_FOR_ANTE1",
)

#: What each state is allowed to mean.  Reported verbatim so a reader never has
#: to infer the semantics from the stage list.
READINESS_SEMANTICS: dict[str, str] = {
    "READY_FOR_ARTIFACT_BUILD": (
        "the software path is consistent and the real setup can begin; nothing "
        "about the real artifact or the real environment has been checked yet"
    ),
    "READY_FOR_CALIBRATION": (
        "the real FlyWire artifact loads and validates and the configured "
        "backend is usable, so reward-free calibration may begin"
    ),
    "READY_FOR_TINY_REAL_RUN": (
        "sensory, representation, motor and plasticity calibration evidence "
        "exist, are provenance-matched to this exact configuration, were "
        "produced by this experiment's real environment and pinned simulator, "
        "and satisfy the initial preflight with no FAIL"
    ),
    "READY_FOR_ANTE1": (
        "the tiny real run, its matched control, the benchmark, chosen-action "
        "specificity, the replicate protocol and every strict scientific gate "
        "satisfy the ante1 preflight with no FAIL. Failed KC/plastic-edge "
        "reachability, insufficient calibration-context coverage, a corpus from "
        "the wrong simulator or backend, a stale or wrong-kind report, or "
        "invalid motor evidence all prevent this state"
    ),
}

#: Gates that must not merely warn before an Ante-1 run is authorized.  Listed
#: explicitly so the readiness report can name the blocker.
ANTE1_BLOCKING_GATES: tuple[str, ...] = (
    "kc_subtype_reachability",
    "calibration_corpus_identity",
    "calibration_corpus_provenance",
    "calibration_corpus_coverage",
    "motor_calibration",
    "representation_post",
    "final_motor_artifact_identity",
    "motor_candidate_validity",
    "motor_normalization_validity",
)


@dataclass(frozen=True, slots=True)
class EvidencePaths:
    """Where each evidence artifact lives; defaults come from the config."""

    sensory_health: Path | None = None
    representation_pre: Path | None = None
    representation_post: Path | None = None
    motor_calibration: Path | None = None
    reachability: Path | None = None
    plasticity: Path | None = None
    specificity: Path | None = None
    benchmark: Path | None = None
    protocol: Path | None = None
    control_manifests: tuple[Path, ...] = ()

    @classmethod
    def from_config(cls, config: PlasticExperimentConfig) -> "EvidencePaths":
        def optional(value: str) -> Path | None:
            if not value:
                return None
            path = resolve_path(config, value)
            return path if path.exists() else None

        settings = config.calibration
        return cls(
            sensory_health=optional(settings.sensory_health_report),
            representation_pre=optional(settings.representation_pre_report),
            representation_post=optional(settings.representation_post_report),
            motor_calibration=optional(settings.motor_calibration_report),
            reachability=optional(settings.reachability_report),
            plasticity=optional(settings.plasticity_report),
            specificity=optional(settings.specificity_report),
            benchmark=optional(settings.benchmark_report),
        )

    def merge(self, other: "EvidencePaths") -> "EvidencePaths":
        return EvidencePaths(
            sensory_health=other.sensory_health or self.sensory_health,
            representation_pre=other.representation_pre or self.representation_pre,
            representation_post=other.representation_post or self.representation_post,
            motor_calibration=other.motor_calibration or self.motor_calibration,
            reachability=other.reachability or self.reachability,
            plasticity=other.plasticity or self.plasticity,
            specificity=other.specificity or self.specificity,
            benchmark=other.benchmark or self.benchmark,
            protocol=other.protocol or self.protocol,
            control_manifests=other.control_manifests or self.control_manifests,
        )

    def as_reports(self) -> dict[str, Any]:
        return {
            "sensory_health_report": load_json(self.sensory_health),
            "representation_pre_report": load_json(self.representation_pre),
            "representation_post_report": load_json(self.representation_post),
            "motor_calibration_report": load_json(self.motor_calibration),
            "reachability_report": load_json(self.reachability),
            "plasticity_report": load_json(self.plasticity),
            "specificity_report": load_json(self.specificity),
            "benchmark_report": load_json(self.benchmark),
            "protocol_payload": load_json(self.protocol),
            "control_manifests": tuple(
                load_json(path) or {} for path in self.control_manifests
            ),
        }


def run_readiness(
    config: PlasticExperimentConfig,
    evidence: EvidencePaths,
    *,
    thresholds: PreflightThresholds | None = None,
) -> dict[str, Any]:
    reports = evidence.as_reports()
    limits = thresholds or PreflightThresholds()
    initial = run_preflight(config, profile="initial", thresholds=limits, **reports)
    strict = run_preflight(config, profile="ante1", thresholds=limits, **reports)
    stages = _stages(config, evidence, initial, strict)
    state = "NEEDS_CONFIGURATION"
    for stage in stages:
        if stage["satisfied"]:
            state = stage["unlocks"]
        else:
            break
    real_experiment = (
        config.fly.backend == "flywire" and config.environment.backend == "balatro_sim"
    )
    synthetic_note = None
    if not real_experiment and state in {
        "READY_FOR_TINY_REAL_RUN",
        "READY_FOR_ANTE1",
    }:
        # A synthetic circuit or mock Balatro can prove software semantics. It
        # can never be evidence that a *real* experiment is ready to start.
        state = "READY_FOR_CALIBRATION"
        synthetic_note = (
            "capped at READY_FOR_CALIBRATION: this configuration uses "
            f"fly.backend={config.fly.backend} and environment.backend="
            f"{config.environment.backend}, so all of its evidence is synthetic "
            "and cannot authorize a real run"
        )
    return {
        "version": READINESS_VERSION,
        "config": str(config.source_path),
        "config_sha256": config.sha256,
        "state": state,
        "states": list(READINESS_STATES),
        "state_semantics": dict(READINESS_SEMANTICS),
        "ante1_blocking_gates": [
            {"name": item["name"], "status": item["status"], "reason": item["reason"]}
            for item in strict["gates"]
            if item["name"] in ANTE1_BLOCKING_GATES and item["status"] != "PASS"
        ],
        "synthetic_evidence_cap": synthetic_note,
        "real_experiment_configuration": real_experiment,
        "stages": stages,
        "remaining": [
            {"stage": stage["stage"], "blocked_by": stage["blocked_by"], "next": stage["next_command"]}
            for stage in stages
            if not stage["satisfied"]
        ],
        "preflight_initial": {
            "status": initial["status"],
            "failed": initial["failed"],
            "warnings": initial["warnings"],
        },
        "preflight_ante1": {
            "status": strict["status"],
            "failed": strict["failed"],
            "warnings": strict["warnings"],
        },
    }


def _stages(
    config: PlasticExperimentConfig,
    evidence: EvidencePaths,
    initial: Mapping[str, Any],
    strict: Mapping[str, Any],
) -> list[dict[str, Any]]:
    software_ok, software_reason = _software(config)
    artifact_ok, artifact_reason = _artifact(config)
    corpus_ok, corpus_reason = _corpus(config)
    initial_failed = list(initial["failed"])
    strict_failed = list(strict["failed"])
    return [
        {
            "stage": "software_and_configuration",
            "unlocks": "READY_FOR_ARTIFACT_BUILD",
            "satisfied": software_ok,
            "blocked_by": [] if software_ok else [software_reason],
            "next_command": "python -m pytest",
        },
        {
            "stage": "flywire_artifact",
            "unlocks": "READY_FOR_CALIBRATION",
            "satisfied": artifact_ok,
            "blocked_by": [] if artifact_ok else [artifact_reason],
            "next_command": (
                "python -m flylatro.fly.build_flywire --source-dir "
                '"$FLYLATRO_FLYWIRE_SOURCE" --output-dir data/flywire --full'
            ),
        },
        {
            "stage": "calibration_evidence",
            "unlocks": "READY_FOR_TINY_REAL_RUN",
            "satisfied": bool(corpus_ok and not initial_failed),
            "blocked_by": ([] if corpus_ok else [corpus_reason])
            + [f"preflight-initial FAIL: {name}" for name in initial_failed],
            "next_command": (
                "flylatro-build-calibration-corpus ... then flylatro-preflight "
                "--profile initial"
            ),
        },
        {
            "stage": "measured_evidence_and_controls",
            "unlocks": "READY_FOR_ANTE1",
            "satisfied": not strict_failed,
            "blocked_by": [f"preflight-ante1 FAIL: {name}" for name in strict_failed],
            "next_command": "flylatro-preflight --profile ante1",
        },
    ]


def _software(config: PlasticExperimentConfig) -> tuple[bool, str]:
    missing = [
        name
        for name in ("numpy", "torch")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return False, f"missing required packages: {missing}"
    if config.fly.backend == "flywire" and "PLACEHOLDER" in config.training.budget_basis:
        return True, "configuration loads; the exposure budget is still a placeholder"
    return True, "configuration loads and required packages are installed"


def _artifact(config: PlasticExperimentConfig) -> tuple[bool, str]:
    if config.fly.backend != "flywire":
        return True, "synthetic backend needs no FlyWire artifact"
    path = resolve_path(config, config.fly.artifact_path)
    if not path.exists():
        return False, f"FlyWire artifact is missing: {path}"
    try:
        FlyWireArtifact.load(path).validate()
    except Exception as error:
        return False, f"FlyWire artifact does not validate: {error}"
    return True, "FlyWire artifact validates"


def _corpus(config: PlasticExperimentConfig) -> tuple[bool, str]:
    if not config.calibration.corpus_path:
        return False, "no frozen calibration corpus is configured"
    path = resolve_path(config, config.calibration.corpus_path)
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest.exists():
        return False, f"calibration corpus manifest is missing: {manifest}"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("contains_reward_or_strategy_labels") is not False:
        return False, "calibration corpus does not declare itself reward-free"
    backend, simulator = expected_corpus_environment(config)
    if str(payload.get("environment_backend")) != backend:
        return False, (
            "calibration corpus was generated by "
            f"{payload.get('environment_backend')!r}, not this experiment's "
            f"{backend!r}"
        )
    if str(payload.get("simulator_version")) != simulator:
        return False, (
            "calibration corpus records simulator "
            f"{payload.get('simulator_version')!r}, not the pinned {simulator!r}"
        )
    return True, (
        "frozen reward-free calibration corpus is present and was generated by "
        f"{backend} @ {simulator}"
    )
