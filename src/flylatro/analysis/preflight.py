"""Formal, evidence-addressable pre-training gate evaluation.

A gate never trusts a report's own ``PASS``.  Every supplied report must also
prove, through its recorded evidence identity, that it belongs to *this*
configuration: the same artifact, topology, sensory mapping, calibration
corpus, motor mapping, plasticity rule and neural duration.  Stale evidence
fails with an exact mismatch message.

Two profiles exist.  ``initial`` authorises the first tiny real experiments and
downgrades the measurements that can only happen later.  ``ante1`` is the
strict gate before significant Ante-1 training and requires all of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from flylatro.analysis.corpus import (
    CalibrationCorpus,
    SUPPORTED_CALIBRATION_CORPUS_VERSIONS,
)
from flylatro.analysis.coverage import CoveragePolicy, evaluate_coverage
from flylatro.analysis.provenance import (
    EvidenceIdentity,
    describe_mismatches,
    git_identity,
)
from flylatro.analysis.reachability import REACHABILITY_REPORT_VERSION
from flylatro.analysis.representation import REPRESENTATION_REPORT_VERSION
from flylatro.analysis.sensory_health import SENSORY_HEALTH_VERSION
from flylatro.analysis.specificity import SPECIFICITY_REPORT_VERSION
from flylatro.interface.motor_calibration import MOTOR_CALIBRATION_VERSION
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.fly.mushroom_body.state import PlasticEdgeState
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology, weak_edge_diagnostics
from flylatro.interface.motor import (
    RESERVED_ACTION_TYPES,
    SUPPORTED_ACTION_TYPES,
    MotorMapping,
)
from flylatro.interface.motor_candidates import canonical_motor_candidates
from flylatro.interface.sensory import SensoryMapping
from flylatro.learning.config import (
    SYNTHETIC_ARTIFACT_IDENTITY,
    SYNTHETIC_SENSORY_IDENTITY,
    PlasticExperimentConfig,
    calibration_corpus_hash,
    coverage_policy,
    expected_corpus_environment,
    fly_dynamics_payload,
    payload_sha256,
    resolve_path,
    synthetic_circuit_spec,
)
from flylatro.learning.protocol import ExperimentProtocol, validate_control_group


PREFLIGHT_VERSION = "flylatro-preflight-v3"

@dataclass(frozen=True, slots=True)
class ReportKindPolicy:
    """What *kind* of artifact a gate will accept, and which versions.

    Verifying component hashes proves the evidence was measured under this
    configuration.  It does not prove the supplied file is the report the gate
    asked for: a sensory-health report and a motor-calibration report can carry
    identical component hashes.  Each gate therefore also pins the report kind
    and an explicit list of compatible report versions — compatibility is
    declared, never inferred from an arbitrary version string.
    """

    kind: str
    compatible_versions: tuple[str, ...]


#: The only report kinds and versions the current gates accept.  A version is
#: listed here only when this build has been checked against that report's
#: exact field layout; anything else must be regenerated.
REPORT_KIND_POLICY: dict[str, ReportKindPolicy] = {
    "sensory_health": ReportKindPolicy(
        "sensory_health", (SENSORY_HEALTH_VERSION,)
    ),
    "representation_pre": ReportKindPolicy(
        "representation_pre", (REPRESENTATION_REPORT_VERSION,)
    ),
    "representation_post": ReportKindPolicy(
        "representation_post", (REPRESENTATION_REPORT_VERSION,)
    ),
    "motor_calibration": ReportKindPolicy(
        "motor_calibration", (MOTOR_CALIBRATION_VERSION,)
    ),
    "kc_reachability": ReportKindPolicy(
        "kc_reachability", (REACHABILITY_REPORT_VERSION,)
    ),
    "plasticity_calibration": ReportKindPolicy(
        "plasticity_calibration", ("plasticity-calibration-v2",)
    ),
    "chosen_action_specificity": ReportKindPolicy(
        "chosen_action_specificity", (SPECIFICITY_REPORT_VERSION,)
    ),
    "benchmark": ReportKindPolicy(
        "benchmark", ("plastic-end-to-end-benchmark-v2",)
    ),
}


#: Identity fields each report kind must bind to the current experiment.
REPORT_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "sensory_health": (
        "artifact_sha256",
        "sensory_mapping_sha256",
        "sensory_feature_contract_sha256",
        "calibration_corpus_sha256",
    ),
    "representation_pre": (
        "artifact_sha256",
        "population_sha256",
        "plastic_topology_sha256",
        "minimum_synapse_count",
        "sensory_mapping_sha256",
        "calibration_corpus_sha256",
        "decision_duration_ms",
        "fly_dynamics_sha256",
        "output_mode",
        "fly_backend",
    ),
    "representation_post": (
        "artifact_sha256",
        "population_sha256",
        "plastic_topology_sha256",
        "minimum_synapse_count",
        "sensory_mapping_sha256",
        "calibration_corpus_sha256",
        "decision_duration_ms",
        "fly_dynamics_sha256",
        "output_mode",
        "fly_backend",
        "motor_mapping_sha256",
        "motor_candidate_set_sha256",
    ),
    "motor_calibration": (
        "artifact_sha256",
        "sensory_mapping_sha256",
        "calibration_corpus_sha256",
        "decision_duration_ms",
        "fly_dynamics_sha256",
        "output_mode",
        "motor_candidate_set_sha256",
        "motor_mapping_sha256",
    ),
    "kc_reachability": (
        "artifact_sha256",
        "plastic_topology_sha256",
        "minimum_synapse_count",
        "sensory_mapping_sha256",
        "calibration_corpus_sha256",
        "decision_duration_ms",
        "fly_dynamics_sha256",
    ),
    "plasticity_calibration": (
        "plasticity_rule_sha256",
        "plastic_topology_sha256",
        "minimum_synapse_count",
        "artifact_sha256",
    ),
    "chosen_action_specificity": (
        "plasticity_rule_sha256",
        "plastic_topology_sha256",
        "motor_mapping_sha256",
        "reinforcement_mapping_sha256",
    ),
    "benchmark": ("fly_backend", "fly_dynamics_sha256", "device"),
}


@dataclass(frozen=True, slots=True)
class PreflightProfile:
    """Which evidence a profile demands; anything else is a WARN."""

    name: str
    require_calibration_corpus: bool = True
    require_sensory_health: bool = True
    require_representation_pre: bool = True
    require_representation_post: bool = True
    require_motor_calibration: bool = True
    require_reachability: bool = True
    require_plasticity: bool = True
    require_specificity: bool = False
    require_benchmark: bool = False
    require_control_manifests: bool = False
    require_tiny_real_run: bool = False
    require_protocol: bool = False
    #: ``initial`` still allows a failed real KC-reachability measurement to be
    #: a WARN: at that stage the duration and the sensory route are still being
    #: calibrated and the measurement exists precisely to inform that choice.
    #: A strict Ante-1 preflight must not pass when the predeclared reachability
    #: gates fail for the configured ALPN route and neural duration.
    reachability_failure_is_warning: bool = True
    #: Whether missing/weak calibration-corpus coverage blocks the profile.
    require_corpus_coverage: bool = False
    #: Whether the corpus must have been produced by this experiment's real
    #: environment backend and pinned simulator revision.
    require_corpus_environment_match: bool = True


PREFLIGHT_PROFILES: dict[str, PreflightProfile] = {
    "initial": PreflightProfile(name="initial"),
    "ante1": PreflightProfile(
        name="ante1",
        require_specificity=True,
        require_benchmark=True,
        require_control_manifests=True,
        require_tiny_real_run=True,
        require_protocol=True,
        reachability_failure_is_warning=False,
        require_corpus_coverage=True,
    ),
}


@dataclass(frozen=True, slots=True)
class PreflightThresholds:
    """Every value is an engineering gate, not a biological fact."""

    minimum_alpn_used_fraction: float = 0.20
    #: Structural assignment counts are informational. Set a number only to
    #: deliberately re-enable a structural gate; readiness comes from the
    #: state-conditioned sensory health report.
    maximum_alpn_assignment_p99: float | None = None
    maximum_weight_bound_fraction: float = 0.10
    minimum_motor_pool_width: int = 2
    require_representation_pass: bool = True
    require_plasticity_pass: bool = True
    require_benchmark_report: bool = False
    require_control_manifest_check: bool = False


def run_preflight(
    config: PlasticExperimentConfig,
    *,
    profile: str = "initial",
    sensory_health_report: Mapping[str, Any] | None = None,
    representation_pre_report: Mapping[str, Any] | None = None,
    representation_post_report: Mapping[str, Any] | None = None,
    motor_calibration_report: Mapping[str, Any] | None = None,
    reachability_report: Mapping[str, Any] | None = None,
    plasticity_report: Mapping[str, Any] | None = None,
    specificity_report: Mapping[str, Any] | None = None,
    benchmark_report: Mapping[str, Any] | None = None,
    control_manifests: Sequence[Mapping[str, Any]] = (),
    protocol_payload: Mapping[str, Any] | None = None,
    thresholds: PreflightThresholds | None = None,
) -> dict[str, Any]:
    if profile not in PREFLIGHT_PROFILES:
        raise ValueError(f"unknown preflight profile: {profile}")
    selected = PREFLIGHT_PROFILES[profile]
    limits = thresholds or PreflightThresholds()
    if limits.require_benchmark_report:
        selected = replace(selected, require_benchmark=True)
    if limits.require_control_manifest_check:
        selected = replace(selected, require_control_manifests=True)
    gates: list[dict[str, Any]] = []

    def gate(name: str, status: str, reason: str, evidence: object | None = None) -> None:
        gates.append({"name": name, "status": status, "reason": reason, "evidence": evidence})

    expected, artifact, motor, sensory = _expected_identity(config, gate)
    verify = _identity_verifier(expected, gate)

    # ---- external data and populations -----------------------------------
    if artifact is not None:
        quality = {
            key: artifact.manifest.get(key)
            for key in (
                "n_unresolved_nt_neurons",
                "n_synapses_unresolved_sign",
                "fraction_raw_synapses_dropped",
                "n_missing_coordinates",
            )
        }
        missing_quality = [key for key, value in quality.items() if value is None]
        gate(
            "flywire_data_quality",
            "WARN" if missing_quality else "PASS",
            "artifact predates data-quality counters"
            if missing_quality
            else "unresolved-sign and coordinate effects are explicit",
            quality,
        )
        gate(
            "weak_edge_analysis",
            "PASS",
            "thresholds 1, 2, 5 and 10 reported",
            weak_edge_diagnostics(artifact),
        )

    # ---- frozen calibration corpus ---------------------------------------
    _corpus_gate(config, gate, selected, expected)

    # ---- fixed sensory interface -----------------------------------------
    _sensory_structure_gate(sensory, gate, limits)
    verify(
        "sensory_state_health",
        sensory_health_report,
        "sensory_health",
        # The synthetic development circuit has no annotated ALPN route, so a
        # missing state-conditioned sensory report is a WARN there.
        required=selected.require_sensory_health and config.fly.backend == "flywire",
    )

    # ---- neural representation -------------------------------------------
    verify(
        "representation_pre",
        representation_pre_report,
        "representation_pre",
        required=selected.require_representation_pre,
        extra=_representation_stage_check("pre"),
    )
    verify(
        "kc_subtype_reachability",
        reachability_report,
        "kc_reachability",
        required=selected.require_reachability,
        # Strict Ante-1 must not pass on a failed reachability measurement; the
        # thresholds themselves are never loosened to make it pass.
        treat_fail_as="WARN" if selected.reachability_failure_is_warning else "FAIL",
    )

    # ---- fixed motor interface -------------------------------------------
    _motor_gates(config, motor, artifact, gate, limits, expected)
    verify(
        "motor_calibration",
        motor_calibration_report,
        "motor_calibration",
        required=selected.require_motor_calibration,
    )
    verify(
        "representation_post",
        representation_post_report,
        "representation_post",
        required=selected.require_representation_post,
        extra=_representation_stage_check("post"),
    )

    # ---- plasticity -------------------------------------------------------
    _eligibility_gates(config, gate)
    verify(
        "plasticity_calibration",
        plasticity_report,
        "plasticity_calibration",
        required=selected.require_plasticity,
        extra=_bound_occupancy_check(limits),
    )
    verify(
        "chosen_action_specificity",
        specificity_report,
        "chosen_action_specificity",
        required=selected.require_specificity,
        # Action specificity is a measurement, not a pass/fail criterion: the
        # gate checks that provenance-matched evidence exists and was read.
        status_key=None,
        extra=_specificity_summary(),
    )

    reward = asdict(config.reinforcement)
    finite_reward = all(
        not isinstance(value, float) or np.isfinite(value) for value in reward.values()
    )
    gate(
        "reward_mapping",
        "PASS" if finite_reward else "FAIL",
        f"fixed synthetic reinforcement condition {config.reinforcement.condition!r} "
        "is predeclared, finite and versioned",
        reward,
    )

    # ---- checkpoint, device, runtime --------------------------------------
    _checkpoint_gates(gate)
    _device_gate(config, gate)

    # ---- measured evidence -------------------------------------------------
    verify(
        "throughput_and_memory_benchmark",
        benchmark_report,
        "benchmark",
        required=selected.require_benchmark,
        extra=_benchmark_fields_check(),
        status_key=None,
    )
    _control_gates(gate, control_manifests, selected)
    _protocol_gate(config, gate, protocol_payload, selected)

    gate("environment_backend", "PASS" if config.environment.backend == "mock" or importlib.util.find_spec("balatro_sim") is not None else "FAIL", f"backend={config.environment.backend}")
    gate("recording_and_replay", "PASS" if importlib.util.find_spec("pyarrow") is not None else "FAIL", "PyArrow supports streaming decision row groups")
    gate("visualization_tooling", "PASS" if shutil.which("ffmpeg") else "WARN", "ffmpeg found" if shutil.which("ffmpeg") else "ffmpeg not found; scientific runs can proceed but final rendering cannot")
    gate(
        "external_trainable_policy",
        "PASS",
        "primary architecture declares zero external trainable parameters and "
        "fixed encoder/motor",
        {"external_trainable_parameter_count": 0},
    )

    status = preflight_status(gates)
    return {
        "version": PREFLIGHT_VERSION,
        "profile": selected.name,
        "status": status,
        "profile_requirements": asdict(selected),
        "thresholds": asdict(limits),
        "expected_identity": expected.to_dict(),
        "evidence_identity": expected.to_dict(),
        "gates": gates,
        "failed": [item["name"] for item in gates if item["status"] == "FAIL"],
        "warnings": [item["name"] for item in gates if item["status"] == "WARN"],
    }


def preflight_status(gates: Sequence[Mapping[str, Any]]) -> str:
    if any(item.get("status") == "FAIL" for item in gates):
        return "FAIL"
    if any(item.get("status") == "WARN" for item in gates):
        return "WARN"
    return "PASS"


# --------------------------------------------------------------------------
# expected experiment identity
# --------------------------------------------------------------------------


def _expected_identity(
    config: PlasticExperimentConfig, gate: Callable[..., None]
) -> tuple[EvidenceIdentity, FlyWireArtifact | None, MotorMapping | None, SensoryMapping | None]:
    artifact: FlyWireArtifact | None = None
    sensory: SensoryMapping | None = None
    motor: MotorMapping | None = None
    candidate_hash: str | None = None
    topology_hash: str | None = None
    artifact_hash: str | None = None
    population_hash: str | None = None
    if config.fly.backend == "flywire":
        try:
            artifact = FlyWireArtifact.load(resolve_path(config, config.fly.artifact_path))
            artifact.validate()
        except Exception as error:  # report exact external readiness failure
            gate("artifact_and_populations", "FAIL", str(error))
        else:
            artifact_hash = str(artifact.manifest["artifact_sha256"])
            population_hash = artifact.population_hash
            gate(
                "artifact_and_populations",
                "PASS",
                "v783 artifact validates, including exact 5,177 KC census",
                {"artifact_sha256": artifact_hash, "population_sha256": population_hash},
            )
            shuffled_post = None
            if config.fly.topology != "real":
                _, shuffled_post, _, _ = artifact.edge_arrays(
                    shuffle_seed=config.fly.shuffle_seed,
                    preserve_populations=True,
                    shuffle_scope=(
                        "kc_mbon"
                        if config.fly.topology == "kc_mbon_shuffled"
                        else "whole_brain"
                    ),
                )
            scope = (
                "kc_mbon" if config.fly.topology == "kc_mbon_shuffled" else "whole_brain"
            )
            topology_hash = PlasticEdgeTopology.from_artifact(
                artifact,
                post_indices=shuffled_post,
                version=(
                    f"flywire-v783-{scope}-shuffle-v1-seed-{config.fly.shuffle_seed}"
                    if shuffled_post is not None
                    else "flywire-v783-kc-mbon-neuron-pairs-v1"
                ),
                minimum_synapse_count=config.fly.minimum_synapse_count,
            ).sha256
            sensory = SensoryMapping.from_artifact(
                artifact,
                mapping_seed=config.fly.sensory_mapping_seed,
                population_width=config.fly.sensory_population_width,
                max_rate_hz=config.fly.max_rate_hz,
            )
            candidate_hash = canonical_motor_candidates(
                artifact, mode=config.fly.mode
            ).sha256
    else:
        gate(
            "artifact_and_populations",
            "WARN",
            "synthetic backend cannot validate real FlyWire readiness",
        )
        # The development circuit still declares a checkable identity, so local
        # evidence cannot silently belong to a different synthetic circuit.
        from flylatro.fly.synthetic_plastic import SyntheticPlasticFlyProcessor

        artifact_hash = SYNTHETIC_ARTIFACT_IDENTITY
        population_hash = SYNTHETIC_ARTIFACT_IDENTITY
        topology_hash = SyntheticPlasticFlyProcessor(
            synthetic_circuit_spec(config), mode=config.fly.mode
        ).topology.sha256
    if config.fly.motor_mapping_path:
        path = resolve_path(config, config.fly.motor_mapping_path)
        if path.exists():
            try:
                motor = MotorMapping.load(path)
            except Exception as error:
                gate("canonical_motor_mapping", "FAIL", str(error))
    from flylatro.fly.plastic_features import channel_manifest

    dynamics = fly_dynamics_payload(config)
    git = git_identity()
    identity = EvidenceIdentity(
        report_kind="expected_experiment_identity",
        report_version=PREFLIGHT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        config_sha256=config.sha256,
        artifact_sha256=artifact_hash,
        population_sha256=population_hash,
        plastic_topology_sha256=topology_hash,
        minimum_synapse_count=config.fly.minimum_synapse_count,
        sensory_feature_contract_sha256=str(channel_manifest()["sha256"]),
        sensory_mapping_sha256=(
            sensory.sha256
            if sensory is not None
            else (None if config.fly.backend == "flywire" else SYNTHETIC_SENSORY_IDENTITY)
        ),
        calibration_corpus_sha256=_safe_corpus_hash(config),
        motor_candidate_set_sha256=candidate_hash,
        motor_mapping_sha256=None if motor is None else motor.structure_sha256,
        reinforcement_mapping_sha256=config.reinforcement.sha256,
        plasticity_rule_sha256=config.plasticity.sha256,
        fly_dynamics_sha256=payload_sha256(dynamics),
        # The declared dynamics payload is authoritative: a backend that fixes
        # its own decision window must be compared against that window, not
        # against a configuration field it ignores.
        decision_duration_ms=float(dynamics["decision_duration_ms"]),
        output_mode=config.fly.mode,
        topology_condition=config.fly.topology,
        fly_backend=(
            "flywire-shiu-torch-plastic-kc-mbon-v1"
            if config.fly.backend == "flywire"
            else "synthetic-plastic-mushroom-body-v1"
        ),
        device=config.fly.device,
    )
    return identity, artifact, motor, sensory


def _safe_corpus_hash(config: PlasticExperimentConfig) -> str | None:
    try:
        return calibration_corpus_hash(config)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------
# report verification
# --------------------------------------------------------------------------


def _identity_verifier(
    expected: EvidenceIdentity, gate: Callable[..., None]
) -> Callable[..., None]:
    def verify(
        name: str,
        report: Mapping[str, Any] | None,
        kind: str,
        *,
        required: bool,
        extra: Callable[[Mapping[str, Any]], tuple[bool, str]] | None = None,
        treat_fail_as: str = "FAIL",
        status_key: str | None = "gates",
    ) -> None:
        if report is None:
            gate(
                name,
                "FAIL" if required else "WARN",
                f"no {name} report supplied",
            )
            return
        identity = EvidenceIdentity.from_mapping(report.get("evidence_identity"))
        if identity is None:
            gate(
                name,
                "FAIL",
                f"{name} report carries no evidence identity and cannot be "
                "matched to this experiment; regenerate it",
            )
            return
        # A matching component hash does not prove the file is the report this
        # gate asked for: verify the declared kind and version first.
        policy = REPORT_KIND_POLICY[kind]
        if identity.report_kind != policy.kind:
            gate(
                name,
                "FAIL",
                f"{name} expects a {policy.kind!r} report but the supplied "
                f"artifact declares report_kind={identity.report_kind!r}",
                {
                    "expected_report_kind": policy.kind,
                    "report_kind": identity.report_kind,
                    "report_version": identity.report_version,
                },
            )
            return
        if identity.report_version not in policy.compatible_versions:
            gate(
                name,
                "FAIL",
                f"{name} report version {identity.report_version!r} is not "
                f"compatible with this build; regenerate it "
                f"(accepted: {list(policy.compatible_versions)})",
                {
                    "expected_report_kind": policy.kind,
                    "report_version": identity.report_version,
                    "compatible_versions": list(policy.compatible_versions),
                },
            )
            return
        fields = REPORT_IDENTITY_FIELDS[kind]
        mismatches = identity.mismatches(expected, required_fields=fields)
        if mismatches:
            gate(
                name,
                "FAIL",
                f"{name} report belongs to a different configuration: "
                + describe_mismatches(mismatches),
                {"mismatches": mismatches, "report_kind": identity.report_kind},
            )
            return
        status = "PASS"
        reason = f"{name} evidence matches this experiment"
        if status_key is not None:
            reported = str(report.get(status_key, {}).get("status", "FAIL"))
            status = reported if reported in {"PASS", "WARN", "FAIL"} else "FAIL"
            reason = f"{name} provenance verified; report gate={reported}"
        if extra is not None:
            ok, note = extra(report)
            if not ok:
                status = "FAIL"
                reason = f"{name} provenance verified but {note}"
            elif note:
                reason = f"{reason}; {note}"
        if status == "FAIL" and treat_fail_as == "WARN":
            status = "WARN"
        gate(
            name,
            status,
            reason,
            {
                "report_kind": identity.report_kind,
                "report_version": identity.report_version,
                "generated_at": identity.generated_at,
                "git_commit": identity.git_commit,
                "gates": report.get("gates"),
            },
        )

    return verify


def _representation_stage_check(stage: str) -> Callable[[Mapping[str, Any]], tuple[bool, str]]:
    def check(report: Mapping[str, Any]) -> tuple[bool, str]:
        if str(report.get("stage")) != stage:
            return False, f"it is a {report.get('stage')!r} stage report, not {stage!r}"
        if stage == "post" and not report.get("final_motor_interface_evaluated"):
            return False, "it does not evaluate the final motor interface"
        if stage == "pre" and report.get("final_motor_interface_evaluated"):
            return False, "a pre-motor report must not claim a motor evaluation"
        population = report.get("motor_activity_population")
        if stage == "post" and population not in {"mbon", "descending"}:
            return False, "it does not name the measured motor population"
        return True, f"motor_activity_population={population}"

    return check


def _bound_occupancy_check(
    limits: PreflightThresholds,
) -> Callable[[Mapping[str, Any]], tuple[bool, str]]:
    def check(report: Mapping[str, Any]) -> tuple[bool, str]:
        diagnostics = report.get("diagnostics", {})
        lower = float(diagnostics.get("lower_bound_fraction", 0.0))
        upper = float(diagnostics.get("upper_bound_fraction", 0.0))
        total = lower + upper
        if total > limits.maximum_weight_bound_fraction:
            return False, (
                f"weight-bound occupancy {total:.4f} exceeds "
                f"{limits.maximum_weight_bound_fraction}"
            )
        return True, f"weight-bound occupancy {total:.4f}"

    return check


def _specificity_summary() -> Callable[[Mapping[str, Any]], tuple[bool, str]]:
    def check(report: Mapping[str, Any]) -> tuple[bool, str]:
        chosen = report.get("chosen_motor_update_fraction")
        competing = report.get("competing_motor_update_fraction")
        non_motor = report.get("non_motor_update_fraction")
        if chosen is None:
            return False, "the report has no chosen-action attribution"
        return True, (
            f"chosen={chosen:.4f} competing={competing:.4f} "
            f"non_motor={non_motor:.4f} over {report.get('decisions')} decisions"
        )

    return check


def _benchmark_fields_check() -> Callable[[Mapping[str, Any]], tuple[bool, str]]:
    required = {"decisions_per_second", "peak_memory_bytes", "batch_size", "execution_mode"}

    def check(report: Mapping[str, Any]) -> tuple[bool, str]:
        missing = sorted(required - report.keys())
        if missing:
            return False, f"measured fields are missing: {missing}"
        components = report.get("component_seconds")
        note = "component timings recorded" if components else "no component timings"
        return True, f"{report['decisions_per_second']:.3f} decisions/s; {note}"

    return check


# --------------------------------------------------------------------------
# individual gates
# --------------------------------------------------------------------------


def _corpus_gate(
    config: PlasticExperimentConfig,
    gate: Callable[..., None],
    profile: PreflightProfile,
    expected: EvidenceIdentity,
) -> None:
    """Identity, provenance and coverage of the frozen calibration corpus.

    Three different questions are gated separately:

    1. does a reward-free corpus exist and does the configuration bind to it;
    2. was it produced by *this* experiment's environment — the real Balatro
       adapter at the pinned simulator revision for a real run, never a
       mock-generated corpus that merely happens to be the configured file;
    3. does it cover the observable phases and motor contexts that later
       reward-free calibration is about to be measured in.
    """

    if not config.calibration.corpus_path:
        gate(
            "calibration_corpus_identity",
            "FAIL" if profile.require_calibration_corpus else "WARN",
            "no frozen reward-free calibration corpus is configured",
        )
        gate(
            "calibration_corpus_provenance",
            "FAIL" if profile.require_calibration_corpus else "WARN",
            "no calibration corpus to verify against this environment",
        )
        gate(
            "calibration_corpus_coverage",
            "FAIL" if profile.require_corpus_coverage else "WARN",
            "no calibration corpus to measure coverage on",
        )
        return
    path = resolve_path(config, config.calibration.corpus_path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        for name in (
            "calibration_corpus_identity",
            "calibration_corpus_provenance",
            "calibration_corpus_coverage",
        ):
            gate(
                name,
                "FAIL",
                f"configured calibration corpus manifest is missing: {manifest_path}",
            )
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The manifest alone is not trusted: the corpus arrays themselves are
    # loaded and re-hashed, so an edited manifest cannot authorize a run.
    corpus: CalibrationCorpus | None = None
    load_error: str | None = None
    try:
        corpus = CalibrationCorpus.load(path)
    except Exception as error:  # exact external readiness failure
        load_error = str(error)
    coverage = (
        corpus.coverage() if corpus is not None else manifest.get("coverage", {})
    )
    reward_free = manifest.get("contains_reward_or_strategy_labels") is False
    version_ok = str(manifest.get("version")) in SUPPORTED_CALIBRATION_CORPUS_VERSIONS
    hash_ok = (
        expected.calibration_corpus_sha256 is None
        or manifest.get("sha256") == expected.calibration_corpus_sha256
    )
    identity_checks = {
        "corpus_loads_and_rehashes": load_error is None,
        "declares_reward_free": reward_free,
        "supported_corpus_version": version_ok,
        "matches_configured_sha256": hash_ok,
    }
    gate(
        "calibration_corpus_identity",
        "PASS" if all(identity_checks.values()) else "FAIL",
        "frozen reward-free calibration corpus is versioned, hashed and loads"
        if all(identity_checks.values())
        else f"calibration corpus identity failed: {load_error or identity_checks}",
        {
            "sha256": manifest.get("sha256"),
            "expected_sha256": expected.calibration_corpus_sha256,
            "version": manifest.get("version"),
            "supported_versions": list(SUPPORTED_CALIBRATION_CORPUS_VERSIONS),
            "states": manifest.get("state_count"),
            "generation_method": manifest.get("generation_method"),
            "snapshot_identity_policy": manifest.get("snapshot_identity_policy"),
            "stores_environment_snapshots": manifest.get(
                "stores_environment_snapshots"
            ),
            "records_run_provenance": manifest.get("records_run_provenance"),
            "checks": identity_checks,
            "load_error": load_error,
        },
    )
    _corpus_provenance_gate(config, gate, profile, manifest, corpus)
    _corpus_coverage_gate(config, gate, profile, coverage)


def _corpus_provenance_gate(
    config: PlasticExperimentConfig,
    gate: Callable[..., None],
    profile: PreflightProfile,
    manifest: Mapping[str, Any],
    corpus: CalibrationCorpus | None,
) -> None:
    """Bind the corpus to the environment that is actually configured.

    A mock-generated corpus must never authorize a real experiment merely
    because the configuration points at its SHA-256.
    """

    backend, simulator = expected_corpus_environment(config)
    declared_backend = str(
        (corpus.environment_backend if corpus is not None else manifest.get("environment_backend"))
        or ""
    )
    declared_simulator = str(
        (corpus.simulator_version if corpus is not None else manifest.get("simulator_version"))
        or ""
    )
    checks = {
        "environment_backend_matches": declared_backend == backend,
        "simulator_version_matches": declared_simulator == simulator,
    }
    real_experiment = config.environment.backend == "balatro_sim"
    status = "PASS" if all(checks.values()) else "FAIL"
    if status == "FAIL" and not profile.require_corpus_environment_match:
        status = "WARN"
    gate(
        "calibration_corpus_provenance",
        status,
        (
            "calibration corpus was generated by this experiment's environment "
            f"({backend} @ {simulator})"
            if status == "PASS"
            else (
                "calibration corpus was generated by "
                f"{declared_backend or 'an unknown adapter'} @ "
                f"{declared_simulator or 'an unknown simulator'}, but this "
                f"configuration requires {backend} @ {simulator}"
            )
        ),
        {
            "required_environment_backend": backend,
            "required_simulator_version": simulator,
            "corpus_environment_backend": declared_backend,
            "corpus_simulator_version": declared_simulator,
            "real_experiment_configuration": real_experiment,
            "checks": checks,
        },
    )


def _corpus_coverage_gate(
    config: PlasticExperimentConfig,
    gate: Callable[..., None],
    profile: PreflightProfile,
    coverage: Mapping[str, Any],
) -> None:
    """Require the corpus to actually reach the states calibration needs."""

    try:
        policy = coverage_policy(config)
    except Exception as error:
        gate(
            "calibration_corpus_coverage",
            "FAIL",
            f"configured coverage policy is invalid: {error}",
        )
        return
    result = evaluate_coverage(coverage, policy)
    if result["satisfied"]:
        status = "PASS"
        reason = (
            f"calibration corpus satisfies coverage policy {policy.version} "
            "for every required phase and motor context"
        )
    else:
        status = "FAIL" if profile.require_corpus_coverage else "WARN"
        reason = (
            "calibration corpus coverage is insufficient: "
            + "; ".join(result["deficiency_summary"])
        )
    gate("calibration_corpus_coverage", status, reason, result)


def _sensory_structure_gate(
    sensory: SensoryMapping | None,
    gate: Callable[..., None],
    limits: PreflightThresholds,
) -> None:
    from flylatro.fly.plastic_features import channel_manifest

    contract = channel_manifest()
    gate(
        "sensory_feature_contract",
        "PASS",
        f"field-aware observable contract {contract['version']} with "
        f"{contract['channel_count']} channels and no strategy features",
        {"sha256": contract["sha256"], "version": contract["version"]},
    )
    if sensory is None:
        gate(
            "sensory_mapping_identity",
            "WARN",
            "requires the real v783 artifact",
        )
        return
    audit = sensory.collision_audit()
    checks = {
        "used_fraction": audit["fraction_alpns_used"] >= limits.minimum_alpn_used_fraction,
        "collision_policy": audit["policy"] == "clipped_sum_preserve_sparse_indicators",
    }
    if limits.maximum_alpn_assignment_p99 is not None:
        checks["structural_p99_assignment"] = (
            audit["assignments_per_alpn"]["p99"] <= limits.maximum_alpn_assignment_p99
        )
    gate(
        "sensory_mapping_identity",
        "PASS" if all(checks.values()) else "FAIL",
        "deterministic field-aware ALPN mapping; structural assignment counts "
        "are informational and readiness comes from sensory_state_health",
        {"sha256": sensory.sha256, "checks": checks, "structural_audit": audit},
    )


def _motor_gates(
    config: PlasticExperimentConfig,
    motor: MotorMapping | None,
    artifact: FlyWireArtifact | None,
    gate: Callable[..., None],
    limits: PreflightThresholds,
    expected: EvidenceIdentity,
) -> None:
    if motor is None:
        gate(
            "final_motor_artifact_identity",
            "FAIL" if config.fly.backend == "flywire" else "WARN",
            "no persisted reward-free motor mapping artifact is configured",
        )
        gate("motor_candidate_validity", "WARN", "no motor artifact to check")
        gate("motor_normalization_validity", "WARN", "no motor artifact to check")
        gate("reserved_action_slots", "WARN", "no motor artifact to check")
        return
    calibrated = (
        motor.calibration_sha256 is not None
        and motor.selection_method.startswith("reward-free")
    )
    widths = [len(pool) for pool in motor.pool_indices]
    identity_ok = motor.mode == config.fly.mode
    gate(
        "final_motor_artifact_identity",
        "PASS" if calibrated and identity_ok and min(widths) >= limits.minimum_motor_pool_width else "FAIL",
        "persisted reward-free motor artifact checked",
        {
            "structure_sha256": motor.structure_sha256,
            "artifact_sha256": motor.sha256,
            "mode": motor.mode,
            "selection_method": motor.selection_method,
            "reward_free_calibration": calibrated,
            "minimum_pool_width": min(widths),
            "pool_count": motor.routing.pool_count,
        },
    )
    candidates_match = (
        expected.motor_candidate_set_sha256 is None
        or motor.candidate_set_sha256 == expected.motor_candidate_set_sha256
    )
    roots_match = True
    reachable = None
    if artifact is not None:
        candidate_set = canonical_motor_candidates(artifact, mode=config.fly.mode)
        roots_match = bool(
            np.array_equal(motor.output_root_ids, candidate_set.root_ids)
        )
        reachable = len(candidate_set)
    gate(
        "motor_candidate_validity",
        "PASS" if candidates_match and roots_match else "FAIL",
        "motor candidates come from the canonical plastic-reachable universe of "
        "the real unshuffled anatomy and are identical across topology controls",
        {
            "artifact_candidate_set_sha256": motor.candidate_set_sha256,
            "expected_candidate_set_sha256": expected.motor_candidate_set_sha256,
            "canonical_candidates": reachable,
            "root_ids_match": roots_match,
        },
    )
    normalization = motor.normalization
    normalization_ok = (
        normalization is not None
        and normalization.version != "identity-uncalibrated-v1"
        and bool(np.isfinite(normalization.baseline).all())
        and bool(np.isfinite(normalization.scale).all())
        and float(normalization.scale.min()) >= normalization.minimum_scale_hz - 1e-12
    )
    gate(
        "motor_normalization_validity",
        "PASS" if normalization_ok else "FAIL",
        "fixed reward-free baseline/scale normalization is present, finite and "
        "floored",
        None
        if normalization is None
        else {
            "version": normalization.version,
            "sha256": normalization.sha256,
            "minimum_scale_hz": normalization.minimum_scale_hz,
            "scale_min": float(normalization.scale.min()),
            "scale_max": float(normalization.scale.max()),
            "floored_pools": int(
                sum(1 for entry in normalization.statistics if entry.get("scale_floored"))
            ),
        },
    )
    reserved_unrouted = all(
        motor.routing.head_routes["action_type"][value] == -1
        for value in RESERVED_ACTION_TYPES
    )
    gate(
        "reserved_action_slots",
        "PASS" if reserved_unrouted else "FAIL",
        "permanently reserved contract slots own no neural population and can "
        "never be selected",
        {
            "full_action_space_width": len(motor.routing.head_routes["action_type"]),
            "scientifically_represented": list(SUPPORTED_ACTION_TYPES),
            "reserved_unrepresented": list(RESERVED_ACTION_TYPES),
        },
    )


def _eligibility_gates(config: PlasticExperimentConfig, gate: Callable[..., None]) -> None:
    rule = config.plasticity
    fixed_reference = (
        rule.version == "three-factor-global-v1"
        and rule.kc_reference_hz > 0
        and rule.mbon_reference_hz > 0
        and rule.max_eligibility > 0
    )
    gate(
        "fixed_reference_eligibility",
        "PASS" if fixed_reference else "FAIL",
        "eligibility uses configured fixed Hz references and a bounded trace, "
        "never per-decision maximum normalization",
        {
            "version": rule.version,
            "kc_reference_hz": rule.kc_reference_hz,
            "mbon_reference_hz": rule.mbon_reference_hz,
            "max_eligibility": rule.max_eligibility,
            "sha256": rule.sha256,
        },
    )
    state = PlasticEdgeState.initialize(3, learners=2)
    state.eligibility[:] = 1
    state.dopamine[:] = 1
    state.efficacy[0, 0] = 1.1
    state.reset_fast_traces(learners=(0,), eligibility=True)
    reset_ok = bool(
        np.all(state.eligibility[0] == 0)
        and np.all(state.dopamine[0] == 0)
        and np.all(state.eligibility[1] == 1)
        and state.efficacy[0, 0] == np.float32(1.1)
    )
    gate(
        "episode_boundary_reset",
        "PASS" if reset_ok else "FAIL",
        "learner-local fast traces reset while efficacy persists",
    )


def _checkpoint_gates(gate: Callable[..., None]) -> None:
    state = PlasticEdgeState.initialize(3, learners=2)
    restored = PlasticEdgeState.from_state_dict(state.state_dict())
    gate(
        "checkpoint_resume",
        "PASS" if restored.weight_sha256 == state.weight_sha256 else "FAIL",
        "plastic state round-trip hash checked",
    )


def _device_gate(config: PlasticExperimentConfig, gate: Callable[..., None]) -> None:
    device = config.fly.device
    expected_state = "TorchPlasticEdgeState" if config.fly.backend == "flywire" else "PlasticEdgeState"
    if not device.startswith("cuda"):
        gate(
            "device_state_consistency",
            "PASS",
            f"device={device}; plastic state class={expected_state}",
            {"device": device, "plastic_state": expected_state},
        )
        return
    try:
        import torch
    except ImportError:
        gate("device_state_consistency", "FAIL", "CUDA requested but Torch is missing")
        return
    available = bool(torch.cuda.is_available())
    gate(
        "device_state_consistency",
        "PASS" if available else "FAIL",
        "CUDA is available and plastic state is device-resident"
        if available
        else "CUDA was requested but is not available",
        {
            "device": device,
            "plastic_state": expected_state,
            "cuda_available": available,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if available
            else [],
        },
    )


def _control_gates(
    gate: Callable[..., None],
    control_manifests: Sequence[Mapping[str, Any]],
    profile: PreflightProfile,
) -> None:
    if len(control_manifests) < 2:
        gate(
            "matched_control_readiness",
            "FAIL" if profile.require_control_manifests else "WARN",
            "at least two control manifests were not supplied",
        )
        gate(
            "tiny_real_plastic_run",
            "FAIL" if profile.require_tiny_real_run else "WARN",
            "no tiny real plastic run manifest was supplied",
        )
        return
    report = validate_control_group(control_manifests)
    gate(
        "matched_control_readiness",
        report["gates"]["status"],
        "matched-control semantics validated for exact action-matched and "
        "behaviourally independent topology controls",
        {"failed": report["gates"]["failed"], "arms": report["arms"]},
    )
    conditions = [arm["condition"] for arm in report["arms"]]
    has_real = "plastic_real" in conditions
    has_control = any(
        condition in {"no_plasticity", "kc_mbon_shuffled", "whole_brain_shuffled", "shuffled_reward"}
        for condition in conditions
    )
    ok = has_real and has_control
    gate(
        "tiny_real_plastic_run",
        "PASS" if ok else ("FAIL" if profile.require_tiny_real_run else "WARN"),
        "a real plastic run and at least one matched control are present",
        {"conditions": conditions},
    )


def _protocol_gate(
    config: PlasticExperimentConfig,
    gate: Callable[..., None],
    payload: Mapping[str, Any] | None,
    profile: PreflightProfile,
) -> None:
    if payload is None:
        gate(
            "protocol_identity",
            "FAIL" if profile.require_protocol else "WARN",
            "no replicate protocol manifest was supplied",
        )
        return
    try:
        protocol = ExperimentProtocol.from_payload(payload)
    except Exception as error:
        gate("protocol_identity", "FAIL", f"protocol manifest is invalid: {error}")
        return
    matches = payload.get("sha256") == protocol.sha256
    bound = config.protocol.protocol_sha256
    binding_ok = not bound or bound == protocol.sha256
    gate(
        "protocol_identity",
        "PASS" if matches and binding_ok else "FAIL",
        "replicate protocol hashes and matches this configuration's binding",
        {
            "protocol_sha256": protocol.sha256,
            "declared_sha256": payload.get("sha256"),
            "configuration_binding": bound or None,
            "replicates": protocol.replicate_count,
            "conditions": list(protocol.conditions),
            "motor_mapping_id": protocol.motor_mapping_id,
        },
    )


def _resolve(config: PlasticExperimentConfig, value: str) -> Path:
    return resolve_path(config, value)


def load_json(path: Path | None) -> Mapping[str, Any] | None:
    return None if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
