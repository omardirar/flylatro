"""Explicit experiment identity attached to every generated evidence artifact.

A report that merely says ``PASS`` proves nothing unless it also proves which
experiment produced it.  Every field here is either a content hash of a fixed
component or an exactly recorded configuration value, so a later gate can
refuse evidence that belongs to a different configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import subprocess
from typing import Any, Mapping


EVIDENCE_IDENTITY_VERSION = "flylatro-evidence-identity-v1"


#: Fields that must agree between a stored report and the current experiment
#: before the report may be used as evidence for that experiment.  Gates select
#: the subset that is relevant to the specific report kind.
COMPARABLE_IDENTITY_FIELDS: tuple[str, ...] = (
    "config_sha256",
    "simulator_version",
    "artifact_sha256",
    "population_sha256",
    "fly_connectivity_sha256",
    "plastic_topology_sha256",
    "minimum_synapse_count",
    "sensory_feature_contract_sha256",
    "sensory_mapping_sha256",
    "calibration_corpus_sha256",
    "motor_candidate_set_sha256",
    "motor_mapping_sha256",
    "reinforcement_mapping_sha256",
    "plasticity_rule_sha256",
    "fly_dynamics_sha256",
    "decision_duration_ms",
    "output_mode",
    "topology_condition",
    "fly_backend",
    "device",
)


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    """Provenance of one evidence artifact.

    Every field is optional because different report kinds bind to different
    components; a field that is ``None`` is explicitly *not claimed* rather than
    silently assumed to match.
    """

    report_kind: str
    report_version: str
    identity_version: str = EVIDENCE_IDENTITY_VERSION
    generated_at: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_sha256: str | None = None
    simulator_version: str | None = None
    artifact_sha256: str | None = None
    population_sha256: str | None = None
    fly_connectivity_sha256: str | None = None
    plastic_topology_sha256: str | None = None
    minimum_synapse_count: int | None = None
    sensory_feature_contract_sha256: str | None = None
    sensory_mapping_sha256: str | None = None
    calibration_corpus_sha256: str | None = None
    motor_candidate_set_sha256: str | None = None
    motor_mapping_sha256: str | None = None
    reinforcement_mapping_sha256: str | None = None
    plasticity_rule_sha256: str | None = None
    fly_dynamics_sha256: str | None = None
    decision_duration_ms: float | None = None
    output_mode: str | None = None
    topology_condition: str | None = None
    fly_backend: str | None = None
    device: str | None = None
    backend_evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return every recorded field, including explicit ``None`` claims."""

        return asdict(self)

    def claimed(self) -> dict[str, Any]:
        """Return only the fields this artifact actually binds itself to."""

        return {key: value for key, value in asdict(self).items() if value is not None}

    def mismatches(
        self, expected: "EvidenceIdentity", *, required_fields: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        """Compare against the current experiment identity.

        A required field that the report does not claim is a mismatch: stale
        evidence must not pass simply because it omitted the binding.
        """

        result: dict[str, dict[str, Any]] = {}
        mine = asdict(self)
        theirs = asdict(expected)
        for field in required_fields:
            if field not in mine:
                raise KeyError(f"unknown evidence identity field: {field}")
            reported = mine[field]
            current = theirs[field]
            if current is None:
                continue
            if reported is None or reported != current:
                result[field] = {"report": reported, "expected": current}
        return result

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "EvidenceIdentity | None":
        if not payload:
            return None
        known = {field.name for field in fields(cls)}
        values = {key: value for key, value in payload.items() if key in known}
        values.setdefault("report_kind", "unknown")
        values.setdefault("report_version", "unknown")
        return cls(**values)

    @classmethod
    def from_components(
        cls,
        components: Mapping[str, Any],
        *,
        report_kind: str,
        report_version: str,
        config_sha256: str | None = None,
        simulator_version: str | None = None,
        calibration_corpus_sha256: str | None = None,
        device: str | None = None,
        backend_evidence: str | None = None,
        include_git: bool = True,
    ) -> "EvidenceIdentity":
        """Build an identity from a ``build_plastic_stack`` component manifest."""

        git = git_identity() if include_git else {"commit": None, "dirty": None}
        dynamics = components.get("fly_dynamics", {})
        duration = None
        if isinstance(dynamics, Mapping):
            raw = dynamics.get("decision_duration_ms")
            duration = float(raw) if isinstance(raw, (int, float)) else None
        sensory = components.get("sensory_mapping", {})
        contract_hash = None
        if isinstance(sensory, Mapping):
            contract = sensory.get("feature_contract")
            if isinstance(contract, Mapping):
                contract_hash = contract.get("sha256")
        return cls(
            report_kind=report_kind,
            report_version=report_version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git["commit"],
            git_dirty=git["dirty"],
            config_sha256=config_sha256,
            simulator_version=simulator_version,
            artifact_sha256=_text(components.get("artifact_sha256")),
            population_sha256=_text(components.get("population_sha256")),
            fly_connectivity_sha256=_text(components.get("fly_connectivity_sha256")),
            plastic_topology_sha256=_text(components.get("plastic_topology_sha256")),
            minimum_synapse_count=_integer(components.get("minimum_synapse_count")),
            sensory_feature_contract_sha256=_text(contract_hash),
            sensory_mapping_sha256=_text(components.get("sensory_mapping_sha256")),
            calibration_corpus_sha256=_text(calibration_corpus_sha256),
            motor_candidate_set_sha256=_text(
                components.get("canonical_motor_candidate_set_sha256")
                or components.get("canonical_motor_root_ids_sha256")
            ),
            motor_mapping_sha256=_text(components.get("motor_mapping_sha256")),
            reinforcement_mapping_sha256=_text(
                components.get("reinforcement_mapping_sha256")
            ),
            plasticity_rule_sha256=_text(components.get("plasticity_rule_sha256")),
            fly_dynamics_sha256=_text(components.get("fly_dynamics_sha256")),
            decision_duration_ms=duration,
            output_mode=_text(components.get("output_mode")),
            topology_condition=_text(components.get("topology_condition")),
            fly_backend=_text(components.get("fly_backend")),
            device=_text(device),
            backend_evidence=_text(backend_evidence),
        )


def git_identity() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ("git", *args), capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def describe_mismatches(mismatches: Mapping[str, Mapping[str, Any]]) -> str:
    """Render a precise, greppable mismatch explanation for a gate reason."""

    parts = [
        f"{field}: report={json.dumps(values.get('report'))} "
        f"expected={json.dumps(values.get('expected'))}"
        for field, values in sorted(mismatches.items())
    ]
    return "; ".join(parts)


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _integer(value: Any) -> int | None:
    return None if value is None else int(value)
