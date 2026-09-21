"""Configurable engineering coverage requirements for the calibration corpus.

The frozen corpus decides what every later reward-free calibration can see.  A
corpus that is almost entirely ``PLAYING`` states cannot authorize a motor
mapping that also interprets shop, pack, joker and consumable slots: the
contextual pools would be selected and normalized on evidence that never
exercises three of the four contexts they are read in.

Every number here is an **engineering gate on the corpus**, not a biological
fact and not a claim about Balatro.  The policy is versioned, hashable and
loadable from JSON so an experiment can record exactly which requirements it
was authorized against.

Two strengths exist, matching the preflight profiles:

``initial``
    weak or missing phase/context coverage is a WARN.  The first real corpus is
    exploratory: the navigation budget needed to reach ``SHOP`` and ``PACK`` on
    the real simulator is itself something the experimenter measures.

``strict``
    the motor-relevant contexts must have real evidence before a motor mapping
    or an Ante-1 run may be authorized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from flylatro.analysis.corpus import PHASE_NAMES
from flylatro.interface.motor_contexts import MOTOR_CONTEXT_NAMES


CALIBRATION_COVERAGE_POLICY_VERSION = "calibration-corpus-coverage-policy-v1"

#: Observable phases a real Balatro corpus is expected to reach.  Slot 5 is
#: unnamed in the pinned contract and is deliberately not required.
REQUIRED_PHASES: tuple[str, ...] = (
    "BLIND_SELECT",
    "PLAYING",
    "ROUND_EVAL",
    "SHOP",
    "PACK",
)

#: Contexts whose pools the fixed motor interface actually interprets.  Every
#: one of them must have usable evidence before motor calibration is trusted.
MOTOR_RELEVANT_CONTEXTS: tuple[str, ...] = MOTOR_CONTEXT_NAMES


@dataclass(frozen=True, slots=True)
class CoveragePolicy:
    """Versioned, configurable corpus-coverage requirements."""

    version: str = CALIBRATION_COVERAGE_POLICY_VERSION
    minimum_states: int = 32
    minimum_unique_state_hashes: int = 24
    minimum_distinct_source_runs: int = 2
    required_phases: tuple[str, ...] = REQUIRED_PHASES
    minimum_states_per_required_phase: int = 4
    minimum_unique_states_per_required_phase: int = 3
    #: Contexts the motor interface reads; each needs enough relevant states and
    #: enough states where more than one option actually competes.
    required_motor_contexts: tuple[str, ...] = MOTOR_RELEVANT_CONTEXTS
    minimum_states_per_motor_context: int = 4
    minimum_competing_states_per_motor_context: int = 2
    minimum_active_options_per_motor_context: int = 2
    require_run_provenance: bool = True

    def __post_init__(self) -> None:
        unknown_phases = set(self.required_phases) - set(PHASE_NAMES)
        if unknown_phases:
            raise ValueError(f"unknown required phases: {sorted(unknown_phases)}")
        unknown_contexts = set(self.required_motor_contexts) - set(MOTOR_CONTEXT_NAMES)
        if unknown_contexts:
            raise ValueError(f"unknown motor contexts: {sorted(unknown_contexts)}")
        if min(
            self.minimum_states,
            self.minimum_unique_state_hashes,
            self.minimum_states_per_required_phase,
            self.minimum_unique_states_per_required_phase,
            self.minimum_states_per_motor_context,
            self.minimum_competing_states_per_motor_context,
            self.minimum_active_options_per_motor_context,
            self.minimum_distinct_source_runs,
        ) < 0:
            raise ValueError("coverage requirements cannot be negative")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_phases"] = list(self.required_phases)
        payload["required_motor_contexts"] = list(self.required_motor_contexts)
        return payload

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**self.to_payload(), "sha256": self.sha256}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CoveragePolicy":
        known = {field for field in cls.__dataclass_fields__}
        unknown = set(payload) - known - {"sha256"}
        if unknown:
            raise ValueError(f"unknown coverage policy fields: {sorted(unknown)}")
        values = {key: value for key, value in payload.items() if key in known}
        for key in ("required_phases", "required_motor_contexts"):
            if key in values:
                values[key] = tuple(str(item) for item in values[key])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "CoveragePolicy":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    def relaxed_for_initial_calibration(self) -> "CoveragePolicy":
        """The same requirements; the caller decides WARN versus FAIL."""

        return replace(self)


@dataclass(frozen=True, slots=True)
class CoverageDeficiency:
    """One exact, greppable shortfall."""

    scope: str
    name: str
    metric: str
    observed: int
    required: int

    def describe(self) -> str:
        return f"{self.name}: {self.observed} {self.metric}, required >= {self.required}"


def evaluate_coverage(
    coverage: Mapping[str, Any], policy: CoveragePolicy | None = None
) -> dict[str, Any]:
    """Compare a corpus ``coverage()`` payload against the policy.

    Works from a loaded corpus or from its manifest, because both carry the
    same coverage payload.
    """

    limits = policy or CoveragePolicy()
    deficiencies: list[CoverageDeficiency] = []
    states = int(coverage.get("states", 0))
    unique = int(coverage.get("unique_state_hashes", 0))
    if states < limits.minimum_states:
        deficiencies.append(
            CoverageDeficiency("corpus", "states", "states", states, limits.minimum_states)
        )
    if unique < limits.minimum_unique_state_hashes:
        deficiencies.append(
            CoverageDeficiency(
                "corpus",
                "unique_state_hashes",
                "unique states",
                unique,
                limits.minimum_unique_state_hashes,
            )
        )
    distinct_runs = int(coverage.get("distinct_source_run_seeds", 0))
    if distinct_runs < limits.minimum_distinct_source_runs:
        deficiencies.append(
            CoverageDeficiency(
                "corpus",
                "distinct_source_run_seeds",
                "distinct runs",
                distinct_runs,
                limits.minimum_distinct_source_runs,
            )
        )
    phase_detail = coverage.get("phase_detail") or {}
    observed_phases = coverage.get("phases_observed") or {}
    for phase in limits.required_phases:
        detail = phase_detail.get(phase) or {}
        count = int(detail.get("states", observed_phases.get(phase, 0)))
        unique_count = int(detail.get("unique_states", 0))
        if count < limits.minimum_states_per_required_phase:
            deficiencies.append(
                CoverageDeficiency(
                    "phase",
                    phase,
                    "states",
                    count,
                    limits.minimum_states_per_required_phase,
                )
            )
        elif unique_count < limits.minimum_unique_states_per_required_phase:
            deficiencies.append(
                CoverageDeficiency(
                    "phase",
                    phase,
                    "unique states",
                    unique_count,
                    limits.minimum_unique_states_per_required_phase,
                )
            )
    contexts = coverage.get("motor_context_coverage") or {}
    for name in limits.required_motor_contexts:
        detail = contexts.get(name) or {}
        relevant = int(detail.get("relevant_states", 0))
        competing = int(detail.get("states_with_competing_options", 0))
        active = int(detail.get("active_option_count", 0))
        if relevant < limits.minimum_states_per_motor_context:
            deficiencies.append(
                CoverageDeficiency(
                    "motor_context",
                    name,
                    "relevant states",
                    relevant,
                    limits.minimum_states_per_motor_context,
                )
            )
            continue
        if competing < limits.minimum_competing_states_per_motor_context:
            deficiencies.append(
                CoverageDeficiency(
                    "motor_context",
                    name,
                    "states with competing legal options",
                    competing,
                    limits.minimum_competing_states_per_motor_context,
                )
            )
        if active < limits.minimum_active_options_per_motor_context:
            deficiencies.append(
                CoverageDeficiency(
                    "motor_context",
                    name,
                    "active legal options",
                    active,
                    limits.minimum_active_options_per_motor_context,
                )
            )
    provenance_ok = (
        bool(coverage.get("records_run_provenance", False))
        or not limits.require_run_provenance
    )
    if not provenance_ok:
        deficiencies.append(
            CoverageDeficiency("corpus", "records_run_provenance", "states with run identity", 0, states)
        )
    motor_relevant = {
        item.name for item in deficiencies if item.scope == "motor_context"
    }
    return {
        "policy_version": limits.version,
        "policy_sha256": limits.sha256,
        "policy": limits.to_payload(),
        "satisfied": not deficiencies,
        "motor_context_evidence_satisfied": not motor_relevant,
        "deficiencies": [asdict(item) for item in deficiencies],
        "deficiency_summary": [item.describe() for item in deficiencies],
        "phases_missing": list(coverage.get("phases_missing", ())),
        "observed": {
            "states": states,
            "unique_state_hashes": unique,
            "distinct_source_run_seeds": distinct_runs,
            "phase_detail": dict(phase_detail),
            "motor_context_coverage": dict(contexts),
            "records_run_provenance": bool(coverage.get("records_run_provenance", False)),
        },
    }
