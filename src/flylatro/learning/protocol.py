"""First-class replicate and matched-control experiment protocols.

Every seed stored here controls a named stochastic mechanism.  A seed with no
effect is worse than useless: it implies an independent source of randomness
that does not exist.  `SEED_EFFECTS` is the audit record, and the protocol
version changes whenever that table changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from flylatro.seeds import derive_seed


EXPERIMENT_PROTOCOL_VERSION = "plastic-replicate-protocol-v2"

CONTROL_CONDITIONS = (
    "plastic_real",
    "no_plasticity",
    "kc_mbon_shuffled",
    "whole_brain_shuffled",
    "shuffled_reward",
)

#: Exactly what each stored seed controls. Audited by tests.
#:
#: `plasticity_seed` was removed in v2: initial efficacy is a fixed constant and
#: the update rule is deterministic given neural activity and reinforcement, so
#: no independent plasticity randomness exists to seed.
SEED_EFFECTS: dict[str, str] = {
    "training_seed_offset": (
        "offset into the reserved training seed range; selects the exact "
        "environment reset seed sequence (SeedPlan.training)"
    ),
    "environment_seed_stream": (
        "descriptive label for the seed stream; the operative value is "
        "training_seed_offset"
    ),
    "sensory_mapping_seed": (
        "fly.sensory_mapping_seed; draws the fixed Balatro-feature-to-ALPN "
        "random projection. Changing it is a mapping replicate"
    ),
    "fly_seed": (
        "training.base_fly_seed; seeds the per-decision Poisson input streams "
        "of each independent fly"
    ),
    "motor_seed": (
        "motor.exploration_seed; seeds the softmax/epsilon exploration branch "
        "via derive_seed(motor_seed, learner_id, decision_id). It never changes "
        "the calibrated motor mapping itself"
    ),
    "topology_seed": (
        "fly.shuffle_seed; only active for kc_mbon_shuffled and "
        "whole_brain_shuffled, where it draws the label permutation"
    ),
    "reward_seed": (
        "seed of the offline deterministic synthetic-reinforcement permutation; "
        "only active for the shuffled_reward condition"
    ),
}

CONDITION_SEED_USAGE: dict[str, tuple[str, ...]] = {
    "plastic_real": ("training_seed_offset", "sensory_mapping_seed", "fly_seed", "motor_seed"),
    "no_plasticity": ("training_seed_offset", "sensory_mapping_seed", "fly_seed", "motor_seed"),
    "kc_mbon_shuffled": (
        "training_seed_offset", "sensory_mapping_seed", "fly_seed", "motor_seed", "topology_seed",
    ),
    "whole_brain_shuffled": (
        "training_seed_offset", "sensory_mapping_seed", "fly_seed", "motor_seed", "topology_seed",
    ),
    "shuffled_reward": (
        "training_seed_offset", "sensory_mapping_seed", "fly_seed", "motor_seed", "reward_seed",
    ),
}


@dataclass(frozen=True, slots=True)
class ReplicateSpec:
    replicate_id: str
    training_seed_offset: int
    environment_seed_stream: str
    sensory_mapping_seed: int
    fly_seed: int
    motor_seed: int
    topology_seed: int
    reward_seed: int


@dataclass(frozen=True, slots=True)
class ConditionArm:
    arm_id: str
    replicate_id: str
    condition: str
    training_seed_offset: int
    environment_seed_stream: str
    sensory_mapping_seed: int
    fly_seed: int
    motor_seed: int
    topology_seed: int
    reward_seed: int
    motor_mapping_id: str
    exposure_budget_decisions: int
    curriculum_ladder: tuple[int, ...]
    reinforcement_condition: str
    active_seeds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    version: str
    name: str
    base_seed: int
    replicate_count: int
    conditions: tuple[str, ...]
    exposure_budget_decisions: int
    curriculum_ladder: tuple[int, ...]
    motor_mapping_id: str
    reinforcement_condition: str
    replicates: tuple[ReplicateSpec, ...]
    arms: tuple[ConditionArm, ...]

    @classmethod
    def create(
        cls,
        *,
        name: str,
        base_seed: int,
        replicate_count: int,
        conditions: Sequence[str],
        exposure_budget_decisions: int,
        curriculum_ladder: Sequence[int],
        motor_mapping_id: str,
        reinforcement_condition: str = "primary-progress",
    ) -> "ExperimentProtocol":
        unknown = set(conditions) - set(CONTROL_CONDITIONS)
        if unknown or replicate_count < 1 or exposure_budget_decisions < 1:
            raise ValueError(
                f"invalid replicate protocol; unknown conditions={sorted(unknown)}"
            )
        if not motor_mapping_id or "REPLACE_WITH" in motor_mapping_id:
            raise ValueError("protocol requires the measured motor mapping SHA-256")
        replicates = tuple(
            ReplicateSpec(
                replicate_id=f"replicate-{index:03d}",
                training_seed_offset=index,
                environment_seed_stream=f"training-replicate-{index:03d}",
                sensory_mapping_seed=derive_seed("sensory-mapping", base_seed, index)
                % (2**31),
                fly_seed=derive_seed("fly", base_seed, index) % (2**31),
                motor_seed=derive_seed("motor", base_seed, index) % (2**31),
                topology_seed=derive_seed("topology", base_seed, index) % (2**31),
                reward_seed=derive_seed("reward", base_seed, index) % (2**31),
            )
            for index in range(replicate_count)
        )
        ladder = tuple(int(value) for value in curriculum_ladder)
        arms = tuple(
            ConditionArm(
                arm_id=f"{replicate.replicate_id}:{condition}",
                replicate_id=replicate.replicate_id,
                condition=condition,
                training_seed_offset=replicate.training_seed_offset,
                environment_seed_stream=replicate.environment_seed_stream,
                sensory_mapping_seed=replicate.sensory_mapping_seed,
                fly_seed=replicate.fly_seed,
                motor_seed=replicate.motor_seed,
                topology_seed=replicate.topology_seed,
                reward_seed=replicate.reward_seed,
                motor_mapping_id=motor_mapping_id,
                exposure_budget_decisions=exposure_budget_decisions,
                curriculum_ladder=ladder,
                reinforcement_condition=reinforcement_condition,
                active_seeds=CONDITION_SEED_USAGE[condition],
            )
            for replicate in replicates
            for condition in conditions
        )
        return cls(
            version=EXPERIMENT_PROTOCOL_VERSION,
            name=name,
            base_seed=base_seed,
            replicate_count=replicate_count,
            conditions=tuple(conditions),
            exposure_budget_decisions=exposure_budget_decisions,
            curriculum_ladder=ladder,
            motor_mapping_id=motor_mapping_id,
            reinforcement_condition=reinforcement_condition,
            replicates=replicates,
            arms=arms,
        )

    def arm(self, arm_id: str) -> ConditionArm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise KeyError(f"protocol has no arm {arm_id!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "seed_effects": dict(SEED_EFFECTS),
            "condition_seed_usage": {
                name: list(values) for name, values in CONDITION_SEED_USAGE.items()
            },
            "sha256": self.sha256,
        }

    @property
    def sha256(self) -> str:
        payload = {**asdict(self), "seed_effects": dict(SEED_EFFECTS)}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "ExperimentProtocol":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = cls.from_payload(payload)
        if payload.get("sha256") != protocol.sha256:
            raise ValueError("experiment protocol hash mismatch")
        return protocol

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExperimentProtocol":
        return cls(
            version=str(payload["version"]),
            name=str(payload["name"]),
            base_seed=int(payload["base_seed"]),
            replicate_count=int(payload["replicate_count"]),
            conditions=tuple(str(value) for value in payload["conditions"]),
            exposure_budget_decisions=int(payload["exposure_budget_decisions"]),
            curriculum_ladder=tuple(int(value) for value in payload["curriculum_ladder"]),
            motor_mapping_id=str(payload["motor_mapping_id"]),
            reinforcement_condition=str(
                payload.get("reinforcement_condition", "primary-progress")
            ),
            replicates=tuple(
                ReplicateSpec(
                    replicate_id=str(item["replicate_id"]),
                    training_seed_offset=int(item["training_seed_offset"]),
                    environment_seed_stream=str(item["environment_seed_stream"]),
                    sensory_mapping_seed=int(item["sensory_mapping_seed"]),
                    fly_seed=int(item["fly_seed"]),
                    motor_seed=int(item["motor_seed"]),
                    topology_seed=int(item["topology_seed"]),
                    reward_seed=int(item["reward_seed"]),
                )
                for item in payload["replicates"]
            ),
            arms=tuple(
                ConditionArm(
                    arm_id=str(item["arm_id"]),
                    replicate_id=str(item["replicate_id"]),
                    condition=str(item["condition"]),
                    training_seed_offset=int(item["training_seed_offset"]),
                    environment_seed_stream=str(item["environment_seed_stream"]),
                    sensory_mapping_seed=int(item["sensory_mapping_seed"]),
                    fly_seed=int(item["fly_seed"]),
                    motor_seed=int(item["motor_seed"]),
                    topology_seed=int(item["topology_seed"]),
                    reward_seed=int(item["reward_seed"]),
                    motor_mapping_id=str(item["motor_mapping_id"]),
                    exposure_budget_decisions=int(item["exposure_budget_decisions"]),
                    curriculum_ladder=tuple(
                        int(value) for value in item["curriculum_ladder"]
                    ),
                    reinforcement_condition=str(
                        item.get("reinforcement_condition", "primary-progress")
                    ),
                    active_seeds=tuple(str(value) for value in item["active_seeds"]),
                )
                for item in payload["arms"]
            ),
        )


def assert_configuration_matches_arm(
    payload: Mapping[str, Any], config: Any
) -> dict[str, Any]:
    """Refuse a run whose configuration drifted from its protocol arm."""

    protocol = ExperimentProtocol.from_payload(payload)
    arm = protocol.arm(config.protocol.arm_id)
    expected: dict[str, tuple[Any, Any]] = {
        "condition": (config.training.condition, arm.condition),
        "replicate_id": (config.protocol.replicate_id, arm.replicate_id),
        "training_seed_offset": (
            config.training.training_seed_offset,
            arm.training_seed_offset,
        ),
        "sensory_mapping_seed": (
            config.fly.sensory_mapping_seed,
            arm.sensory_mapping_seed,
        ),
        "base_fly_seed": (config.training.base_fly_seed, arm.fly_seed),
        "motor_exploration_seed": (config.motor.exploration_seed, arm.motor_seed),
        "exposure_budget_decisions": (
            config.training.max_environment_decisions,
            arm.exposure_budget_decisions,
        ),
        "curriculum_ladder": (tuple(config.curriculum.ladder), arm.curriculum_ladder),
        "reinforcement_condition": (
            config.reinforcement.condition,
            arm.reinforcement_condition,
        ),
    }
    if arm.condition in {"kc_mbon_shuffled", "whole_brain_shuffled"}:
        expected["topology_shuffle_seed"] = (config.fly.shuffle_seed, arm.topology_seed)
    differences = {
        name: {"config": actual, "protocol": wanted}
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if differences:
        raise ValueError(
            f"configuration does not match protocol arm {arm.arm_id}: "
            + json.dumps(differences, sort_keys=True)
        )
    return {
        "protocol_sha256": protocol.sha256,
        "protocol_name": protocol.name,
        "replicate_id": arm.replicate_id,
        "arm_id": arm.arm_id,
        "condition": arm.condition,
        "motor_mapping_id": arm.motor_mapping_id,
        "active_seeds": list(arm.active_seeds),
    }


CONTROL_VALIDATION_VERSION = "matched-control-validation-v2"

#: Matched in every condition: the same exposure, seeds and fixed interfaces.
COMMON_MATCHED_FIELDS = (
    "curriculum_ladder",
    "exposure_budget_decisions",
    "training_seeds",
    "sensory_mapping_sha256",
    "motor_mapping_sha256",
    "canonical_motor_root_ids_sha256",
    "reinforcement_mapping_sha256",
)

#: Conditions whose behaviour is, by construction, the real fly's own
#: trajectory replayed: they must reproduce the exact executed actions and
#: before/after state hashes.
EXACT_ACTION_MATCHED_CONDITIONS = frozenset({"no_plasticity", "shuffled_reward"})
ACTION_MATCHED_CONDITIONS = EXACT_ACTION_MATCHED_CONDITIONS  # compatibility alias
ACTION_MATCHED_FIELDS = (
    "action_schedule_sha256",
    "state_hash_schedule_sha256",
)

#: Conditions whose topology changes behaviour, so their action/state
#: trajectory *cannot* match the real fly's.  They are matched on the
#: experimental protocol instead.
INDEPENDENT_TOPOLOGY_CONDITIONS = frozenset(
    {"kc_mbon_shuffled", "whole_brain_shuffled"}
)
TOPOLOGY_MATCHED_FIELDS = (
    "artifact_sha256",
    "population_sha256",
    "fly_dynamics_sha256",
    "plasticity_rule_sha256",
    "reinforcement_mode",
    "canonical_motor_candidate_set_sha256",
)


def control_class(condition: str | None) -> str:
    if condition in EXACT_ACTION_MATCHED_CONDITIONS:
        return "exact_action_and_state_matched"
    if condition in INDEPENDENT_TOPOLOGY_CONDITIONS:
        return "behaviourally_independent_topology_control"
    return "reference"


def _value(manifest: Mapping[str, Any], field: str) -> Any:
    if field in manifest:
        return manifest[field]
    components = manifest.get("components", {})
    if isinstance(components, Mapping) and field in components:
        return components[field]
    return None


def validate_control_group(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate a matched-control group and explain each arm's matching rule."""

    if len(manifests) < 2:
        raise ValueError("matched-control validation needs at least two manifests")
    reference = manifests[0]
    differences: dict[str, list[Any]] = {}
    for field in COMMON_MATCHED_FIELDS:
        values = [_value(manifest, field) for manifest in manifests]
        if any(item is None for item in values) or any(
            item != values[0] for item in values[1:]
        ):
            differences[field] = values
    arms: list[dict[str, Any]] = []
    reference_actions = {field: _value(reference, field) for field in ACTION_MATCHED_FIELDS}
    reference_topology = _value(reference, "plastic_topology_sha256")
    for manifest in manifests:
        condition = _value(manifest, "condition")
        kind = control_class(condition)
        arm: dict[str, Any] = {
            "condition": condition,
            "control_class": kind,
            "topology_condition": _value(manifest, "topology_condition"),
            "plastic_topology_sha256": _value(manifest, "plastic_topology_sha256"),
            "matched_fields": list(COMMON_MATCHED_FIELDS),
            "differences": {},
        }
        if kind == "exact_action_and_state_matched":
            arm["matched_fields"] += list(ACTION_MATCHED_FIELDS)
            for field, wanted in reference_actions.items():
                actual = _value(manifest, field)
                if wanted is None or actual is None or actual != wanted:
                    arm["differences"][field] = {"arm": actual, "reference": wanted}
                    differences.setdefault(field, []).append(actual)
        elif kind == "behaviourally_independent_topology_control":
            arm["matched_fields"] += list(TOPOLOGY_MATCHED_FIELDS)
            arm["action_schedule_matching"] = (
                "not applicable: a shuffled topology produces its own behaviour, "
                "so the real fly's future actions and states cannot be replayed"
            )
            for field in TOPOLOGY_MATCHED_FIELDS:
                actual = _value(manifest, field)
                wanted = _value(reference, field)
                if wanted is None or actual is None or actual != wanted:
                    arm["differences"][field] = {"arm": actual, "reference": wanted}
                    differences.setdefault(field, []).append(actual)
            if (
                reference_topology is not None
                and arm["plastic_topology_sha256"] == reference_topology
            ):
                arm["differences"]["plastic_topology_sha256"] = {
                    "arm": arm["plastic_topology_sha256"],
                    "reference": reference_topology,
                    "reason": "a topology control must not reuse the real topology",
                }
                differences.setdefault("topology_control_is_identical", []).append(
                    arm["plastic_topology_sha256"]
                )
        arms.append(arm)
    checks = {
        "common_fields_match": not any(
            field in differences for field in COMMON_MATCHED_FIELDS
        ),
        "exact_action_controls_match": not any(
            field in differences for field in ACTION_MATCHED_FIELDS
        ),
        "topology_controls_match_protocol": not any(
            field in differences for field in TOPOLOGY_MATCHED_FIELDS
        ),
        "topology_controls_actually_shuffled": "topology_control_is_identical"
        not in differences,
    }
    return {
        "version": CONTROL_VALIDATION_VERSION,
        "manifests": len(manifests),
        "reference_condition": _value(reference, "condition"),
        "arms": arms,
        "differences": differences,
        "control_semantics": {
            "exact_action_and_state_matched": sorted(EXACT_ACTION_MATCHED_CONDITIONS),
            "behaviourally_independent_topology_control": sorted(
                INDEPENDENT_TOPOLOGY_CONDITIONS
            ),
        },
        "gates": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "failed": [name for name, passed in checks.items() if not passed],
        },
    }


def assert_matched_control_manifests(manifests: Sequence[Mapping[str, Any]]) -> None:
    report = validate_control_group(manifests)
    if report["gates"]["status"] != "PASS":
        raise ValueError(
            "matched-control manifests differ: "
            + json.dumps(report["differences"], sort_keys=True, default=str)
        )
