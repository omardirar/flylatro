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


EXPERIMENT_PROTOCOL_VERSION = "plastic-replicate-protocol-v3"

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
#:
#: `sensory_mapping_seed` stopped being a per-replicate seed in v3.  Redrawing
#: the fixed Balatro-to-ALPN projection changes the experiment's *interface*,
#: not its stochastic realization, and it invalidates every downstream
#: calibration measured through the old projection.  It is now a block-level
#: experimental factor (`SensoryMappingVariant`), constant across the ordinary
#: stochastic replicates inside a block.
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
        "random projection. It is a BLOCK-LEVEL experimental factor, fixed "
        "across the ordinary stochastic replicates of one mapping block; "
        "changing it defines a separate sensory-mapping replicate whose motor "
        "calibration and representation evidence must be regenerated"
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
        "only active for the shuffled_reward condition. The materialized plan "
        "generates the schedule with exactly this seed and the run refuses a "
        "schedule shuffled with another one"
    ),
}

#: Seeds that vary between the ordinary stochastic replicates of one mapping
#: block.  Everything else is a fixed interface or an explicit factor.
STOCHASTIC_REPLICATE_SEEDS: tuple[str, ...] = (
    "training_seed_offset",
    "environment_seed_stream",
    "fly_seed",
    "motor_seed",
)

#: Factors that are deliberately *not* ordinary replicate noise.  Changing one
#: is a separate experimental block with its own regenerated calibration.
BLOCK_LEVEL_FACTORS: dict[str, str] = {
    "sensory_mapping_seed": (
        "fixed synthetic Balatro-to-ALPN projection; a change invalidates "
        "sensory health, representation pre/post and motor calibration"
    ),
    "motor_mapping_id": (
        "the calibrated reward-free motor artifact's structure SHA-256; it is "
        "measured through one sensory mapping and cannot be reused across "
        "mapping replicates"
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

#: Which of a condition's active seeds actually vary between its ordinary
#: replicates.  `sensory_mapping_seed` appears in every condition's usage
#: because every arm configures it, but it is constant inside a mapping block.
CONDITION_STOCHASTIC_SEEDS: dict[str, tuple[str, ...]] = {
    condition: tuple(
        name for name in seeds if name in set(STOCHASTIC_REPLICATE_SEEDS)
    )
    for condition, seeds in CONDITION_SEED_USAGE.items()
}


@dataclass(frozen=True, slots=True)
class SensoryMappingVariant:
    """One fixed sensory interface plus the motor artifact calibrated for it.

    A protocol conceptually contains ``mapping-000 -> replicates 0..n`` and,
    for an explicit sensitivity block, ``mapping-001 -> replicates 0..n``.  The
    ordinary replicate count never introduces a new mapping.
    """

    mapping_id: str
    sensory_mapping_seed: int
    motor_mapping_id: str
    role: str = "primary"

    def __post_init__(self) -> None:
        if not self.mapping_id:
            raise ValueError("a sensory mapping variant needs an identifier")
        if not self.motor_mapping_id or "REPLACE_WITH" in self.motor_mapping_id:
            raise ValueError(
                f"sensory mapping {self.mapping_id!r} requires the measured motor "
                "mapping SHA-256 calibrated through that exact mapping"
            )
        if self.role not in {"primary", "mapping_sensitivity"}:
            raise ValueError("mapping variant role must be primary or mapping_sensitivity")


@dataclass(frozen=True, slots=True)
class ReplicateSpec:
    replicate_id: str
    mapping_id: str
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
    mapping_id: str
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
    stochastic_seeds: tuple[str, ...] = ()

    @property
    def reference_arm_id(self) -> str:
        """The paired ``plastic_real`` arm this arm is matched against."""

        return f"{self.replicate_id}:{REFERENCE_CONDITION}"


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
    sensory_mappings: tuple[SensoryMappingVariant, ...] = ()

    @property
    def mapping_replicate_count(self) -> int:
        return len(self.sensory_mappings)

    def mapping(self, mapping_id: str) -> SensoryMappingVariant:
        for variant in self.sensory_mappings:
            if variant.mapping_id == mapping_id:
                return variant
        raise KeyError(f"protocol has no sensory mapping {mapping_id!r}")

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
        sensory_mapping_seed: int = 0,
        mapping_sensitivity_variants: Sequence[Mapping[str, Any]] = (),
        reinforcement_condition: str = "primary-progress",
    ) -> "ExperimentProtocol":
        """Build a protocol.

        ``replicate_count`` is the number of ordinary *stochastic* replicates
        per sensory mapping block.  It never redraws the sensory mapping.  A
        deliberate sensory-mapping sensitivity block is added by supplying
        ``mapping_sensitivity_variants`` — each needs its own
        ``sensory_mapping_seed`` *and* its own motor mapping SHA-256, because
        motor calibration is measured through the sensory mapping.
        """

        unknown = set(conditions) - set(CONTROL_CONDITIONS)
        if unknown or replicate_count < 1 or exposure_budget_decisions < 1:
            raise ValueError(
                f"invalid replicate protocol; unknown conditions={sorted(unknown)}"
            )
        if not motor_mapping_id or "REPLACE_WITH" in motor_mapping_id:
            raise ValueError("protocol requires the measured motor mapping SHA-256")
        mappings = [
            SensoryMappingVariant(
                mapping_id="mapping-000",
                sensory_mapping_seed=int(sensory_mapping_seed),
                motor_mapping_id=motor_mapping_id,
                role="primary",
            )
        ]
        for position, payload in enumerate(mapping_sensitivity_variants, start=1):
            mappings.append(
                SensoryMappingVariant(
                    mapping_id=str(payload.get("mapping_id", f"mapping-{position:03d}")),
                    sensory_mapping_seed=int(payload["sensory_mapping_seed"]),
                    motor_mapping_id=str(payload["motor_mapping_id"]),
                    role="mapping_sensitivity",
                )
            )
        _validate_mapping_variants(mappings)
        ladder = tuple(int(value) for value in curriculum_ladder)
        replicates: list[ReplicateSpec] = []
        for block, variant in enumerate(mappings):
            for index in range(replicate_count):
                replicates.append(
                    ReplicateSpec(
                        replicate_id=f"{variant.mapping_id}-replicate-{index:03d}",
                        mapping_id=variant.mapping_id,
                        training_seed_offset=block * replicate_count + index,
                        environment_seed_stream=(
                            f"training-{variant.mapping_id}-replicate-{index:03d}"
                        ),
                        # Constant across the ordinary replicates of this block.
                        sensory_mapping_seed=variant.sensory_mapping_seed,
                        fly_seed=derive_seed("fly", base_seed, block, index) % (2**31),
                        motor_seed=derive_seed("motor", base_seed, block, index) % (2**31),
                        topology_seed=derive_seed("topology", base_seed, block, index)
                        % (2**31),
                        reward_seed=derive_seed("reward", base_seed, block, index)
                        % (2**31),
                    )
                )
        arms = tuple(
            ConditionArm(
                arm_id=f"{replicate.replicate_id}:{condition}",
                replicate_id=replicate.replicate_id,
                mapping_id=replicate.mapping_id,
                condition=condition,
                training_seed_offset=replicate.training_seed_offset,
                environment_seed_stream=replicate.environment_seed_stream,
                sensory_mapping_seed=replicate.sensory_mapping_seed,
                fly_seed=replicate.fly_seed,
                motor_seed=replicate.motor_seed,
                topology_seed=replicate.topology_seed,
                reward_seed=replicate.reward_seed,
                motor_mapping_id=next(
                    variant.motor_mapping_id
                    for variant in mappings
                    if variant.mapping_id == replicate.mapping_id
                ),
                exposure_budget_decisions=exposure_budget_decisions,
                curriculum_ladder=ladder,
                reinforcement_condition=reinforcement_condition,
                active_seeds=CONDITION_SEED_USAGE[condition],
                stochastic_seeds=CONDITION_STOCHASTIC_SEEDS[condition],
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
            replicates=tuple(replicates),
            arms=arms,
            sensory_mappings=tuple(mappings),
        )

    def arm(self, arm_id: str) -> ConditionArm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise KeyError(f"protocol has no arm {arm_id!r}")

    def replicate(self, replicate_id: str) -> ReplicateSpec:
        for replicate in self.replicates:
            if replicate.replicate_id == replicate_id:
                return replicate
        raise KeyError(f"protocol has no replicate {replicate_id!r}")

    def seed_audit(self) -> dict[str, Any]:
        """Which quantities vary with what. Asserted by tests."""

        varying = {
            name: len({getattr(item, name) for item in self.replicates})
            for name in (
                "training_seed_offset",
                "sensory_mapping_seed",
                "fly_seed",
                "motor_seed",
                "topology_seed",
                "reward_seed",
            )
        }
        return {
            "distinct_values_across_replicates": varying,
            "stochastic_replicate_seeds": list(STOCHASTIC_REPLICATE_SEEDS),
            "block_level_factors": dict(BLOCK_LEVEL_FACTORS),
            "sensory_mapping_seeds_per_block": {
                variant.mapping_id: variant.sensory_mapping_seed
                for variant in self.sensory_mappings
            },
            "ordinary_replicates_share_one_sensory_mapping": all(
                len(
                    {
                        item.sensory_mapping_seed
                        for item in self.replicates
                        if item.mapping_id == variant.mapping_id
                    }
                )
                == 1
                for variant in self.sensory_mappings
            ),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "seed_effects": dict(SEED_EFFECTS),
            "condition_seed_usage": {
                name: list(values) for name, values in CONDITION_SEED_USAGE.items()
            },
            "condition_stochastic_seeds": {
                name: list(values) for name, values in CONDITION_STOCHASTIC_SEEDS.items()
            },
            "block_level_factors": dict(BLOCK_LEVEL_FACTORS),
            "seed_audit": self.seed_audit(),
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
        version = str(payload["version"])
        if version != EXPERIMENT_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported experiment protocol version {version!r}; this build "
                f"requires {EXPERIMENT_PROTOCOL_VERSION!r}. Regenerate the protocol: "
                "older protocols redrew the sensory mapping per replicate"
            )
        return cls(
            version=version,
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
            sensory_mappings=tuple(
                SensoryMappingVariant(
                    mapping_id=str(item["mapping_id"]),
                    sensory_mapping_seed=int(item["sensory_mapping_seed"]),
                    motor_mapping_id=str(item["motor_mapping_id"]),
                    role=str(item.get("role", "primary")),
                )
                for item in payload.get("sensory_mappings", ())
            ),
            replicates=tuple(
                ReplicateSpec(
                    replicate_id=str(item["replicate_id"]),
                    mapping_id=str(item["mapping_id"]),
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
                    mapping_id=str(item["mapping_id"]),
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
                    stochastic_seeds=tuple(
                        str(value) for value in item.get("stochastic_seeds", ())
                    ),
                )
                for item in payload["arms"]
            ),
        )


def _validate_mapping_variants(variants: Sequence[SensoryMappingVariant]) -> None:
    ids = [variant.mapping_id for variant in variants]
    if len(set(ids)) != len(ids):
        raise ValueError("sensory mapping variants need distinct mapping_id values")
    seeds = [variant.sensory_mapping_seed for variant in variants]
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "sensory mapping variants need distinct sensory_mapping_seed values; "
            "an identical mapping is not a mapping replicate"
        )
    motors = [variant.motor_mapping_id for variant in variants]
    if len(set(motors)) != len(motors):
        raise ValueError(
            "each sensory mapping replicate needs its own motor mapping SHA-256: "
            "motor calibration is measured through the sensory mapping and must "
            "be regenerated when the mapping changes"
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
        "mapping_id": arm.mapping_id,
        "arm_id": arm.arm_id,
        "reference_arm_id": arm.reference_arm_id,
        "condition": arm.condition,
        "motor_mapping_id": arm.motor_mapping_id,
        "reward_seed": arm.reward_seed,
        "active_seeds": list(arm.active_seeds),
        "stochastic_seeds": list(arm.stochastic_seeds),
    }


CONTROL_VALIDATION_VERSION = "matched-control-validation-v3"

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


#: The one condition a matched comparison group measures everything against.
REFERENCE_CONDITION = "plastic_real"


def find_reference(manifests: Sequence[Mapping[str, Any]]) -> tuple[int, Mapping[str, Any]]:
    """Locate the single ``plastic_real`` reference, whatever the CLI order.

    Trusting ``manifests[0]`` silently made the report depend on the order the
    ``--manifest`` flags happened to be typed in: a group listed control-first
    would have been validated against a control.
    """

    positions = [
        index
        for index, manifest in enumerate(manifests)
        if _value(manifest, "condition") == REFERENCE_CONDITION
    ]
    if not positions:
        observed = sorted(
            str(_value(manifest, "condition")) for manifest in manifests
        )
        raise ValueError(
            "matched-control validation needs exactly one plastic_real reference "
            f"manifest; supplied conditions: {observed}"
        )
    if len(positions) > 1:
        raise ValueError(
            "matched-control validation needs exactly one plastic_real reference "
            f"manifest; {len(positions)} were supplied"
        )
    return positions[0], manifests[positions[0]]


def validate_control_group(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate a matched-control group and explain each arm's matching rule.

    The result is invariant to the order the manifests are supplied in: the
    reference is located by condition, and the arms are reported in a stable
    condition/arm order rather than in argument order.
    """

    if len(manifests) < 2:
        raise ValueError("matched-control validation needs at least two manifests")
    reference_index, reference = find_reference(manifests)
    order = sorted(
        range(len(manifests)),
        key=lambda index: (
            0 if index == reference_index else 1,
            str(_value(manifests[index], "condition")),
            str(_value(manifests[index], "protocol_arm_id") or ""),
            str(_value(manifests[index], "plastic_topology_sha256") or ""),
        ),
    )
    ordered = [manifests[index] for index in order]
    differences: dict[str, list[Any]] = {}
    for field in COMMON_MATCHED_FIELDS:
        values = [_value(manifest, field) for manifest in ordered]
        if any(item is None for item in values) or any(
            item != values[0] for item in values[1:]
        ):
            differences[field] = values
    arms: list[dict[str, Any]] = []
    reference_actions = {field: _value(reference, field) for field in ACTION_MATCHED_FIELDS}
    reference_topology = _value(reference, "plastic_topology_sha256")
    for manifest in ordered:
        condition = _value(manifest, "condition")
        kind = control_class(condition)
        arm: dict[str, Any] = {
            "condition": condition,
            "control_class": kind,
            "is_reference": manifest is reference,
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
        "single_plastic_real_reference": True,
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
        "reference_selection": (
            "located by condition == plastic_real; independent of manifest order"
        ),
        "reference_arm_id": _value(reference, "protocol_arm_id"),
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
