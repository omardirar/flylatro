"""First-class replicate and matched-control experiment protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from flylatro.seeds import derive_seed


CONTROL_CONDITIONS = (
    "plastic_real",
    "no_plasticity",
    "kc_mbon_shuffled",
    "whole_brain_shuffled",
    "shuffled_reward",
)


@dataclass(frozen=True, slots=True)
class ReplicateSpec:
    replicate_id: str
    experiment_seed: int
    training_seed_offset: int
    plasticity_seed: int
    sensory_mapping_seed: int
    environment_seed_stream: str
    fly_seed: int
    motor_seed: int
    topology_seed: int
    reward_seed: int


@dataclass(frozen=True, slots=True)
class ConditionArm:
    replicate_id: str
    condition: str
    experiment_seed: int
    plasticity_seed: int
    sensory_mapping_seed: int
    motor_mapping_id: str
    environment_seed_stream: str
    fly_seed: int
    motor_seed: int
    topology_seed: int
    reward_seed: int
    exposure_budget_decisions: int
    curriculum_ladder: tuple[int, ...]


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
    ) -> "ExperimentProtocol":
        unknown = set(conditions) - set(CONTROL_CONDITIONS)
        if unknown or replicate_count < 1 or exposure_budget_decisions < 1:
            raise ValueError(f"invalid replicate protocol; unknown conditions={sorted(unknown)}")
        replicates = tuple(
            ReplicateSpec(
                replicate_id=f"replicate-{index:03d}",
                experiment_seed=derive_seed("replicate", base_seed, index),
                training_seed_offset=index,
                plasticity_seed=derive_seed("plasticity", base_seed, index),
                sensory_mapping_seed=derive_seed("sensory-mapping", base_seed, index),
                environment_seed_stream=f"training-replicate-{index:03d}",
                fly_seed=derive_seed("fly", base_seed, index),
                motor_seed=derive_seed("motor", base_seed, index),
                topology_seed=derive_seed("topology", base_seed, index),
                reward_seed=derive_seed("reward", base_seed, index),
            )
            for index in range(replicate_count)
        )
        ladder = tuple(int(value) for value in curriculum_ladder)
        arms = tuple(
            ConditionArm(
                replicate_id=replicate.replicate_id,
                condition=condition,
                experiment_seed=replicate.experiment_seed,
                plasticity_seed=replicate.plasticity_seed,
                sensory_mapping_seed=replicate.sensory_mapping_seed,
                motor_mapping_id=motor_mapping_id,
                environment_seed_stream=replicate.environment_seed_stream,
                fly_seed=replicate.fly_seed,
                motor_seed=replicate.motor_seed,
                topology_seed=replicate.topology_seed,
                reward_seed=replicate.reward_seed,
                exposure_budget_decisions=exposure_budget_decisions,
                curriculum_ladder=ladder,
            )
            for replicate in replicates
            for condition in conditions
        )
        return cls(
            version="plastic-replicate-protocol-v1",
            name=name,
            base_seed=base_seed,
            replicate_count=replicate_count,
            conditions=tuple(conditions),
            exposure_budget_decisions=exposure_budget_decisions,
            curriculum_ladder=ladder,
            motor_mapping_id=motor_mapping_id,
            replicates=replicates,
            arms=arms,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def save(self, path: Path) -> Path:
        payload = {**asdict(self), "sha256": self.sha256}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


COMMON_MATCHED_FIELDS = (
    "curriculum_ladder",
    "exposure_budget_decisions",
    "training_seeds",
    "sensory_mapping_sha256",
    "motor_mapping_sha256",
    "canonical_motor_root_ids_sha256",
    "reinforcement_mapping_sha256",
)

ACTION_MATCHED_CONDITIONS = frozenset({"no_plasticity", "shuffled_reward"})
ACTION_MATCHED_FIELDS = (
    "action_schedule_sha256",
    "state_hash_schedule_sha256",
)


def assert_matched_control_manifests(manifests: Sequence[Mapping[str, Any]]) -> None:
    if len(manifests) < 2:
        raise ValueError("matched-control validation needs at least two manifests")
    def value(manifest: Mapping[str, Any], field: str) -> Any:
        if field in manifest:
            return manifest[field]
        components = manifest.get("components", {})
        if isinstance(components, Mapping) and field in components:
            return components[field]
        return None

    reference = manifests[0]
    differences: dict[str, list[object]] = {}
    for field in COMMON_MATCHED_FIELDS:
        values = [value(manifest, field) for manifest in manifests]
        if any(item is None for item in values):
            differences[field] = values
            continue
        if any(value != values[0] for value in values[1:]):
            differences[field] = values
    reference_action_values = {
        field: value(reference, field) for field in ACTION_MATCHED_FIELDS
    }
    for manifest in manifests:
        condition = value(manifest, "condition")
        if condition in ACTION_MATCHED_CONDITIONS:
            for field, reference_value in reference_action_values.items():
                candidate = value(manifest, field)
                if reference_value is None or candidate is None or candidate != reference_value:
                    differences.setdefault(field, []).append(candidate)
        topology = value(manifest, "topology_condition")
        if topology != "real":
            if value(manifest, "artifact_sha256") != value(reference, "artifact_sha256"):
                differences.setdefault("artifact_sha256", []).append(value(manifest, "artifact_sha256"))
            if value(manifest, "population_sha256") != value(reference, "population_sha256"):
                differences.setdefault("population_sha256", []).append(value(manifest, "population_sha256"))
    if differences:
        raise ValueError("matched-control manifests differ: " + json.dumps(differences, sort_keys=True))
