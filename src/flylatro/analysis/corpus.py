"""Frozen, reward-free calibration state corpus.

Calibration must not depend on trajectories produced by whatever bootstrap
decoder happened to exist at the time.  This module builds a deterministic,
versioned corpus of *observable Balatro states and legal masks only*.

The corpus deliberately contains no reward, no best action, no expected value,
no heuristic recommendation and no strategy annotation.  Scripted legal
navigation is used only to reach diverse observable states; the navigation
actions are never stored and must never be treated as targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import (
    BLIND_REQ_OFF,
    BLIND_SCORED_OFF,
    GLOBAL_ANTE_OFF,
    GLOBAL_DISCARDS_LEFT,
    GLOBAL_HANDS_LEFT,
    GLOBAL_MONEY,
    GLOBAL_PHASE_OFF,
    MASK_SPEC,
    MAX_CARD_PICKS,
    N_PHASES,
    OBS_SPEC,
    ActionDict,
    MaskDict,
    ObsDict,
    empty_action_batch,
    validate_batch,
)
from flylatro.interface.motor import SUPPORTED_ACTION_TYPES
from flylatro.seeds import derive_seed


CALIBRATION_CORPUS_VERSION = "reward-free-calibration-state-corpus-v1"
SCRIPTED_NAVIGATION_RULE = "seeded-uniform-legal-navigation-v1"

#: Phase slots 0-4 are corroborated by the pinned adapter's own phase offsets.
#: Slot 5 is present in the contract width but is not named by Flylatro's pinned
#: contract, so it is reported without inventing a meaning.
PHASE_NAMES: tuple[str, ...] = (
    "BLIND_SELECT",
    "PLAYING",
    "ROUND_EVAL",
    "SHOP",
    "PACK",
    "unnamed_phase_5",
)

#: Fields deliberately excluded from the corpus. Asserted by tests.
FORBIDDEN_CORPUS_FIELDS: tuple[str, ...] = (
    "reward",
    "reward_components",
    "best_action",
    "recommended_action",
    "expected_value",
    "strategy",
    "hand_strength",
    "action",
)


def scripted_navigation_actions(masks: MaskDict, rng: np.random.Generator) -> ActionDict:
    """Reach diverse observable states with fixed, seeded legal navigation.

    This is explicitly *not* a policy, a heuristic or a demonstration.  It never
    reads observations, reward, score, or any measure of action quality; it only
    samples uniformly from the environment's own legality mask.
    """

    batch = masks["action_type_mask"].shape[0]
    actions = empty_action_batch(batch)
    supported = np.zeros(masks["action_type_mask"].shape[1], dtype=np.bool_)
    supported[list(SUPPORTED_ACTION_TYPES)] = True
    for row in range(batch):
        legal = np.flatnonzero(np.asarray(masks["action_type_mask"][row]) & supported)
        if not legal.size:
            raise ValueError("no supported legal action type for corpus navigation")
        action_type = int(rng.choice(legal))
        actions["action_type"][row] = action_type
        cards = np.flatnonzero(masks["card_select_mask"][row])
        needs_cards = action_type in (0, 1)
        optional_cards = action_type == 8
        if (needs_cards or optional_cards) and cards.size:
            maximum = min(MAX_CARD_PICKS, int(cards.size))
            low = 1 if needs_cards else 0
            count = int(rng.integers(low, maximum + 1))
            chosen = rng.choice(cards, size=count, replace=False) if count else np.empty(0, np.int64)
            chosen = np.sort(np.asarray(chosen, dtype=np.int64))
            actions["n_cards"][row] = count
            if count:
                actions["cards"][row, :count] = chosen
        elif needs_cards:
            raise ValueError("card-consuming action is legal with no selectable card")
        else:
            actions["n_cards"][row] = 0
        for field, mask_name in (
            ("joker_target", "joker_target_mask"),
            ("consumable_target", "consumable_target_mask"),
            ("shop_target", "shop_target_mask"),
            ("pack_target", "pack_target_mask"),
        ):
            available = np.flatnonzero(masks[mask_name][row])
            if available.size:
                actions[field][row] = int(rng.choice(available))
    return actions


@dataclass(frozen=True, slots=True)
class CalibrationCorpus:
    version: str
    simulator_version: str
    environment_backend: str
    generation_method: str
    navigation_rule: str
    navigation_seed: int
    environment_seeds: tuple[int, ...]
    sample_every: int
    observations: Mapping[str, NDArray[Any]]
    masks: Mapping[str, NDArray[Any]]
    state_hashes: tuple[str, ...]
    phase_labels: tuple[int, ...]
    source_seed: tuple[int, ...]
    source_decision: tuple[int, ...]
    snapshots: tuple[bytes, ...] | None = None

    def __post_init__(self) -> None:
        if set(self.observations) != set(OBS_SPEC) or set(self.masks) != set(MASK_SPEC):
            raise ValueError("calibration corpus must carry the full pinned contract")
        validate_batch(OBS_SPEC, dict(self.observations), len(self), "corpus observations")
        validate_batch(MASK_SPEC, dict(self.masks), len(self), "corpus masks")
        if len(self.state_hashes) != len(self) or len(self.phase_labels) != len(self):
            raise ValueError("corpus labels must cover every state")
        if self.snapshots is not None and len(self.snapshots) != len(self):
            raise ValueError("corpus snapshots must cover every state")

    def __len__(self) -> int:
        return int(next(iter(self.observations.values())).shape[0])

    def row(self, index: int) -> tuple[ObsDict, MaskDict]:
        """One state as a batch of size one, ready for the fixed encoder."""

        return (
            {key: value[index : index + 1].copy() for key, value in self.observations.items()},
            {key: value[index : index + 1].copy() for key, value in self.masks.items()},
        )

    def batch(self, indices: Sequence[int]) -> tuple[ObsDict, MaskDict]:
        selected = np.asarray(indices, dtype=np.int64)
        return (
            {key: value[selected].copy() for key, value in self.observations.items()},
            {key: value[selected].copy() for key, value in self.masks.items()},
        )

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "version": self.version,
                    "simulator_version": self.simulator_version,
                    "environment_backend": self.environment_backend,
                    "generation_method": self.generation_method,
                    "navigation_rule": self.navigation_rule,
                    "navigation_seed": self.navigation_seed,
                    "environment_seeds": list(self.environment_seeds),
                    "sample_every": self.sample_every,
                    "state_hashes": list(self.state_hashes),
                    "phase_labels": list(self.phase_labels),
                    "source_seed": list(self.source_seed),
                    "source_decision": list(self.source_decision),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for key in sorted(self.observations):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(self.observations[key]).tobytes())
        for key in sorted(self.masks):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(self.masks[key]).tobytes())
        return digest.hexdigest()

    def coverage(self) -> dict[str, Any]:
        """Observable-phase and state-dimension coverage, labels only."""

        phases = np.asarray(self.phase_labels, dtype=np.int64)
        global_values = np.asarray(self.observations["global"], dtype=np.float64)
        blind = np.asarray(self.observations["blind"], dtype=np.float64)
        ante_hot = global_values[:, GLOBAL_ANTE_OFF : GLOBAL_ANTE_OFF + 9]
        ante = np.where(ante_hot.any(axis=1), ante_hot.argmax(axis=1) + 1, 0)
        required = np.expm1(blind[:, BLIND_REQ_OFF])
        scored = np.expm1(blind[:, BLIND_SCORED_OFF])
        progress = np.clip(scored / np.maximum(required, 1e-12), 0.0, 1.0)
        jokers = np.count_nonzero(
            np.asarray(self.observations["joker_ids"]) >= 0, axis=1
        )
        shop = np.count_nonzero(np.asarray(self.observations["shop_ids"]) >= 0, axis=1)
        pack = np.count_nonzero(np.asarray(self.masks["pack_target_mask"]), axis=1)
        return {
            "states": len(self),
            "unique_state_hashes": len(set(self.state_hashes)),
            "phases_observed": {
                PHASE_NAMES[index]: int(np.count_nonzero(phases == index))
                for index in range(N_PHASES)
            },
            "phases_missing": [
                PHASE_NAMES[index]
                for index in range(N_PHASES)
                if not np.count_nonzero(phases == index)
            ],
            "unlabelled_phase_states": int(np.count_nonzero(phases < 0)),
            "ante_histogram": {
                str(int(value)): int(np.count_nonzero(ante == value))
                for value in np.unique(ante)
            },
            "hands_remaining": _spread(global_values[:, GLOBAL_HANDS_LEFT]),
            "discards_remaining": _spread(global_values[:, GLOBAL_DISCARDS_LEFT]),
            "money": _spread(global_values[:, GLOBAL_MONEY]),
            "blind_progress": _spread(progress),
            "hand_cards": _spread(
                np.asarray(self.observations["hand_len"], dtype=np.float64)
            ),
            "joker_occupancy": _spread(jokers.astype(np.float64)),
            "shop_occupancy": _spread(shop.astype(np.float64)),
            "pack_occupancy": _spread(pack.astype(np.float64)),
            "distinct_hand_compositions": int(
                len({row.tobytes() for row in np.asarray(self.observations["hand"])})
            ),
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "simulator_version": self.simulator_version,
            "environment_backend": self.environment_backend,
            "generation_method": self.generation_method,
            "navigation_rule": self.navigation_rule,
            "navigation_seed": self.navigation_seed,
            "environment_seeds": list(self.environment_seeds),
            "sample_every": self.sample_every,
            "state_count": len(self),
            "state_hashes": list(self.state_hashes),
            "phase_labels": list(self.phase_labels),
            "phase_names": list(PHASE_NAMES),
            "source_seed": list(self.source_seed),
            "source_decision": list(self.source_decision),
            "stores_environment_snapshots": self.snapshots is not None,
            "contains_reward_or_strategy_labels": False,
            "excluded_fields": list(FORBIDDEN_CORPUS_FIELDS),
            "coverage": self.coverage(),
            "sha256": self.sha256,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, NDArray[Any]] = {
            f"obs__{key}": np.ascontiguousarray(value)
            for key, value in self.observations.items()
        }
        payload.update(
            {
                f"mask__{key}": np.ascontiguousarray(value)
                for key, value in self.masks.items()
            }
        )
        payload["phase_labels"] = np.asarray(self.phase_labels, dtype=np.int64)
        payload["source_seed"] = np.asarray(self.source_seed, dtype=np.int64)
        payload["source_decision"] = np.asarray(self.source_decision, dtype=np.int64)
        if self.snapshots is not None:
            offsets = np.cumsum([0, *(len(item) for item in self.snapshots)])
            payload["snapshot_bytes"] = np.frombuffer(
                b"".join(self.snapshots), dtype=np.uint8
            )
            payload["snapshot_offsets"] = offsets.astype(np.int64)
        np.savez_compressed(path, **payload)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    @classmethod
    def load(cls, path: Path, *, verify_hash: bool = True) -> "CalibrationCorpus":
        path = Path(path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as data:
            observations = {
                key: data[f"obs__{key}"].astype(dtype, copy=True)
                for key, (_, dtype) in OBS_SPEC.items()
            }
            masks = {
                key: data[f"mask__{key}"].astype(dtype, copy=True)
                for key, (_, dtype) in MASK_SPEC.items()
            }
            snapshots = None
            if "snapshot_bytes" in data:
                buffer = data["snapshot_bytes"].tobytes()
                offsets = data["snapshot_offsets"].astype(np.int64)
                snapshots = tuple(
                    buffer[int(offsets[index]) : int(offsets[index + 1])]
                    for index in range(len(offsets) - 1)
                )
            corpus = cls(
                version=str(manifest["version"]),
                simulator_version=str(manifest["simulator_version"]),
                environment_backend=str(manifest["environment_backend"]),
                generation_method=str(manifest["generation_method"]),
                navigation_rule=str(manifest["navigation_rule"]),
                navigation_seed=int(manifest["navigation_seed"]),
                environment_seeds=tuple(int(v) for v in manifest["environment_seeds"]),
                sample_every=int(manifest["sample_every"]),
                observations=observations,
                masks=masks,
                state_hashes=tuple(str(v) for v in manifest["state_hashes"]),
                phase_labels=tuple(int(v) for v in data["phase_labels"]),
                source_seed=tuple(int(v) for v in data["source_seed"]),
                source_decision=tuple(int(v) for v in data["source_decision"]),
                snapshots=snapshots,
            )
        if verify_hash and corpus.sha256 != manifest["sha256"]:
            raise ValueError("calibration corpus SHA-256 does not match its manifest")
        return corpus


def build_calibration_corpus(
    env: Any,
    *,
    environment_seeds: Sequence[int],
    states_per_seed: int,
    sample_every: int = 1,
    navigation_seed: int,
    maximum_decisions_per_seed: int | None = None,
    store_snapshots: bool = False,
) -> CalibrationCorpus:
    """Deterministically collect observable states with scripted navigation."""

    if env.num_envs != 1:
        raise ValueError("corpus generation uses one sequential environment")
    if states_per_seed < 1 or sample_every < 1 or not environment_seeds:
        raise ValueError("corpus generation needs positive seeds and state counts")
    budget = maximum_decisions_per_seed or states_per_seed * sample_every * 4
    observations_out: dict[str, list[NDArray[Any]]] = {key: [] for key in OBS_SPEC}
    masks_out: dict[str, list[NDArray[Any]]] = {key: [] for key in MASK_SPEC}
    hashes: list[str] = []
    phases: list[int] = []
    seed_of: list[int] = []
    decision_of: list[int] = []
    snapshots: list[bytes] = []
    from flylatro.evaluation.state_hash import hash_observation_row

    for seed in environment_seeds:
        observations, masks = env.reset((int(seed),))
        rng = np.random.default_rng(
            derive_seed("calibration-corpus-navigation", navigation_seed, int(seed))
        )
        collected = 0
        for decision in range(budget):
            if decision % sample_every == 0 and collected < states_per_seed:
                for key in OBS_SPEC:
                    observations_out[key].append(np.array(observations[key][0], copy=True))
                for key in MASK_SPEC:
                    masks_out[key].append(np.array(masks[key][0], copy=True))
                hashes.append(hash_observation_row(observations, 0))
                phases.append(_phase_label(observations, 0))
                seed_of.append(int(seed))
                decision_of.append(decision)
                collected += 1
                if store_snapshots:
                    snapshot = env.snapshot(0)
                    if not isinstance(snapshot, (bytes, bytearray)):
                        raise TypeError(
                            "this environment does not provide byte snapshots; "
                            "regenerate the corpus from its recorded seeds instead"
                        )
                    snapshots.append(bytes(snapshot))
            if collected >= states_per_seed:
                break
            actions = scripted_navigation_actions(masks, rng)
            step = env.step(actions)
            observations, masks = step.observations, step.masks
        if collected < states_per_seed:
            raise ValueError(
                f"seed {seed} produced {collected} of {states_per_seed} states "
                f"within {budget} decisions"
            )
    return CalibrationCorpus(
        version=CALIBRATION_CORPUS_VERSION,
        simulator_version=str(getattr(env, "simulator_version", "unknown")),
        environment_backend=type(env).__name__,
        generation_method=(
            "deterministic scripted legal navigation from fixed simulator seeds; "
            "navigation actions are discarded and are never training labels"
        ),
        navigation_rule=SCRIPTED_NAVIGATION_RULE,
        navigation_seed=int(navigation_seed),
        environment_seeds=tuple(int(seed) for seed in environment_seeds),
        sample_every=int(sample_every),
        observations={
            key: np.stack(values).astype(OBS_SPEC[key][1], copy=False)
            for key, values in observations_out.items()
        },
        masks={
            key: np.stack(values).astype(MASK_SPEC[key][1], copy=False)
            for key, values in masks_out.items()
        },
        state_hashes=tuple(hashes),
        phase_labels=tuple(phases),
        source_seed=tuple(seed_of),
        source_decision=tuple(decision_of),
        snapshots=tuple(snapshots) if store_snapshots else None,
    )


def _phase_label(observations: ObsDict, row: int) -> int:
    encoded = np.asarray(
        observations["global"][row, GLOBAL_PHASE_OFF : GLOBAL_PHASE_OFF + N_PHASES]
    )
    return int(np.argmax(encoded)) if bool(np.any(encoded)) else -1


def _spread(values: NDArray[np.float64]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "distinct": int(len(np.unique(np.round(array, 6)))),
    }
