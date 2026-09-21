"""Shared fixtures for fast, artifact-free regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np

from flylatro.analysis.corpus import CalibrationCorpus, build_calibration_corpus
from flylatro.env.array_mock import MockArrayBalatroEnv
from flylatro.fly.flywire_artifact import FlyWireArtifact
from flylatro.learning.protocol_materialize import render_toml

SMOKE_CONFIG = Path("configs/plastic-smoke.toml")


def build_mock_corpus(
    path: Path,
    *,
    seeds: tuple[int, ...] = (900_001, 900_002, 900_003),
    states_per_seed: int = 3,
    navigation_seed: int = 4242,
) -> CalibrationCorpus:
    env = MockArrayBalatroEnv(1, blind_target=18.0, initial_hands=3, initial_discards=2)
    corpus = build_calibration_corpus(
        env,
        environment_seeds=seeds,
        states_per_seed=states_per_seed,
        navigation_seed=navigation_seed,
    )
    corpus.save(path)
    return corpus


def write_config(
    tmp_path: Path,
    *,
    corpus_path: Path | None = None,
    name: str = "test-smoke",
    overrides: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Write a smoke configuration into ``tmp_path`` with optional overrides."""

    raw = tomllib.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    document: dict[str, object] = {"name": name}
    for section in (
        "environment",
        "fly",
        "motor",
        "plasticity",
        "reinforcement",
        "training",
        "curriculum",
        "runtime",
        "calibration",
        "protocol",
    ):
        document[section] = dict(raw.get(section, {}))
    if corpus_path is not None:
        document["calibration"] = {
            **document["calibration"],  # type: ignore[dict-item]
            "corpus_path": str(corpus_path.resolve()),
        }
    for section, values in (overrides or {}).items():
        document[section] = {**document.get(section, {}), **values}  # type: ignore[dict-item]
    # The repository root is two levels above the config path, so keep the same
    # depth the loader expects for relative artefact paths.
    directory = tmp_path / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.toml"
    path.write_text(render_toml(document), encoding="utf-8")
    return path


def tiny_artifact(
    *,
    neurons: int = 24,
    kenyon: tuple[int, ...] = (0, 1, 2),
    mbon: tuple[int, ...] = (4, 5, 6, 7),
    weights: tuple[float, ...] = (3.0, 2.0, 5.0, 1.0),
) -> FlyWireArtifact:
    """A minimal artifact for canonical-candidate and topology unit tests."""

    pre = np.asarray([0, 1, 2, 0], dtype=np.int64)
    post = np.asarray([4, 5, 6, 5], dtype=np.int64)
    types = ["KC"] * 4 + ["MBON-gamma1pedc>a/b", "MBON-a2", "MBON-b1", "MBON-c1"]
    types += ["ALPN"] * 4 + ["DN"] * 4 + ["other"] * (neurons - len(types) - 8)
    return FlyWireArtifact(
        path=Path("tiny.npz"),
        manifest={"connectivity_sha256": "x", "artifact_sha256": "y"},
        root_ids=np.arange(1_000, 1_000 + neurons, dtype=np.int64),
        pre_indices=pre,
        post_indices=post,
        signed_synapse_counts=np.asarray(weights, dtype=np.float32),
        sensory_indices=np.asarray([8, 9, 10, 11], dtype=np.int64),
        descending_indices=np.asarray([12, 13, 14, 15], dtype=np.int64),
        coordinates_nm=np.zeros((neurons, 3), dtype=np.float32),
        kenyon_indices=np.asarray(kenyon, dtype=np.int64),
        mbon_indices=np.asarray(mbon, dtype=np.int64),
        dan_indices=np.asarray([16], dtype=np.int64),
        projection_indices=np.asarray([8, 9, 10, 11], dtype=np.int64),
        kc_mbon_edge_indices=np.arange(4, dtype=np.int64),
        primary_types=np.asarray(types, dtype=np.str_),
    )


class ScriptedPhaseEnv:
    """Contract-faithful array environment that visits every observable phase.

    This is a *test double* for the pinned simulator's **contract**, not a model
    of Balatro.  It reproduces the mask semantics, the phase one-hot, the
    ``1 + index`` vocabulary convention, auto-reset with a new run seed, and —
    crucially — the strict referee, so an action the real
    ``BalatroVecEnv(strict=True)`` would reject also fails here.  It carries no
    reward, no strategy information and no action quality signal.
    """

    simulator_version = "scripted-phase-contract-double-v1"

    #: Action types legal in each observable phase, before inventory actions.
    PHASE_ACTIONS: dict[int, tuple[int, ...]] = {
        0: (2, 3),        # BLIND_SELECT: SELECT_BLIND, SKIP_BLIND
        1: (0, 1),        # PLAYING: PLAY_HAND, DISCARD
        2: (4,),          # ROUND_EVAL: CASH_OUT
        3: (5, 6, 7),     # SHOP: BUY_SHOP, REROLL, LEAVE_SHOP
        4: (11, 12),      # PACK: PICK_PACK, SKIP_PACK
    }
    PHASES = (0, 1, 1, 2, 3, 4)

    def __init__(
        self,
        num_envs: int = 1,
        *,
        jokers: int = 3,
        consumables: int = 2,
        hand_len: int = 6,
        shop_slots: int = 4,
        pack_slots: int = 3,
    ) -> None:
        from flylatro.env.upstream_contract import validate_strict_action_batch

        self.num_envs = num_envs
        self.jokers = jokers
        self.consumables = consumables
        self.hand_len = hand_len
        self.shop_slots = shop_slots
        self.pack_slots = pack_slots
        self._validate = validate_strict_action_batch
        self._seeds: list[int] = []
        self._steps: list[int] = []
        self._episodes: list[int] = []
        self._runs: list[int] = []
        self._win_ante = 8

    # -- environment protocol ------------------------------------------------

    def reset(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError("one seed per environment")
        self._seeds = [int(seed) for seed in seeds]
        self._steps = [0] * self.num_envs
        self._episodes = [0] * self.num_envs
        self._runs = [int(seed) for seed in seeds]
        return self._observe(), self._masks()

    def step(self, actions):
        from flylatro.env.balatro_sim import ArrayStep

        masks = self._masks()
        self._validate(actions, masks)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos = []
        for row in range(self.num_envs):
            self._steps[row] += 1
            terminal = self._steps[row] % len(self.PHASES) == 0
            dones[row] = terminal
            if terminal:
                # Auto-reset: the same root seed continues, but the run behind
                # it is a different one, exactly like the pinned vector env.
                self._episodes[row] += 1
                self._runs[row] = self._runs[row] * 31 + 17
                infos.append(
                    {
                        "reward_components": {
                            "blind_progress": 0.0,
                            "blind_clear": 0.0,
                            "win": 0.0,
                        },
                        "episode": {"r": 0.0, "l": len(self.PHASES), "ante": 1, "won": False},
                    }
                )
            else:
                infos.append(
                    {
                        "reward_components": {
                            "blind_progress": 0.0,
                            "blind_clear": 0.0,
                            "win": 0.0,
                        }
                    }
                )
        return ArrayStep(
            self._observe(),
            self._masks(),
            np.zeros(self.num_envs, dtype=np.float32),
            dones,
            tuple(infos),
        )

    def set_shaping_beta(self, beta: float) -> None:
        return None

    def set_win_ante(self, win_ante: int) -> None:
        self._win_ante = int(win_ante)

    def run_seed(self, env_index: int) -> str:
        return f"RUN{self._runs[env_index]:08d}"

    def snapshot(self, env_index: int) -> bytes:
        return json.dumps(
            {
                "seed": self._seeds[env_index],
                "step": self._steps[env_index],
                "episode": self._episodes[env_index],
                "run": self._runs[env_index],
            },
            sort_keys=True,
        ).encode()

    def restore(self, env_index: int, snapshot: bytes) -> None:
        payload = json.loads(bytes(snapshot).decode())
        self._seeds[env_index] = int(payload["seed"])
        self._steps[env_index] = int(payload["step"])
        self._episodes[env_index] = int(payload["episode"])
        self._runs[env_index] = int(payload["run"])

    def snapshot_all(self):
        return tuple(self.snapshot(index) for index in range(self.num_envs))

    def restore_all(self, snapshots) -> None:
        for index, snapshot in enumerate(snapshots):
            self.restore(index, snapshot)

    # -- contract encoding ---------------------------------------------------

    def phase(self, row: int) -> int:
        return self.PHASES[self._steps[row] % len(self.PHASES)]

    def _observe(self):
        from flylatro.env.upstream_contract import (
            BLIND_REQ_OFF,
            BLIND_SCORED_OFF,
            GLOBAL_ANTE_OFF,
            GLOBAL_DISCARDS_LEFT,
            GLOBAL_HANDS_LEFT,
            GLOBAL_MONEY,
            GLOBAL_PHASE_OFF,
            GLOBAL_ROUND,
            OBS_SPEC,
        )

        observations = {
            key: np.zeros((self.num_envs, *shape), dtype=dtype)
            for key, (shape, dtype) in OBS_SPEC.items()
        }
        for row in range(self.num_envs):
            phase = self.phase(row)
            step = self._steps[row]
            rng = np.random.default_rng(self._runs[row] * 1_000 + step)
            observations["global"][row, GLOBAL_PHASE_OFF + phase] = 1.0
            observations["global"][row, GLOBAL_ANTE_OFF] = 1.0
            observations["global"][row, GLOBAL_HANDS_LEFT] = (3 - step % 3) / 4.0
            observations["global"][row, GLOBAL_DISCARDS_LEFT] = (2 - step % 2) / 4.0
            observations["global"][row, GLOBAL_MONEY] = float(np.log1p(4 + step))
            observations["global"][row, GLOBAL_ROUND] = (1 + step % 3) / 24.0
            observations["blind"][row, BLIND_REQ_OFF] = float(np.log1p(300.0))
            observations["blind"][row, BLIND_SCORED_OFF] = float(np.log1p(20.0 * step))
            observations["hand_len"][row] = self.hand_len
            for card in range(self.hand_len):
                observations["hand"][row, card, int(rng.integers(0, 13))] = 1.0
                observations["hand"][row, card, 13 + (card % 4)] = 1.0
            # Vocabulary ids are 1-based; 0 means "empty slot".
            for slot in range(self.jokers):
                observations["joker_ids"][row, slot] = 1 + slot + step % 3
            observations["consumables_len"][row] = self.consumables
            for slot in range(self.consumables):
                observations["consumable_ids"][row, slot] = 1 + slot
            if phase == 3:
                for slot in range(self.shop_slots):
                    observations["shop_ids"][row, slot] = 1 + slot + step
                    observations["shop_feats"][row, slot, 0] = float(np.log1p(3 + slot))
            observations["deck_counts"][row] = 1.0
            observations["drawpile_counts"][row] = 1.0
        return observations

    def _masks(self):
        from flylatro.env.upstream_contract import MASK_SPEC, UpstreamActionType

        masks = {
            key: np.zeros((self.num_envs, *shape), dtype=dtype)
            for key, (shape, dtype) in MASK_SPEC.items()
        }
        for row in range(self.num_envs):
            phase = self.phase(row)
            for action_type in self.PHASE_ACTIONS[phase]:
                masks["action_type_mask"][row, action_type] = True
            # Inventory actions are legal in every interactive phase, exactly
            # like `push_inventory_actions` upstream.
            if self.jokers:
                masks["action_type_mask"][row, int(UpstreamActionType.SELL_JOKER)] = True
                masks["joker_target_mask"][row, : self.jokers] = True
            if self.consumables:
                masks["action_type_mask"][row, int(UpstreamActionType.SELL_CONSUMABLE)] = True
                masks["action_type_mask"][row, int(UpstreamActionType.USE_CONSUMABLE)] = True
                masks["consumable_target_mask"][row, : self.consumables] = True
            if phase == 3:
                masks["shop_target_mask"][row, : self.shop_slots] = True
            if phase == 4:
                masks["pack_target_mask"][row, : self.pack_slots] = True
            cards_needed = any(
                masks["action_type_mask"][row, value]
                for value in (
                    int(UpstreamActionType.PLAY_HAND),
                    int(UpstreamActionType.DISCARD),
                    int(UpstreamActionType.USE_CONSUMABLE),
                )
            )
            if cards_needed:
                masks["card_select_mask"][row, : self.hand_len] = True
        return masks


def build_phase_corpus(
    path: Path,
    *,
    seeds: tuple[int, ...] = (11, 12, 13),
    states_per_seed: int = 12,
    navigation_seed: int = 909,
    store_snapshots: bool = False,
) -> CalibrationCorpus:
    """A corpus that actually reaches BLIND_SELECT, PLAYING, ROUND_EVAL, SHOP and PACK."""

    corpus = build_calibration_corpus(
        ScriptedPhaseEnv(1),
        environment_seeds=seeds,
        states_per_seed=states_per_seed,
        navigation_seed=navigation_seed,
        maximum_decisions_per_seed=states_per_seed * 4,
        store_snapshots=store_snapshots,
    )
    corpus.save(path)
    return corpus
