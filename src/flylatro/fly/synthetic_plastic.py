"""Cheap plastic mushroom-body circuit for local integration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.env.upstream_contract import ObsDict
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.fly.plastic_backend import PlasticFlyDecisionActivity
from flylatro.fly.plastic_features import feature_names, observation_features
from flylatro.interface.motor import HEAD_SIZES


@dataclass(frozen=True, slots=True)
class SyntheticPlasticCircuitSpec:
    seed: int = 1701
    kenyon_count: int = 48
    output_count: int = sum(HEAD_SIZES.values())
    kc_active_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.kenyon_count < 2 or self.output_count < sum(HEAD_SIZES.values()):
            raise ValueError("synthetic circuit populations are too small")
        if not 0 < self.kc_active_fraction <= 1:
            raise ValueError("kc_active_fraction must be in (0, 1]")


class SyntheticPlasticFlyProcessor:
    """Fixed sensory expansion + plastic KC->MBON + fixed downstream graph.

    This is an engineering test double. It validates learning state and the two
    output modes but supplies no evidence about real FlyWire dynamics.
    """

    version = "synthetic-plastic-mushroom-body-v1"
    duration_ms = 1.0

    def __init__(
        self,
        spec: SyntheticPlasticCircuitSpec | None = None,
        *,
        mode: str = "mbon_direct",
    ) -> None:
        self.spec = spec or SyntheticPlasticCircuitSpec()
        if mode not in {"mbon_direct", "whole_brain"}:
            raise ValueError("unknown output mode")
        self.mode = mode
        rng = np.random.default_rng(self.spec.seed)
        feature_count = len(feature_names())
        self._sensory_projection = rng.normal(
            0.0, 1.0 / np.sqrt(feature_count), (feature_count, self.spec.kenyon_count)
        ).astype(np.float32)
        pre_local = np.repeat(
            np.arange(self.spec.kenyon_count, dtype=np.int64),
            self.spec.output_count,
        )
        post_local = np.tile(
            np.arange(self.spec.output_count, dtype=np.int64),
            self.spec.kenyon_count,
        )
        anatomical = rng.uniform(0.5, 1.5, len(pre_local)).astype(np.float32)
        self.topology = PlasticEdgeTopology.synthetic(
            pre_local,
            post_local + self.spec.kenyon_count,
            anatomical,
        )
        self._edge_pre_local = pre_local
        self._edge_post_local = post_local
        downstream = np.eye(self.spec.output_count, dtype=np.float32)
        downstream += rng.uniform(
            0.0,
            0.05,
            (self.spec.output_count, self.spec.output_count),
        ).astype(np.float32)
        downstream /= downstream.sum(axis=0, keepdims=True)
        self._downstream = downstream
        root_start = 9_000_000 if mode == "mbon_direct" else 9_100_000
        self.output_root_ids = np.arange(
            root_start, root_start + self.spec.output_count, dtype=np.int64
        )
        self.kc_root_ids = np.arange(
            8_900_000, 8_900_000 + self.spec.kenyon_count, dtype=np.int64
        )
        self.mbon_root_ids = np.arange(
            9_000_000, 9_000_000 + self.spec.output_count, dtype=np.int64
        )
        self.dan_root_ids = np.asarray([8_800_001, 8_800_002], dtype=np.int64)
        self.pam_root_ids = self.dan_root_ids[:1]
        self.ppl1_root_ids = self.dan_root_ids[1:]
        self.descending_root_ids = np.arange(
            9_100_000, 9_100_000 + self.spec.output_count, dtype=np.int64
        )

    def process(
        self,
        observations: ObsDict,
        *,
        fly_seeds: Sequence[int],
        efficacy: NDArray[np.floating],
    ) -> PlasticFlyDecisionActivity:
        del fly_seeds  # this deterministic test double has no neural noise
        features = observation_features(observations).astype(np.float32, copy=False)
        batch = features.shape[0]
        efficacy_values = np.asarray(efficacy, dtype=np.float32)
        if efficacy_values.shape != (batch, self.topology.edge_count):
            raise ValueError("one efficacy vector is required per synthetic fly")
        drive = np.maximum(features @ self._sensory_projection, 0.0)
        active_count = max(
            1, int(round(self.spec.kenyon_count * self.spec.kc_active_fraction))
        )
        threshold = np.partition(drive, -active_count, axis=1)[:, -active_count]
        kc = np.where(drive >= threshold[:, None], drive, 0.0)
        peak = np.maximum(kc.max(axis=1, keepdims=True), 1e-8)
        kc = kc / peak
        kc_hz = kc * 20.0
        edge_pre = kc_hz[:, self._edge_pre_local]
        edge_post = np.zeros_like(edge_pre)
        mbon = np.zeros((batch, self.spec.output_count), dtype=np.float32)
        effective = efficacy_values * self.topology.anatomical_weights[None, :]
        contributions = kc[:, self._edge_pre_local] * effective
        for row in range(batch):
            np.add.at(mbon[row], self._edge_post_local, contributions[row])
        scale = np.maximum(mbon.mean(axis=1, keepdims=True), 1e-8)
        mbon = np.tanh(mbon / scale)
        mbon_hz = mbon * 20.0
        edge_post[:] = mbon_hz[:, self._edge_post_local]
        descending = np.tanh(mbon @ self._downstream)
        descending_hz = descending * 20.0
        output = mbon_hz if self.mode == "mbon_direct" else descending_hz
        return PlasticFlyDecisionActivity(
            output_activity=output.astype(np.float32, copy=False),
            edge_pre_activity=edge_pre.astype(np.float32, copy=False),
            edge_post_activity=edge_post.astype(np.float32, copy=False),
            kc_activity=kc_hz.astype(np.float32, copy=False),
            mbon_activity=mbon_hz.astype(np.float32, copy=False),
            dan_activity=np.zeros((batch, 2), dtype=np.float32),
            descending_activity=descending_hz.astype(np.float32, copy=False),
            duration_ms=self.duration_ms,
            activity_unit="synthetic_rate_hz_proxy",
        )
