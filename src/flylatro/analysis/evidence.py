"""Shared plumbing for producing provenance-bound evidence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from flylatro.analysis.corpus import CalibrationCorpus, PHASE_NAMES
from flylatro.analysis.provenance import EvidenceIdentity
from flylatro.env.upstream_contract import (
    BLIND_REQ_OFF,
    BLIND_SCORED_OFF,
    GLOBAL_ANTE_OFF,
    GLOBAL_HANDS_LEFT,
)
from flylatro.learning.config import (
    PlasticExperimentConfig,
    calibration_corpus_hash,
    resolve_path,
)
from flylatro.seeds import derive_seed


@dataclass(frozen=True, slots=True)
class CorpusActivity:
    """Neural activity recorded over a frozen reward-free corpus."""

    kc: NDArray[np.float64]
    mbon: NDArray[np.float64]
    descending: NDArray[np.float64]
    output: NDArray[np.float64]
    state_labels: NDArray[np.int64]
    state_hashes: tuple[str, ...]
    repeats: int
    duration_ms: float


def load_corpus(config: PlasticExperimentConfig, override: Path | None = None) -> CalibrationCorpus:
    """Load the frozen calibration corpus this experiment is bound to."""

    path = override if override is not None else (
        resolve_path(config, config.calibration.corpus_path)
        if config.calibration.corpus_path
        else None
    )
    if path is None:
        raise ValueError(
            "a frozen reward-free calibration corpus is required; build one with "
            "flylatro-build-calibration-corpus and set [calibration].corpus_path"
        )
    return CalibrationCorpus.load(Path(path))


def record_corpus_activity(
    stack: Any,
    corpus: CalibrationCorpus,
    *,
    seed: int,
    repeats: int = 2,
    batch_size: int = 1,
    limit: int | None = None,
) -> CorpusActivity:
    """Run the fixed neural stack over corpus states with initial efficacy.

    Efficacy is held at its initial value, so this measures representation, not
    learning, and it observes no reward or outcome of any kind.
    """

    if repeats < 1 or batch_size < 1:
        raise ValueError("repeats and batch size must be positive")
    processor = stack.agent.processor
    edges = stack.agent.plasticity.topology.edge_count
    count = len(corpus) if limit is None else min(limit, len(corpus))
    kc: list[NDArray[np.float64]] = []
    mbon: list[NDArray[np.float64]] = []
    descending: list[NDArray[np.float64]] = []
    output: list[NDArray[np.float64]] = []
    labels: list[int] = []
    hashes: list[str] = []
    for repeat in range(repeats):
        for start in range(0, count, batch_size):
            indices = list(range(start, min(start + batch_size, count)))
            observations, _ = corpus.batch(indices)
            fly_seeds = tuple(
                derive_seed("corpus-activity", seed, repeat, index) for index in indices
            )
            activity = processor.process(
                observations,
                fly_seeds=fly_seeds,
                efficacy=np.ones((len(indices), edges), dtype=np.float32),
            )
            kc.append(np.asarray(activity.kc_activity, dtype=np.float64))
            mbon.append(np.asarray(activity.mbon_activity, dtype=np.float64))
            descending.append(np.asarray(activity.descending_activity, dtype=np.float64))
            output.append(np.asarray(activity.output_activity, dtype=np.float64))
            labels.extend(indices)
            hashes.extend(corpus.state_hashes[index] for index in indices)
    return CorpusActivity(
        kc=np.concatenate(kc, axis=0),
        mbon=np.concatenate(mbon, axis=0),
        descending=np.concatenate(descending, axis=0),
        output=np.concatenate(output, axis=0),
        state_labels=np.asarray(labels, dtype=np.int64),
        state_hashes=tuple(hashes),
        repeats=repeats,
        duration_ms=float(getattr(processor, "duration_ms", 0.0)),
    )


def observable_category_labels(
    corpus: CalibrationCorpus, order: Sequence[int]
) -> dict[str, NDArray[np.int64]]:
    """Diagnostic-only observable labels. These never reach the policy."""

    selected = np.asarray(order, dtype=np.int64)
    global_values = np.asarray(corpus.observations["global"], dtype=np.float64)[selected]
    blind = np.asarray(corpus.observations["blind"], dtype=np.float64)[selected]
    hand = np.asarray(corpus.observations["hand"], dtype=np.float64)[selected]
    shop = np.asarray(corpus.observations["shop_feats"], dtype=np.float64)[selected]
    ante_hot = global_values[:, GLOBAL_ANTE_OFF : GLOBAL_ANTE_OFF + 9]
    ante = np.where(ante_hot.any(axis=1), ante_hot.argmax(axis=1) + 1, 0)
    required = np.expm1(blind[:, BLIND_REQ_OFF])
    scored = np.expm1(blind[:, BLIND_SCORED_OFF])
    progress = np.clip(
        np.round(10 * scored / np.maximum(required, 1e-12)), 0, 10
    ).astype(np.int64)
    rank_bits = (hand[:, :, :13] > 0).any(axis=1)
    rank_signature = np.asarray(
        [
            int(sum(1 << index for index, present in enumerate(row) if present))
            for row in rank_bits
        ],
        dtype=np.int64,
    )
    return {
        "phase": np.asarray([corpus.phase_labels[index] for index in selected], dtype=np.int64),
        "ante": ante.astype(np.int64),
        "hands_remaining": np.round(global_values[:, GLOBAL_HANDS_LEFT] * 1000).astype(np.int64),
        "blind_progress_decile": progress,
        "rank_presence_signature": rank_signature,
        "shop_presence": (np.abs(shop).sum(axis=(1, 2)) > 0).astype(np.int64),
    }


def experiment_identity(
    config: PlasticExperimentConfig,
    components: Mapping[str, Any],
    *,
    report_kind: str,
    report_version: str,
    corpus: CalibrationCorpus | None = None,
) -> EvidenceIdentity:
    return EvidenceIdentity.from_components(
        components,
        report_kind=report_kind,
        report_version=report_version,
        config_sha256=config.sha256,
        simulator_version=str(components.get("simulator_version", "unknown")),
        calibration_corpus_sha256=(
            corpus.sha256 if corpus is not None else calibration_corpus_hash(config)
        ),
        device=config.fly.device,
        backend_evidence=(
            "real_flywire_spike_rate_hz"
            if config.fly.backend == "flywire"
            else "synthetic_rate_hz_proxy_development_only"
        ),
    )


def write_report(path: Path, report: Mapping[str, Any], identity: EvidenceIdentity) -> Path:
    payload = {**dict(report), "evidence_identity": identity.to_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def phase_name(index: int) -> str:
    return PHASE_NAMES[index] if 0 <= index < len(PHASE_NAMES) else "unlabelled"
