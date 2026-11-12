"""Keep rare phases and legal motor contexts in a compact reward-free corpus."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from flylatro.analysis.corpus import CalibrationCorpus, PHASE_NAMES
from flylatro.analysis.coverage import CoveragePolicy, evaluate_coverage
from flylatro.interface.motor_contexts import motor_context_windows


CORPUS_SELECTION_VERSION = "reward-free-phase-context-selection-v1"


def select_calibration_corpus(
    source: CalibrationCorpus,
    *,
    states: int,
    states_per_stratum: int = 8,
    policy: CoveragePolicy | None = None,
) -> tuple[CalibrationCorpus, dict[str, Any]]:
    """Select distinct observable states, without changing navigation or masks.

    Rare phases and contexts receive the first slots. Within each stratum rows
    are spread across the source trajectory. Remaining slots sample the whole
    source. Coverage is measured after selection; missing evidence stays a
    failure rather than being fabricated or waived.
    """

    if states < 1 or states_per_stratum < 1:
        raise ValueError("state count and states per stratum must be positive")
    source_sha256 = source.sha256
    limits = policy or CoveragePolicy()
    phases = np.asarray(source.phase_labels)
    strata: dict[str, np.ndarray] = {
        f"phase:{name}": np.flatnonzero(phases == PHASE_NAMES.index(name))
        for name in limits.required_phases
    }
    windows = motor_context_windows(source.masks)
    for name in limits.required_motor_contexts:
        window = windows[name]
        strata[f"context:{name}"] = window.relevant_indices
        strata[f"competing:{name}"] = np.flatnonzero(
            window.relevant & (window.option_legal.sum(axis=1) > 1)
        )

    selected: set[int] = set()
    seen_hashes: set[str] = set()

    def add(index: int) -> None:
        state_hash = source.state_hashes[index]
        if len(selected) < states and state_hash not in seen_hashes:
            selected.add(index)
            seen_hashes.add(state_hash)

    def spread_rows(rows: np.ndarray) -> list[int]:
        # Deduplicate *before* spacing, so repeated blind-selection states do
        # not crowd out unique evidence at the end of a navigation stream.
        unique: dict[str, int] = {}
        for row in rows.tolist():
            unique.setdefault(source.state_hashes[row], row)
        candidates = list(unique.values())
        if not candidates:
            return []
        count = min(states_per_stratum, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        return [candidates[int(position)] for position in positions]

    for _, rows in sorted(strata.items(), key=lambda item: (len(item[1]), item[0])):
        for row in spread_rows(rows):
            add(row)
    remaining: dict[str, int] = {}
    for index, state_hash in enumerate(source.state_hashes):
        if state_hash not in seen_hashes:
            remaining.setdefault(state_hash, index)
    candidates = list(remaining.values())
    count = min(states - len(selected), len(candidates))
    if count:
        for position in np.linspace(0, len(candidates) - 1, count, dtype=np.int64):
            add(candidates[int(position)])
    indices = sorted(selected)

    def subset(values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(values[index] for index in indices) if values else ()

    corpus = replace(
        source,
        generation_method=(
            f"{source.generation_method}; {CORPUS_SELECTION_VERSION}: "
            f"source_sha256={source_sha256}, states={states}, "
            f"states_per_stratum={states_per_stratum}, policy_sha256={limits.sha256}; "
            "selection uses phase and legality masks only"
        ),
        observations={key: value[indices].copy() for key, value in source.observations.items()},
        masks={key: value[indices].copy() for key, value in source.masks.items()},
        state_hashes=subset(source.state_hashes),
        phase_labels=subset(source.phase_labels),
        source_seed=subset(source.source_seed),
        source_decision=subset(source.source_decision),
        source_run_seed=subset(source.source_run_seed),
        source_episode=subset(source.source_episode),
        source_collection_index=subset(source.source_collection_index),
        snapshots=subset(source.snapshots) if source.snapshots is not None else None,
    )
    coverage = evaluate_coverage(corpus.coverage(), limits)
    return corpus, {
        "version": CORPUS_SELECTION_VERSION,
        "source_sha256": source_sha256,
        "selected_sha256": corpus.sha256,
        "source_states": len(source),
        "requested_states": states,
        "selected_states": len(corpus),
        "source_row_indices": indices,
        "states_per_stratum": states_per_stratum,
        "source_coverage": evaluate_coverage(source.coverage(), limits),
        "coverage": coverage,
        "status": "PASS" if coverage["satisfied"] else "FAIL",
    }
