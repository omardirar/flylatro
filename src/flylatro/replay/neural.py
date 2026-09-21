"""Opt-in streaming neural and plastic event recording."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


NEURAL_SCHEMA_VERSION = 3


class NeuralEventRecorder:
    """Write one compressed Parquet row group per decision.

    Only the current decision is buffered. Long showcases therefore have
    bounded recorder memory and can be selectively read by row group.
    """

    def __init__(self, output_path: Path) -> None:
        if output_path.suffix != ".parquet":
            raise ValueError("neural event output must use .parquet")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Parquet neural recording requires Flylatro's 'recording' extra") from error
        self.pa = pa
        self.pq = pq
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = output_path.with_suffix(".parquet.tmp")
        dictionary = pa.dictionary(pa.int8(), pa.string())
        self.schema = pa.schema(
            [
                ("decision_id", pa.int32()),
                ("simulated_time_ms", pa.float32()),
                ("flywire_neuron_id", pa.int64()),
                ("role", dictionary),
                ("activity", pa.float32()),
                ("event_kind", dictionary),
                ("pre_root_id", pa.int64()),
                ("post_root_id", pa.int64()),
                ("old_efficacy", pa.float32()),
                ("new_efficacy", pa.float32()),
                ("efficacy_delta", pa.float32()),
            ],
            metadata={
                b"flylatro_schema_version": str(NEURAL_SCHEMA_VERSION).encode(),
                b"row_group_contract": b"one decision per row group",
            },
        )
        self.writer = pq.ParquetWriter(
            self.temporary, self.schema, compression="zstd", use_dictionary=["role", "event_kind"]
        )
        self._current_decision: int | None = None
        self._chunks: list[dict[str, np.ndarray]] = []
        self._closed = False

    def record(
        self,
        *,
        decision_id: int,
        times_ms: Sequence[float] | np.ndarray,
        neuron_ids: Sequence[int] | np.ndarray,
        roles: Sequence[str] | np.ndarray | None = None,
        activities: Sequence[float] | np.ndarray | None = None,
        event_kind: str | Sequence[str] | np.ndarray = "spike",
        pre_root_ids: Sequence[int] | np.ndarray | None = None,
        post_root_ids: Sequence[int] | np.ndarray | None = None,
        old_efficacy: Sequence[float] | np.ndarray | None = None,
        new_efficacy: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("neural recorder is closed")
        if self._current_decision is not None and decision_id < self._current_decision:
            raise ValueError("decision IDs must be recorded in nondecreasing order")
        if self._current_decision is not None and decision_id != self._current_decision:
            self._flush()
        self._current_decision = int(decision_id)
        times = np.asarray(times_ms, dtype=np.float32)
        neurons = np.asarray(neuron_ids, dtype=np.int64)
        if times.ndim != 1 or neurons.shape != times.shape:
            raise ValueError("times and neuron IDs must be matching vectors")
        count = len(times)
        role_values = np.full(count, "internal", dtype=object) if roles is None else np.asarray(roles, dtype=object)
        activity_values = np.ones(count, dtype=np.float32) if activities is None else np.asarray(activities, dtype=np.float32)
        kind_values = np.full(count, event_kind, dtype=object) if isinstance(event_kind, str) else np.asarray(event_kind, dtype=object)
        if any(array.shape != times.shape for array in (role_values, activity_values, kind_values)):
            raise ValueError("roles, activities and event kinds must match event count")
        allowed_kinds = {"spike", "stimulation", "activity", "synthetic_reinforcement", "plasticity", "weight_snapshot"}
        allowed_roles = {"input", "internal", "readout", "kc", "mbon", "dan_anatomy", "descending", "synthetic_appetitive", "synthetic_aversive", "plasticity"}
        if any(str(value) not in allowed_kinds for value in kind_values):
            raise ValueError(f"event kinds must be one of {sorted(allowed_kinds)}")
        if any(str(value) not in allowed_roles for value in role_values):
            raise ValueError(f"neural roles must be one of {sorted(allowed_roles)}")
        def ints(values: Sequence[int] | np.ndarray | None) -> np.ndarray:
            return np.full(count, -1, dtype=np.int64) if values is None else np.asarray(values, dtype=np.int64)
        def floats(values: Sequence[float] | np.ndarray | None) -> np.ndarray:
            return np.full(count, np.nan, dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
        pre = ints(pre_root_ids)
        post = ints(post_root_ids)
        old = floats(old_efficacy)
        new = floats(new_efficacy)
        if any(array.shape != times.shape for array in (pre, post, old, new)):
            raise ValueError("plastic edge details must match event count")
        self._chunks.append({
            "decision_id": np.full(count, decision_id, dtype=np.int32),
            "simulated_time_ms": times,
            "flywire_neuron_id": neurons,
            "role": role_values,
            "activity": activity_values,
            "event_kind": kind_values,
            "pre_root_id": pre,
            "post_root_id": post,
            "old_efficacy": old,
            "new_efficacy": new,
            "efficacy_delta": new - old,
        })

    def _flush(self) -> None:
        if not self._chunks:
            return
        arrays = {
            name: np.concatenate([chunk[name] for chunk in self._chunks])
            for name in self._chunks[0]
        }
        table = self.pa.Table.from_pydict(arrays, schema=self.schema)
        self.writer.write_table(table, row_group_size=len(table))
        self._chunks.clear()

    def close(self) -> Path:
        if self._closed:
            return self.output_path
        self._flush()
        self.writer.close()
        self.temporary.replace(self.output_path)
        self._closed = True
        return self.output_path

    def __enter__(self) -> "NeuralEventRecorder":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.writer.close()
