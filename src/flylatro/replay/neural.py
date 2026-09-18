"""Optional compact neural-event recording for selected evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


NEURAL_SCHEMA_VERSION = 2


class NeuralEventRecorder:
    """Accumulate selected events and write a typed, compressed Parquet table."""

    def __init__(self, output_path: Path) -> None:
        if output_path.suffix != ".parquet":
            raise ValueError("neural event output must use .parquet")
        self.output_path = output_path
        self._decision: list[np.ndarray] = []
        self._time: list[np.ndarray] = []
        self._neuron: list[np.ndarray] = []
        self._role: list[np.ndarray] = []
        self._activity: list[np.ndarray] = []
        self._event_kind: list[np.ndarray] = []

    def record(
        self,
        *,
        decision_id: int,
        times_ms: Sequence[float] | np.ndarray,
        neuron_ids: Sequence[int] | np.ndarray,
        roles: Sequence[str] | np.ndarray | None = None,
        activities: Sequence[float] | np.ndarray | None = None,
        event_kind: str | Sequence[str] | np.ndarray = "spike",
    ) -> None:
        times = np.asarray(times_ms, dtype=np.float32)
        neurons = np.asarray(neuron_ids, dtype=np.int64)
        if times.ndim != 1 or neurons.shape != times.shape:
            raise ValueError("times and neuron IDs must be matching vectors")
        count = len(times)
        role_values = (
            np.full(count, "internal", dtype=object)
            if roles is None else np.asarray(roles, dtype=object)
        )
        activity_values = (
            np.ones(count, dtype=np.float32)
            if activities is None else np.asarray(activities, dtype=np.float32)
        )
        if role_values.shape != times.shape or activity_values.shape != times.shape:
            raise ValueError("roles and activities must match event count")
        kind_values = (
            np.full(count, event_kind, dtype=object)
            if isinstance(event_kind, str) else np.asarray(event_kind, dtype=object)
        )
        if kind_values.shape != times.shape:
            raise ValueError("event kinds must match event count")
        allowed_kinds = {
            "spike",
            "stimulation",
            "activity",
            "dopamine",
            "plasticity",
            "weight_snapshot",
        }
        if any(str(kind) not in allowed_kinds for kind in kind_values):
            raise ValueError(f"event kinds must be one of {sorted(allowed_kinds)}")
        allowed = {
            "input",
            "internal",
            "readout",
            "kc",
            "mbon",
            "dan",
            "descending",
            "plasticity",
        }
        if any(str(role) not in allowed for role in role_values):
            raise ValueError(f"neural roles must be one of {sorted(allowed)}")
        self._decision.append(np.full(count, decision_id, dtype=np.int32))
        self._time.append(times)
        self._neuron.append(neurons)
        self._role.append(role_values)
        self._activity.append(activity_values)
        self._event_kind.append(kind_values)

    def close(self) -> Path:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError(
                "Parquet neural recording requires Flylatro's 'recording' extra"
            ) from error
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        def concatenate(chunks: list[np.ndarray], dtype: object) -> np.ndarray:
            return np.concatenate(chunks) if chunks else np.asarray([], dtype=dtype)

        table = pa.table(
            {
                "decision_id": concatenate(self._decision, np.int32),
                "simulated_time_ms": concatenate(self._time, np.float32),
                "flywire_neuron_id": concatenate(self._neuron, np.int64),
                "role": concatenate(self._role, object),
                "activity": concatenate(self._activity, np.float32),
                "event_kind": concatenate(self._event_kind, object),
            },
            schema=pa.schema(
                [
                    ("decision_id", pa.int32()),
                    ("simulated_time_ms", pa.float32()),
                    ("flywire_neuron_id", pa.int64()),
                    ("role", pa.string()),
                    ("activity", pa.float32()),
                    ("event_kind", pa.string()),
                ],
                metadata={b"flylatro_schema_version": str(NEURAL_SCHEMA_VERSION).encode()},
            ),
        )
        temporary = self.output_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(self.output_path)
        return self.output_path

    def __enter__(self) -> "NeuralEventRecorder":
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
