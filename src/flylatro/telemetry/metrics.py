"""JSONL metrics with optional TensorBoard mirroring."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping
from flylatro.telemetry.buffered import BufferedTextWriter


class MetricLogger:
    def __init__(
        self, run_dir: Path, *, run_id: str, tensorboard: bool = True,
        buffer_bytes: int = 262144, flush_seconds: float = 0.0
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = run_dir / "metrics.jsonl"
        self._stream = BufferedTextWriter(self.path, max_bytes=buffer_bytes, flush_seconds=flush_seconds)
        self._writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except (ImportError, ModuleNotFoundError):
                pass
            else:
                self._writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "step": int(step),
            "metrics": {
                key: (float(value) if math.isfinite(float(value)) else None)
                for key, value in metrics.items()
            },
        }
        self._stream.write(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        )
        self._stream.maybe_flush()
        if self._writer is not None:
            for key, value in metrics.items():
                if math.isfinite(float(value)):
                    self._writer.add_scalar(key, float(value), step)


    def flush(self) -> None:
        self._stream.flush()
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        try:
            if not self._stream.closed:
                self._stream.close()
        finally:
            if self._writer is not None:
                self._writer.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
