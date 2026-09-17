from __future__ import annotations

import json
from pathlib import Path

from flylatro.telemetry.metrics import MetricLogger


def test_structured_metrics_are_jsonl_serializable(tmp_path: Path) -> None:
    with MetricLogger(tmp_path, run_id="run-1", tensorboard=False) as logger:
        logger.log(
            8, {"train/policy_loss": 0.25, "episodes": 2, "undefined": float("nan")}
        )

    payload = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert payload["run_id"] == "run-1"
    assert payload["step"] == 8
    assert payload["metrics"]["train/policy_loss"] == 0.25
    assert payload["metrics"]["undefined"] is None
