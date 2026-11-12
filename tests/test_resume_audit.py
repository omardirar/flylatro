from pathlib import Path

import numpy as np
import pytest

from flylatro.analysis.resume_cli import audit_resume, state_digest
from flylatro.learning.config import PlasticExperimentConfig


def test_resume_audit_compares_future_full_trainer_state(tmp_path: Path) -> None:
    config = PlasticExperimentConfig.load(Path("configs/plastic-smoke.toml"))
    report = audit_resume(config, tmp_path / "probe.pkl", prefix=4, future=4)
    assert report["gates"]["status"] == "PASS"
    assert report["saved_trainer_state_sha256"] == report["restored_trainer_state_sha256"]
    assert report["continuous"] == report["resumed"]
    assert len(report["continuous"]) == 4
    with pytest.raises(FileExistsError):
        audit_resume(config, tmp_path / "probe.pkl", prefix=4, future=4)


def test_resume_digest_tracks_contents_without_array_stride_or_dict_order() -> None:
    values = np.arange(12).reshape(3, 4)
    assert state_digest({"x": values, "y": b"state"}) == state_digest({"y": b"state", "x": values.copy(order="F")})
    changed = values.copy(); changed[0, 0] += 1
    assert state_digest(values) != state_digest(changed)


def test_resume_digest_hashes_deferred_neural_values_like_eager_arrays() -> None:
    torch = pytest.importorskip("torch")
    from flylatro.fly.plastic_backend import DeferredHostReadout

    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    deferred = DeferredHostReadout(torch.tensor(values))
    assert deferred._host is None
    assert state_digest({"kc_activity": deferred}) == state_digest({"kc_activity": values})
    assert deferred._host is not None
    changed = values.copy()
    changed[0, 0] += 1
    assert state_digest(deferred) != state_digest(changed)
