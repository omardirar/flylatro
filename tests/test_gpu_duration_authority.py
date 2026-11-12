import importlib
import json
from pathlib import Path

import pytest


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_lr_pilots")


def test_duration_limit_preserved_without_new_authority(tmp_path, monkeypatch):
    helper = runner(monkeypatch)
    ledger = {"limit_seconds": 21600, "entries": [{"seconds": 100}, {"seconds": 200}]}
    assert helper.command_timeout(ledger, tmp_path / "absent.json") == 21300


def test_explicit_unlimited_local_authority_removes_even_exhausted_limit(tmp_path, monkeypatch):
    helper = runner(monkeypatch)
    authority = tmp_path / "authorization.json"
    authority.write_text(json.dumps({"version": "gpu-duration-authorization-v1",
        "scope": "existing-local-hardware", "maximum_gpu_seconds": None,
        "authorized_at": "2026-10-09", "user_instruction": "synthetic test fixture: unlimited local GPU duration"}))
    ledger = {"limit_seconds": 21600, "entries": [{"seconds": 50000}]}
    assert helper.command_timeout(ledger, authority) is None
    payload = json.loads(authority.read_text()); payload.pop("scope")
    authority.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid GPU duration"):
        helper.command_timeout(ledger, authority)


def test_unlimited_time_keeps_live_job_and_duplicate_launch_protection(tmp_path, monkeypatch):
    helper = runner(monkeypatch)
    monkeypatch.setattr(helper, "OUT", tmp_path)
    path = tmp_path / "gpu-budget.json"
    path.write_text(json.dumps({"limit_seconds": 21600, "entries": [{"label": "live", "pid": 123}]}))
    with pytest.raises(RuntimeError, match="running/unresolved"):
        helper.gpu_command(["unused"], "new")
    path.write_text(json.dumps({"limit_seconds": 21600, "entries": [{"label": "old", "seconds": 1}]}))
    monkeypatch.setattr(helper, "command_timeout", lambda ledger: None)
    with pytest.raises(FileExistsError, match="already recorded"):
        helper.gpu_command(["unused"], "old")
