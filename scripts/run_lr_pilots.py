"""Execute predeclared LR pilots and retain GPU-time accounting and authority."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artefacts/lr-sensitivity-20261009"
GPU_AUTHORITY = OUT / "gpu-duration-authorization.json"


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def command_timeout(ledger, authority_path=None):
    """Keep the historical budget unless explicit later authority supersedes it."""
    path = GPU_AUTHORITY if authority_path is None else Path(authority_path)
    if path.exists():
        authority = json.loads(path.read_text())
        if (authority.get("version") != "gpu-duration-authorization-v1"
                or authority.get("scope") != "existing-local-hardware"
                or authority.get("maximum_gpu_seconds", "missing") is not None
                or not authority.get("user_instruction") or not authority.get("authorized_at")):
            raise ValueError("invalid GPU duration authorization")
        return None
    return ledger["limit_seconds"] - sum(row["seconds"] for row in ledger["entries"])


def gpu_command(command, label, *, allowed=(0,)):
    ledger_path = OUT / "gpu-budget.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    else:
        reconstruction = json.loads((OUT / "fixed-trajectory-comparison.json").read_text())
        ledger = {"limit_seconds": 21600, "accounting": "conservative host wall time of GPU commands",
                  "entries": [{"label": "fixed-trajectory-reconstruction",
                               "seconds": reconstruction["gpu_wall_seconds"], "exit_code": 0}]}
    if any("seconds" not in row for row in ledger["entries"]):
        raise RuntimeError("ledger has a running/unresolved command; inspect its process before continuing")
    used = sum(row["seconds"] for row in ledger["entries"])
    remaining = command_timeout(ledger)
    if remaining is not None and remaining < 60:
        raise RuntimeError("GPU budget exhausted; additional authorization required")
    if any(row["label"] == label for row in ledger["entries"]):
        raise FileExistsError(f"command already recorded: {label}")
    log = OUT / (label + ".log")
    entry = {"label": label, "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
             "log": str(log), "pid": None, "duration_timeout_seconds": remaining}
    if GPU_AUTHORITY.exists():
        import hashlib
        entry["duration_authorization"] = str(GPU_AUTHORITY)
        entry["duration_authorization_sha256"] = hashlib.sha256(GPU_AUTHORITY.read_bytes()).hexdigest()
    ledger["entries"].append(entry)
    started = time.monotonic()
    print(json.dumps({"phase": label, "status": "starting", "gpu_seconds_used": used}), flush=True)
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        entry["pid"] = process.pid
        write(ledger_path, ledger)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    entry.update(seconds=time.monotonic() - started, exit_code=code,
                 finished_at=datetime.now(timezone.utc).isoformat())
    write(ledger_path, ledger)
    print(json.dumps({"phase": label, "status": "finished", "exit_code": code, "seconds": entry["seconds"]}), flush=True)
    if code not in allowed:
        raise RuntimeError(f"{label} failed with exit {code}; preserved log {log}")
    return code


def module(name, *args):
    return [sys.executable, "-m", name, *map(str, args)]


def main():
    status_path = OUT / "pilot-execution-independent-v3.json"
    if status_path.exists():
        raise FileExistsError("pilot execution already exists; inspect its live handle or terminal evidence")
    plan = json.loads((OUT / "predeclaration.json").read_text())
    report = {"predeclaration": str(OUT / "predeclaration.json"), "status": "running", "rates": {}}
    write(status_path, report)
    try:
        for rate in plan["pilots"]["rates"]:
            tag = str(rate).replace(".", "p")
            config = ROOT / plan["rules"][str(rate)]["config"]
            row = {"config": str(config), "status": "calibrating"}
            report["rates"][str(rate)] = row
            write(status_path, report)
            for mode in ("controlled", "integrated"):
                output = OUT / f"plasticity-{tag}-{mode}.json"
                if mode == "controlled" and output.exists():
                    stress = json.loads(output.read_text())
                    row["controlled_stress_report"] = str(output)
                    # Controlled inputs deliberately stimulate every edge;
                    # keep the unchanged global-modification FAIL as evidence.
                    # Initial real readiness has always used integrated mode.
                    if stress["gates"]["failed"] != ["not_runaway_global_change"] or stress["diagnostics"]["modified_fraction"] != 1.0:
                        raise RuntimeError("unexpected controlled stress failure")
                    continue
                if output.exists():
                    calibration = json.loads(output.read_text())
                    assert calibration["gates"]["status"] == "PASS"
                    assert calibration["evidence_identity"]["plasticity_rule_sha256"] == plan["rules"][str(rate)]["rule_sha256"]
                    row["integrated_report"] = str(output)
                    continue
                code = gpu_command(module("flylatro.analysis.calibration_cli", "--config", config,
                    "--mode", mode, "--decisions", 20, "--output", output, "--heavy"),
                    f"calibration-{tag}-{mode}", allowed=(0, 2))
                if code and mode == "integrated":
                    row["status"] = "calibration_failed"
                    break
            if row["status"] == "calibration_failed":
                write(status_path, report)
                continue
            specificity = OUT / f"specificity-{tag}.json"
            if not specificity.exists():
                gpu_command(module("flylatro.analysis.specificity_cli", "--config", config,
                    "--decisions", 20, "--output", specificity, "--heavy"), f"specificity-{tag}")
            assert json.loads(specificity.read_text())["evidence_identity"]["plasticity_rule_sha256"] == plan["rules"][str(rate)]["rule_sha256"]
            # Readiness allows scientific measurements to be reused only for
            # components they actually bind. New LR-dependent reports are fresh.
            preflight = OUT / f"preflight-{tag}-initial.json"
            if not preflight.exists():
                gpu_command(module("flylatro.analysis.preflight_cli", "--config", config, "--profile", "initial",
                    "--plasticity-report", OUT / f"plasticity-{tag}-integrated.json",
                    "--specificity-report", specificity,
                    "--output", preflight), f"preflight-{tag}", allowed=(0, 1))
            readiness = json.loads(preflight.read_text())
            assert not readiness["failed"]
            assert readiness["expected_identity"]["plasticity_rule_sha256"] == plan["rules"][str(rate)]["rule_sha256"]
            run_dir = ROOT / f"runs/lr-{tag}-independent-pilot-20261009"
            row.update(status="training", run_dir=str(run_dir))
            write(status_path, report)
            code = gpu_command(module("flylatro.learning.cli", "--config", config, "--run-dir", run_dir,
                "--heavy", "--no-tensorboard"), f"training-{tag}-250", allowed=(0, 2))
            if code:
                row["status"] = "safety_stopped"
                write(status_path, report)
                continue
            checkpoint = run_dir / "plastic-checkpoint-final.pkl"
            code = gpu_command(module("flylatro.analysis.representation_cli", "--config", config,
                "--checkpoint", checkpoint, "--stage", "post", "--repeats", 3, "--batch-size", 8,
                "--output", OUT / f"representation-{tag}-250.json",
                "--reachability-output", OUT / f"reachability-{tag}-250.json", "--heavy"),
                f"checkpoint-health-{tag}-250", allowed=(0, 2))
            row["status"] = "health_failed" if code else "passed"
            summary = json.loads((run_dir / "run-summary.json").read_text())
            if row["status"] == "passed" and summary["plasticity_events"] < plan["selection"]["minimum_update_events"]:
                row["status"] = "no_updates"
            row["summary"] = summary
            write(status_path, report)
        passed = [rate for rate, row in report["rates"].items() if row["status"] == "passed"]
        selected = str(plan["selection"]["prefer_rate"]) if str(plan["selection"]["prefer_rate"]) in passed else (
            str(plan["selection"]["fallback_rate"]) if str(plan["selection"]["fallback_rate"]) in passed else None)
        report.update(status="complete" if selected else "no_stable_candidate", selected_rate=selected)
        write(status_path, report)
        if selected:
            tag = selected.replace(".", "p")
            config = ROOT / plan["rules"][selected]["config"]
            source = Path(report["rates"][selected]["run_dir"])
            control = ROOT / f"runs/lr-{tag}-matched-pilot-control-20261009"
            gpu_command(module("flylatro.learning.cli", "--config", config, "--run-dir", control,
                "--condition", "no_plasticity", "--action-schedule", source / "training-actions.jsonl",
                "--heavy", "--no-tensorboard"), f"pilot-control-{tag}-250")
            gpu_command(module("flylatro.analysis.control_validation_cli", "--manifest", source / "run-manifest.json",
                "--manifest", control / "run-manifest.json", "--output", OUT / "pilot-control-validation.json"),
                f"pilot-control-validation-{tag}")
            report.update(status="complete", selected_control_dir=str(control))
            write(status_path, report)
    except Exception as error:
        report.update(status="halted", error=str(error))
        write(status_path, report)
        raise


if __name__ == "__main__":
    main()
