"""Fresh reward-free calibration for the source-complete v783 correction.

Runs after the existing LR pilot/control job ends, sharing its six-hour ledger.
Unit-activity reuse is SHA-verified; learned checkpoints never reuse this cache.
"""
from dataclasses import asdict, replace
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml
from run_lr_pilots import OUT, ROOT, gpu_command, module, write


def main():
    destination = OUT / "source-complete-calibration-execution.json"
    if destination.exists():
        raise FileExistsError(destination)
    status = {"version": "source-complete-calibration-v1", "status": "waiting_for_existing_pilots",
              "original_predeclaration_sha256": sha256_file(OUT / "predeclaration.json"),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_scope": "PLAN.MD all anatomical pairs at minimum count 1; no learning-rule architecture change"}
    write(destination, status)
    while True:
        pilots = json.loads((OUT / "pilot-execution-independent-v3.json").read_text())
        ledger = json.loads((OUT / "gpu-budget.json").read_text())
        active = [row for row in ledger["entries"] if "seconds" not in row]
        for row in active:
            # Authoritative process verification, not a stale lock/status file.
            os.kill(row["pid"], 0)
        if pilots["status"] == "halted":
            raise RuntimeError(f"existing pilot executor halted: {pilots.get('error')}")
        if pilots.get("selected_control_dir") and not active:
            break
        if pilots["status"] == "no_stable_candidate":
            raise RuntimeError("existing candidates failed; do not select a new LR by performance")
        time.sleep(10)
    status["status"] = "running"
    write(destination, status)
    config = ROOT / "configs/plastic-v783-source-complete-calibration.toml"
    cache = OUT / "source-complete-unit-activity.npz"
    common = ["--config", config, "--repeats", 3, "--batch-size", 8, "--activity-cache", cache, "--heavy"]
    try:
        gpu_command(module("flylatro.analysis.sensory_cli", "--artifact",
            ROOT / "data/flywire-source-complete-v783/flywire_fafb_v783.npz",
            "--mapping-seed", 0, "--population-width", 3, "--max-rate-hz", 150,
            "--calibration-corpus", ROOT / "artefacts/calibration-corpus-v783-balanced-long.npz",
            "--output", OUT / "sensory-map-source-complete.json", "--health-report",
            OUT / "sensory-health-source-complete.json", "--full"), "source-complete-sensory")
        code = gpu_command(module("flylatro.analysis.representation_cli", *common,
            "--stage", "pre", "--output", OUT / "representation-pre-source-complete.json",
            "--reachability-output", OUT / "reachability-pre-source-complete.json"),
            "source-complete-representation-pre", allowed=(0, 2))
        status["representation_pre_exit"] = code
        write(destination, status)
        # A failed pre report remains FAIL; measuring motor diversity is still
        # useful diagnosis, but cannot authorize training over that failure.
        code = gpu_command(module("flylatro.analysis.motor_calibration_cli", *common,
            "--calibration-seed", 12000000, "--output", OUT / "motor-map-source-complete.json",
            "--report", OUT / "motor-calibration-source-complete.json"),
            "source-complete-motor-calibration", allowed=(0, 2))
        status["motor_exit"] = code
        write(destination, status)
        if code:
            status["status"] = "calibration_failed"
            write(destination, status)
            return
        c = PlasticExperimentConfig.load(config)
        c = replace(c, name="source-complete-v783-lr0p002", fly=replace(c.fly,
            motor_mapping_path=str(OUT / "motor-map-source-complete.json")),
            calibration=replace(c.calibration,
                sensory_health_report=str(OUT / "sensory-health-source-complete.json"),
                representation_pre_report=str(OUT / "representation-pre-source-complete.json"),
                representation_post_report=str(OUT / "representation-post-source-complete.json"),
                reachability_report=str(OUT / "reachability-source-complete.json"),
                motor_calibration_report=str(OUT / "motor-calibration-source-complete.json"),
                plasticity_report=str(OUT / "plasticity-source-complete-integrated.json"),
                specificity_report=str(OUT / "specificity-source-complete.json"),
                benchmark_report=str(OUT / "benchmark-source-complete.json")))
        document = asdict(c); document.pop("source_path")
        calibrated = ROOT / "configs/plastic-v783-source-complete.toml"
        with calibrated.open("x") as stream:
            stream.write(render_toml(document))
        common[1] = calibrated
        code = gpu_command(module("flylatro.analysis.representation_cli", *common,
            "--stage", "post", "--output", OUT / "representation-post-source-complete.json",
            "--reachability-output", OUT / "reachability-source-complete.json"),
            "source-complete-representation-post", allowed=(0, 2))
        status["representation_post_exit"] = code
        if code or status["representation_pre_exit"]:
            status["status"] = "calibration_failed"
            write(destination, status)
            return
        gpu_command(module("flylatro.analysis.calibration_cli", "--config", calibrated,
            "--mode", "integrated", "--decisions", 20, "--output",
            OUT / "plasticity-source-complete-integrated.json", "--heavy"), "source-complete-integrated-calibration")
        gpu_command(module("flylatro.analysis.specificity_cli", "--config", calibrated,
            "--decisions", 20, "--output", OUT / "specificity-source-complete.json", "--heavy"), "source-complete-specificity")
        # Benchmark this exact model before declaring its exposure feasible.
        from flylatro.benchmarks.cli import plastic_end_to_end_main
        gpu_command([module("flylatro.benchmarks.cli")[0], "-c",
            "from flylatro.benchmarks.cli import plastic_end_to_end_main; raise SystemExit(plastic_end_to_end_main())",
            "--config", str(calibrated), "--steps", "10", "--profile-components", "--output",
            str(OUT / "benchmark-source-complete.json"), "--heavy"], "source-complete-benchmark")
        gpu_command(module("flylatro.analysis.preflight_cli", "--config", calibrated,
            "--profile", "initial", "--output", OUT / "preflight-source-complete-initial.json"),
            "source-complete-preflight", allowed=(0, 1))
        status.update(status="calibrated", config=str(calibrated))
        write(destination, status)
    except Exception as error:
        status.update(status="halted", error=str(error))
        write(destination, status)
        raise


if __name__ == "__main__":
    main()
