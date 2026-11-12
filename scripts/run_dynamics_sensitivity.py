"""Execute the named fixed-gain/window readiness sensitivity, within the shared budget."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json

from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml
from run_lr_pilots import OUT, ROOT, gpu_command, module, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, default=OUT / "fixed-dynamics-predeclaration.json")
    parser.add_argument("--output", type=Path, default=OUT / "fixed-dynamics-execution.json")
    args = parser.parse_args()
    destination = args.output
    if destination.exists():
        raise FileExistsError(destination)
    plan = json.loads(args.plan.read_text())
    result = {"status": "running", "variants": [], "selected_config": None}
    write(destination, result)
    try:
        for variant in plan["configurations"]:
            duration = variant["duration_ms"]
            tag = variant.get("tag", f"source-complete-gain0p18-{duration}ms")
            config = ROOT / variant["config"]
            cache = ROOT / variant["activity_cache"] if "activity_cache" in variant else OUT / f"{tag}-unit-activity.npz"
            row = {"tag": tag, "status": "representation_pre", "config": str(config)}
            result["variants"].append(row)
            write(destination, result)
            common = ["--config", config, "--repeats", 3, "--batch-size", 8, "--activity-cache", cache, "--heavy"]
            pre = OUT / f"{tag}-representation-pre.json"
            pre_reach = OUT / f"{tag}-reachability-pre.json"
            code = gpu_command(module("flylatro.analysis.representation_cli", *common,
                "--stage", "pre", "--output", pre, "--reachability-output", pre_reach),
                f"{tag}-representation-pre", allowed=(0, 2))
            row["representation_pre_exit"] = code
            if code:
                row["status"] = "representation_failed"
                write(destination, result)
                continue
            motor = OUT / f"{tag}-motor.json"
            motor_report = OUT / f"{tag}-motor-calibration.json"
            code = gpu_command(module("flylatro.analysis.motor_calibration_cli", *common,
                "--calibration-seed", 12000000, "--output", motor, "--report", motor_report),
                f"{tag}-motor", allowed=(0, 2))
            if code:
                row["status"] = "motor_failed"
                write(destination, result)
                continue
            c = PlasticExperimentConfig.load(config)
            c = replace(c, name=tag + "-calibrated", fly=replace(c.fly, motor_mapping_path=str(motor)),
                calibration=replace(c.calibration,
                    sensory_health_report=str(OUT / "sensory-health-source-complete.json"),
                    representation_pre_report=str(pre), motor_calibration_report=str(motor_report),
                    representation_post_report=str(OUT / f"{tag}-representation-post.json"),
                    reachability_report=str(OUT / f"{tag}-reachability.json"),
                    plasticity_report=str(OUT / f"{tag}-plasticity-integrated.json"),
                    specificity_report=str(OUT / f"{tag}-specificity.json"),
                    benchmark_report=str(OUT / f"{tag}-benchmark.json")))
            calibrated = ROOT / f"configs/plastic-v783-{tag}-calibrated.toml"
            document = asdict(c); document.pop("source_path")
            with calibrated.open("x") as stream:
                stream.write(render_toml(document))
            common[1] = calibrated
            code = gpu_command(module("flylatro.analysis.representation_cli", *common,
                "--stage", "post", "--output", c.calibration.representation_post_report,
                "--reachability-output", c.calibration.reachability_report),
                f"{tag}-representation-post", allowed=(0, 2))
            if code:
                row["status"] = "post_motor_failed"
                write(destination, result)
                continue
            code = gpu_command(module("flylatro.analysis.calibration_cli", "--config", calibrated,
                "--mode", "integrated", "--decisions", 20, "--output", c.calibration.plasticity_report,
                "--heavy"), f"{tag}-integrated", allowed=(0, 2))
            if code:
                row["status"] = "integrated_failed"
                write(destination, result)
                continue
            gpu_command(module("flylatro.analysis.specificity_cli", "--config", calibrated,
                "--decisions", 20, "--output", c.calibration.specificity_report, "--heavy"), f"{tag}-specificity")
            gpu_command([*module("unused")[:1], "-c",
                "from flylatro.benchmarks.cli import plastic_end_to_end_main; raise SystemExit(plastic_end_to_end_main())",
                "--config", str(calibrated), "--steps", "10", "--profile-components", "--output",
                c.calibration.benchmark_report, "--heavy"], f"{tag}-benchmark")
            gpu_command(module("flylatro.analysis.preflight_cli", "--config", calibrated,
                "--profile", "initial", "--output", OUT / f"{tag}-preflight-initial.json"),
                f"{tag}-preflight-initial", allowed=(0, 1))
            row.update(status="passed", calibrated_config=str(calibrated))
            result.update(status="calibrated", selected_config=str(calibrated))
            write(destination, result)
            return
        result["status"] = "no_ready_candidate"
        write(destination, result)
    except Exception as error:
        result.update(status="halted", error=str(error))
        write(destination, result)
        raise


if __name__ == "__main__":
    main()
