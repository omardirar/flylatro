"""Run a fresh source-complete stability pilot, exact control and resume probe."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml
from run_lr_pilots import OUT, ROOT, gpu_command, module, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--readiness-execution", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    readiness = json.loads(args.readiness_execution.read_text())
    if readiness["status"] != "calibrated":
        raise ValueError("a terminal passing reward-free readiness execution is required")
    base_path = Path(readiness["selected_config"])
    base = PlasticExperimentConfig.load(base_path)
    tag = args.tag
    destination = OUT / f"{tag}-pilot-execution.json"
    declaration = OUT / f"{tag}-pilot-predeclaration.json"
    config_path = ROOT / f"configs/plastic-v783-{tag}-pilot.toml"
    for path in (destination, declaration, config_path):
        if path.exists():
            raise FileExistsError(path)
    config = replace(base, name=tag + "-250-pilot", training=replace(base.training,
        max_environment_decisions=250, checkpoint_every_decisions=50,
        base_fly_seed=1971516477, training_seed_offset=3000,
        budget_basis="source-complete readiness benchmark; unchanged predeclared 250-decision stability pilot"),
        motor=replace(base.motor, exploration_seed=757137546))
    payload = asdict(config); payload.pop("source_path")
    with config_path.open("x") as stream:
        stream.write(render_toml(payload))
    primary = ROOT / f"runs/{tag}-independent-pilot-20261009"
    control = ROOT / f"runs/{tag}-matched-pilot-control-20261009"
    plan = {"version": "source-complete-stability-pilot-predeclaration-v1",
        "readiness_execution_sha256": sha256_file(args.readiness_execution),
        "selected_readiness_config": str(base_path), "config": str(config_path),
        "config_sha256": sha256_file(config_path), "rate": 0.002,
        "selection_basis": "Historical same-trajectory and independent-action pilots predeclared lower stable rate preference; corrected source/dynamics now require independent stability and full learned-checkpoint audits again.",
        "decisions": 250, "primary": str(primary), "exact_control": str(control),
        "health": {"states": 96, "repeats": 3, "batch_size": 8},
        "all_original_gates_required": True, "minimum_changed_events": 1,
        "performance_not_used_for_selection": True, "next_exposure": "three independently seeded 1000-decision primary/control pairs, staged 100/250/500/1000", "gpu_limit_seconds": 21600}
    with declaration.open("x") as stream:
        json.dump(plan, stream, indent=2); stream.write("\n")
    result = {"status": "validating", "config": str(config_path), "primary": str(primary),
        "control": str(control), "predeclaration_sha256": sha256_file(declaration)}
    write(destination, result)
    try:
        preflight = OUT / f"{tag}-pilot-preflight.json"
        gpu_command(module("flylatro.analysis.preflight_cli", "--config", config_path,
            "--profile", "initial", "--output", preflight), f"{tag}-pilot-preflight", allowed=(0, 1))
        if json.loads(preflight.read_text())["failed"]:
            raise ValueError("initial readiness contains FAILs")
        result["status"] = "training"; write(destination, result)
        code = gpu_command(module("flylatro.learning.cli", "--config", config_path,
            "--run-dir", primary, "--heavy", "--no-tensorboard"), f"{tag}-pilot-training", allowed=(0, 2))
        if code:
            result["status"] = "safety_stopped"; write(destination, result); return
        checkpoint = primary / "plastic-checkpoint-final.pkl"
        representation = OUT / f"{tag}-pilot-representation.json"
        reachability = OUT / f"{tag}-pilot-reachability.json"
        result["status"] = "checkpoint_health"; write(destination, result)
        code = gpu_command(module("flylatro.analysis.representation_cli", "--config", config_path,
            "--checkpoint", checkpoint, "--stage", "post", "--repeats", 3, "--batch-size", 8,
            "--output", representation, "--reachability-output", reachability, "--heavy"),
            f"{tag}-pilot-checkpoint-health", allowed=(0, 2))
        if code:
            result["status"] = "health_failed"; write(destination, result); return
        summary = json.loads((primary / "run-summary.json").read_text())
        if summary["plasticity_events"] < 1:
            raise ValueError("pilot has no learned update events")
        result.update(status="matched_control", summary=summary); write(destination, result)
        gpu_command(module("flylatro.learning.cli", "--config", config_path, "--run-dir", control,
            "--condition", "no_plasticity", "--action-schedule", primary / "training-actions.jsonl",
            "--heavy", "--no-tensorboard"), f"{tag}-pilot-matched-control")
        validation = OUT / f"{tag}-pilot-control-validation.json"
        gpu_command(module("flylatro.analysis.control_validation_cli", "--manifest", primary / "run-manifest.json",
            "--manifest", control / "run-manifest.json", "--output", validation), f"{tag}-pilot-control-validation")
        result["status"] = "resume_audit"; write(destination, result)
        resume = OUT / f"{tag}-resume-audit.json"
        gpu_command(module("flylatro.analysis.resume_cli", "--config", config_path,
            "--checkpoint", OUT / f"{tag}-resume-probe.pkl", "--prefix-decisions", 8,
            "--future-decisions", 8, "--output", resume, "--heavy"), f"{tag}-resume-audit")
        result.update(status="passed", checkpoint=str(checkpoint), representation=str(representation),
            reachability=str(reachability), control_validation=str(validation), resume_audit=str(resume))
        write(destination, result)
    except Exception as error:
        result.update(status="halted", error=str(error)); write(destination, result)
        raise


if __name__ == "__main__":
    main()
