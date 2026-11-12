"""Qualify frozen batches on complete validation trajectories before final seeds."""
import argparse
import json
from pathlib import Path

from flylatro.evaluation.batch_validation import compare_batch_trajectories
from flylatro.fly.flywire_artifact import sha256_file
from run_confirmatory_evaluation import lock_completed_experiment
from run_lr_pilots import OUT, ROOT, gpu_command, module, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--primary-driver", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    destination = OUT / f"{args.tag}-batch-validation.json"
    if destination.exists():
        raise FileExistsError(destination)
    lock = lock_completed_experiment(args.primary_driver, args.tag)
    driver = json.loads(args.primary_driver.read_text())
    execution = json.loads(Path(driver["execution"]).read_text())
    arms = {str(Path(row["commands"][-1]["checkpoint"])): row for row in execution["arms"]}
    declaration = OUT / "frozen-batch-execution-predeclaration.json"
    width = json.loads(declaration.read_text())["execution_batch_size"]
    implementation = [ROOT / "src/flylatro" / path for path in (
        "evaluation/plastic_cli.py", "evaluation/plastic.py", "evaluation/batch_validation.py",
        "fly/plastic_backend.py", "interface/motor.py", "env/balatro_sim.py")]
    report = {"version": "frozen-batch-qualification-v1", "status": "running", "batch_size": width,
        "predeclaration": str(declaration), "predeclaration_sha256": sha256_file(declaration),
        "primary_driver_sha256": sha256_file(args.primary_driver),
        "implementation_sha256": {str(path): sha256_file(path) for path in implementation}, "comparisons": []}
    write(destination, report)
    seen = set()
    try:
        for pair in lock["pairs"]:
            for role, condition in (("trained", "plastic_real"), ("control", "no_plasticity")):
                checkpoint = pair[role + "_checkpoint"]
                if checkpoint in seen:
                    continue
                seen.add(checkpoint)
                arm = arms[checkpoint]
                output = OUT / f"{args.tag}-batch-validation" / arm["arm_id"].replace(":", "-")
                command = module("flylatro.evaluation.plastic_cli", "--config", pair[role + "_config"],
                    "--checkpoint", checkpoint, "--condition", condition, "--win-ante", 1,
                    "--episodes", 32, "--seed-stream", "validation", "--batch-size", width,
                    "--output-dir", output, "--progress", "--heavy")
                gpu_command(command, args.tag + "-batch-validation-" + arm["arm_id"].replace(":", "-"))
                report["comparisons"].append(compare_batch_trajectories(Path(arm["evaluation_dir"]), output, batch_size=width))
                write(destination, report)
        report.update(status="passed"); write(destination, report)
    except Exception as error:
        report.update(status="halted", error=str(error)); write(destination, report)
        raise
    print(json.dumps({"status": report["status"], "qualified_checkpoints": len(seen), "batch_size": width}))


if __name__ == "__main__":
    main()
