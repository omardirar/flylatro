"""Lock completed primary checkpoints and run the predeclared frozen final test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from flylatro.evaluation.compare import compare_frozen_evaluations
from flylatro.evaluation.batch_validation import qualified_batch_size
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.seeds import SeedPlan
from run_lr_pilots import OUT, ROOT, gpu_command, module, write


def lock_completed_experiment(driver_path: Path, tag: str, batch_validation: Path | None = None) -> dict:
    driver = json.loads(driver_path.read_text())
    if driver["status"] != "passed":
        raise ValueError("all primary/control pairs must complete before final-test lock")
    execution_path = Path(driver["execution"])
    execution = json.loads(execution_path.read_text())
    if execution["status"] != "completed" or len(execution["control_validations"]) != 3:
        raise ValueError("three completed exact control validations are required")
    if any(row["status"] != "PASS" for row in execution["control_validations"]):
        raise ValueError("an exact control validation failed")
    primary = [row for row in execution["arms"] if row["condition"] == "plastic_real"]
    controls = [row for row in execution["arms"] if row["condition"] == "no_plasticity"]
    if len(primary) != 3 or len(controls) != 3:
        raise ValueError("exactly three primary and three control arms required")
    checkpoints = [Path(row["commands"][-1]["checkpoint"]) for row in (*primary, *controls)]
    manifests = [load_plastic_manifest(path) for path in checkpoints]
    if any(item["environment_decisions"] != 1000 for item in manifests):
        raise ValueError("every checkpoint must complete its declared 1000 decisions")
    validation = [json.loads((Path(row["evaluation_dir"]) / "summary.json").read_text()) for row in (*primary, *controls)]
    model_identities = []
    for index in range(3):
        paired = compare_frozen_evaluations(validation[index], validation[index + 3])
        model_identities.append(paired["model_identity"])
        if paired["seed_stream"] != "validation" or len(paired["seeds"]) != 32:
            raise ValueError("every pair requires its full independent-action validation evaluation")
    if any(identity != model_identities[0] for identity in model_identities[1:]):
        raise ValueError("all three flies must share the exact fixed model")
    unit_hash = hashlib.sha256(np.ones(manifests[3]["plastic_parameter_count"], dtype="<f4").tobytes()).hexdigest()
    control_transition_hashes = [sha256_file(Path(row["evaluation_dir"]) / "transitions.jsonl") for row in controls]
    common_baseline = all(item["plastic_weight_sha256"] == unit_hash for item in manifests[3:])
    common_baseline &= all(item["episodes"] == validation[3]["episodes"] for item in validation[3:])
    common_baseline &= len(set(control_transition_hashes)) == 1
    for control in controls:
        summary = json.loads((Path(control["run_dir"]) / "run-summary.json").read_text())
        common_baseline &= summary["status"] == "completed" and summary["plasticity_events"] == 0
    pairing_path = OUT / "confirmatory-pairing-correction.json"
    pairing = json.loads(pairing_path.read_text())
    pairs = []
    for index in range(3):
        control_index = 0 if common_baseline else index
        row = {"replicate": index, "original_exact_control_checkpoint": str(checkpoints[index + 3])}
        for role, arm, manifest, checkpoint in (
            ("trained", primary[index], manifests[index], checkpoints[index]),
            ("control", controls[control_index], manifests[control_index + 3], checkpoints[control_index + 3])):
            config_path = Path(arm["config"])
            row.update({role + "_checkpoint": str(checkpoint),
                role + "_checkpoint_sha256": manifest["checkpoint_sha256"],
                role + "_weight_sha256": manifest["plastic_weight_sha256"],
                role + "_config": str(config_path), role + "_config_file_sha256": sha256_file(config_path),
                role + "_config_sha256": PlasticExperimentConfig.load(config_path).sha256,
                role + "_summary": str(ROOT / "evaluations" / (tag + "-confirmatory-20261009") /
                    arm["arm_id"].replace(":", "-") / "summary.json")})
        pairs.append(row)
    lock = {"version": "confirmatory-frozen-ante1-lock-v1", "locked_at": datetime.now(timezone.utc).isoformat(),
        "driver_sha256": sha256_file(driver_path), "execution_sha256": sha256_file(execution_path),
        "pairs": pairs, "plasticity_rule_sha256": manifests[0]["plasticity_rule_sha256"],
        "model_identity": model_identities[0],
        "evaluation_batch_size": 1,
        "evaluation_implementation_sha256": {str(ROOT / "src/flylatro" / path): sha256_file(ROOT / "src/flylatro" / path)
            for path in ("evaluation/plastic_cli.py", "evaluation/plastic.py", "fly/plastic_backend.py", "interface/motor.py", "env/balatro_sim.py")},
        "analysis_module_sha256": sha256_file(ROOT / "src/flylatro/evaluation/confirmatory.py"),
        "pairing_predeclaration": str(pairing_path), "pairing_predeclaration_sha256": sha256_file(pairing_path),
        "primary_fly_indices": pairing["primary_fly_indices"], "final_seeds": list(SeedPlan().seeds("final_test", 128)),
        "common_untrained_baseline": bool(common_baseline), "baseline_equivalence": {
            "unit_efficacy_sha256": unit_hash, "validation_transition_sha256": control_transition_hashes,
            "independent_validation_agrees": bool(common_baseline),
            "scope": "All three exact training controls remain independent and required. Only identical frozen unit-weight performance evaluation can be reused, as prospectively declared."},
        "validation_summary_sha256": {str(Path(arm["evaluation_dir"]) / "summary.json"):
            sha256_file(Path(arm["evaluation_dir"]) / "summary.json") for arm in (*primary, *controls)}}
    if batch_validation is not None:
        lock["evaluation_batch_size"] = qualified_batch_size(batch_validation, pairs)
        lock.update(batch_validation=str(batch_validation), batch_validation_sha256=sha256_file(batch_validation))
    return lock


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--primary-driver", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batch-validation", type=Path,
        help="complete exact validation qualification for the frozen execution batch size")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    lock_path = OUT / f"{args.tag}-confirmatory-lock.json"
    destination = OUT / f"{args.tag}-confirmatory-{'execution' if args.execute else 'plan'}.json"
    if destination.exists() or (lock_path.exists() and not args.execute):
        raise FileExistsError("confirmatory declaration already exists; inspect it and its live/terminal execution")
    lock = lock_completed_experiment(args.primary_driver, args.tag, args.batch_validation)
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        lock["locked_at"] = previous["locked_at"]
        if lock != previous:
            raise ValueError("experiment differs from its existing pre-evaluation lock")
    else:
        with lock_path.open("x") as stream:
            json.dump(lock, stream, indent=2); stream.write("\n")
    commands = []
    seen = set()
    for pair in lock["pairs"]:
        for role, condition in (("trained", "plastic_real"), ("control", "no_plasticity")):
            if pair[role + "_summary"] in seen:
                continue
            seen.add(pair[role + "_summary"])
            commands.append(module("flylatro.evaluation.plastic_cli", "--config", pair[role + "_config"],
                "--checkpoint", pair[role + "_checkpoint"], "--condition", condition,
                "--win-ante", 1, "--episodes", 128, "--seed-stream", "final_test",
                "--unlock-final-test", "--batch-size", lock["evaluation_batch_size"],
                "--output-dir", Path(pair[role + "_summary"]).parent, "--progress", "--heavy"))
    report = {"status": "running" if args.execute else "locked", "lock": str(lock_path),
        "lock_sha256": sha256_file(lock_path), "commands": commands}
    write(destination, report)
    if not args.execute:
        print(json.dumps(report)); return
    try:
        for index, command in enumerate(commands):
            gpu_command(command, f"{args.tag}-confirmatory-{index}")
        analysis = module("flylatro.evaluation.confirmatory", "--lock", lock_path,
            "--output", OUT / f"{args.tag}-confirmatory-analysis.json")
        for pair in lock["pairs"]:
            analysis += ["--trained", pair["trained_summary"], "--control", pair["control_summary"]]
        gpu_command(analysis, f"{args.tag}-confirmatory-analysis")
        report.update(status="completed"); write(destination, report)
    except Exception as error:
        report.update(status="halted", error=str(error)); write(destination, report)
        raise


if __name__ == "__main__":
    main()
