"""Run the saved separated layout with a fixed budget and periodic frozen evals."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from flylatro.analysis.card_window_pilot import write_json
from flylatro.evaluation.compare import compare_frozen_evaluations
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from flylatro.interface.separated_motor import separation_summary
from flylatro.learning.choice_credit_pilot import implementation_hash
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.protocol_materialize import render_toml
from flylatro.seeds import SeedPlan


def prepare(source, output, *, steps=100000, interval=10000, episodes=256, batch=4):
    if min(steps, interval, episodes, batch) < 1 or steps % interval:
        raise ValueError("positive counts and a budget divisible by the evaluation interval required")
    base = PlasticExperimentConfig.load(source)
    mapping_path = resolve_path(base, base.fly.motor_mapping_path)
    mapping = MotorMapping.load(mapping_path)
    if (not separation_summary(mapping)["card_isolated_and_core_disjoint"]
            or base.environment.allow_skip_blinds or base.fly.duration_ms != 100
            or base.reinforcement.run_failure_pulse != .01
            or base.plasticity.version != "three-factor-choice-gated-v1"
            or base.protocol.bound or not base.training.plasticity_enabled):
        raise ValueError("source must be the qualified separated, choice-credit, 100 ms, no-skip, failure .01 experiment")
    width = base.environment.num_envs
    config = replace(base, name="separated-long-learning-curve-v1",
        fly=replace(base.fly, artifact_path=str(resolve_path(base, base.fly.artifact_path)), motor_mapping_path=str(mapping_path)),
        calibration=replace(base.calibration, corpus_path=str(resolve_path(base, base.calibration.corpus_path))),
        training=replace(base.training, max_environment_decisions=steps*width,
            checkpoint_every_decisions=interval*width,
            budget_basis="Exploratory delayed-learning test of qualified separated pools; fixed larger budget; periodic frozen evaluations"))
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output/"config.toml"
    document = asdict(config); document.pop("source_path")
    config_path.write_text(render_toml(document))
    declaration = {"version": "separated-long-learning-curve-v1", "source_config": str(source.resolve()),
        "source_config_sha256": sha256_file(source), "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path), "mapping": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path), "implementation_sha256": implementation_hash(),
        "launcher_sha256": sha256_file(Path(__file__)), "fresh_unit_weights": True,
        "replicates": width, "steps_per_learner": steps, "total_environment_decisions": steps*width,
        "training_seeds": list(SeedPlan().seeds("training", width, offset=config.training.training_seed_offset)),
        "stages_per_learner": list(range(interval, steps+1, interval)),
        "evaluation_episodes": episodes, "evaluation_batch_size": batch, "evaluation_seed_offset": 50000,
        "evaluation_seeds": list(SeedPlan().seeds("validation", episodes, offset=50000)),
        "initial_control": "one frozen unit-weight evaluation per mapping; same seeds as every learned checkpoint",
        "interpretation": "exploratory learning curve for three fixed training seeds, not a comparison against an equally extended baseline layout",
        "final_test_seeds_used": False}
    write_json(output/"declaration.json", declaration)
    return declaration


def execute(output, declaration):
    config = declaration["config"]
    status = {"status": "running", "completed_stages_per_learner": []}
    save = lambda: write_json(output/"status.json", status)
    save()
    def module(name, *args):
        subprocess.run([sys.executable, "-m", name, *map(str, args)], check=True)
    def evaluate(checkpoint, destination, learner, initial=False):
        module("flylatro.evaluation.plastic_cli", "--config", config, "--checkpoint", checkpoint,
            "--learner-index", learner, "--episodes", declaration["evaluation_episodes"],
            "--batch-size", declaration["evaluation_batch_size"], "--seed-stream", "validation",
            "--seed-offset", declaration["evaluation_seed_offset"], "--win-ante", 1,
            "--max-vector-steps", 1000, "--output-dir", destination, "--heavy",
            *(["--initial-weights"] if initial else []))
        return json.loads((destination/"summary.json").read_text())
    checkpoint = None
    try:
        for stage in declaration["stages_per_learner"]:
            for path, digest in ((config, declaration["config_sha256"]),
                                 (declaration["mapping"], declaration["mapping_sha256"])):
                if sha256_file(Path(path)) != digest:
                    raise ValueError("declared config or mapping changed")
            if implementation_hash() != declaration["implementation_sha256"]:
                raise ValueError("implementation changed during the experiment")
            status.update(stage_per_learner=stage, phase="training"); save()
            print(f"Training to {stage:,} decisions per learner", flush=True)
            module("flylatro.learning.cli", "--config", config, "--run-dir", output/"training",
                "--stop-after-decisions", stage*declaration["replicates"], "--dashboard", "--dashboard-refresh", 2,
                "--no-tensorboard", "--heavy", *(["--resume", checkpoint] if checkpoint else []))
            summary = json.loads((output/"training/run-summary.json").read_text())
            if (summary["status"] not in {"stage_complete", "completed"} or summary["safety_stop"] is not None
                    or summary["environment_decisions"] != stage*declaration["replicates"]):
                raise RuntimeError("training stopped before its declared stage; checkpoint retained")
            checkpoint = summary["checkpoint"]
            status.update(phase="evaluation", checkpoint=checkpoint); save()
            initial_path = output/"initial-evaluation"
            initial = (json.loads((initial_path/"summary.json").read_text()) if initial_path.exists()
                       else evaluate(checkpoint, initial_path, 0, initial=True))
            reports = []
            for learner in range(declaration["replicates"]):
                destination = output/f"evaluation-{stage:06d}-learner-{learner}"
                trained = evaluate(checkpoint, destination, learner)
                comparison = compare_frozen_evaluations(trained, initial)
                write_json(destination/"comparison-to-initial.json", comparison)
                reports.append({"learner": learner,
                    "blinds_beaten": sum(row["blinds_beaten"] for row in trained["episodes"]),
                    "initial_blinds_beaten": sum(row["blinds_beaten"] for row in initial["episodes"]),
                    "wins": sum(int(row["won"]) for row in trained["episodes"]),
                    "mean_return_delta": comparison["mean_return_delta"]})
            write_json(output/f"learning-curve-{stage:06d}.json", {"steps_per_learner": stage, "learners": reports})
            status["completed_stages_per_learner"].append(stage); save()
            print(json.dumps({"steps_per_learner": stage, "results": reports}), flush=True)
        status.update(status="completed", phase="completed")
    except BaseException as error:
        status.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("artefacts/separated-motor-20261024-pilot-v2/configs/separated.toml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps-per-learner", type=int, default=100000)
    parser.add_argument("--evaluate-every", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.heavy:
        parser.error("execution requires --heavy")
    output = (args.output_dir or Path("runs")/datetime.now(timezone.utc).strftime("separated-long-%Y%m%dT%H%M%SZ")).resolve()
    declaration = prepare(args.config, output, steps=args.steps_per_learner, interval=args.evaluate_every,
                          episodes=args.episodes, batch=args.batch_size)
    print(f"Experiment: {output}", flush=True)
    if not args.prepare_only:
        execute(output, declaration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
