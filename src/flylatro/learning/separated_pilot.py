"""Bounded matched pilot of card-isolated motor pools, with fresh learners."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from flylatro.analysis.card_selection import plastic_hashes
from flylatro.analysis.card_window_pilot import write_json
from flylatro.analysis.evidence import load_corpus
from flylatro.evaluation.compare import MATCHED_FIELDS, compare_frozen_evaluations
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from flylatro.interface.motor_contexts import motor_context_windows
from flylatro.interface.separated_motor import calibrate_separated, separation_summary
from flylatro.learning.choice_credit_pilot import implementation_hash
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack, resolve_path
from flylatro.learning.protocol_materialize import render_toml
from flylatro.seeds import SeedPlan, derive_seed
from flylatro.env.upstream_contract import UpstreamActionType

VERSION = "card-isolated-motor-matched-pilot-v1"
SCALE_RULE = "complete healthy equal budgets; separated beats baseline in blind clears in a majority of learners and in total; separated has positive learning gain over its own initial policy and a larger gain than baseline in a majority and in total; trained wins do not decrease"


def validate_pair(left, right):
    documents = []
    for config in (left, right):
        if (config.plasticity.version != "three-factor-choice-gated-v1" or config.environment.allow_skip_blinds
                or config.reinforcement.run_failure_pulse != .01 or config.fly.duration_ms != 100
                or config.protocol.bound or not config.training.plasticity_enabled):
            raise ValueError("pilot retains choice credit, 100 ms, no skips, failure .01 and enabled learning")
        document = asdict(config)
        for key in ("name", "source_path"):
            document.pop(key)
        document["fly"].pop("motor_mapping_path")
        documents.append(document)
    if documents[0] != documents[1]:
        raise ValueError("pilot configurations may differ only in motor mapping and name")


def prepare(base, output, *, steps=10000, replicates=3, episodes=256, batch=4):
    if min(steps, replicates, episodes, batch) < 1:
        raise ValueError("positive pilot sizes required")
    validate_pair(base, base)
    if base.fly.backend != "flywire" or base.environment.backend != "balatro_sim":
        raise ValueError("pilot requires real FlyWire and simulator")
    output.mkdir(parents=True, exist_ok=False)
    source_paths = {"config": base.source_path.resolve(),
                    "motor": resolve_path(base, base.fly.motor_mapping_path),
                    "artifact": resolve_path(base, base.fly.artifact_path),
                    "corpus": resolve_path(base, base.calibration.corpus_path)}
    declaration = {"version": VERSION, "implementation_sha256": implementation_hash(),
        "sources": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in source_paths.items()},
        "steps_per_learner": steps, "replicates": replicates, "evaluation_episodes": episodes,
        "evaluation_batch_size": batch, "evaluation_seed_offset": 30000,
        "training_seeds": list(SeedPlan().seeds("training", replicates, offset=4000)),
        "evaluation_seeds": list(SeedPlan().seeds("validation", episodes, offset=30000)),
        "calibration_repeats": 4, "calibration_seed": 27_200_000,
        "calibration": "fresh unit weights; same reward-free corpus/responses/median-IQR normalization for both arms; masks reflect disabled blind skipping; baseline memberships retained; jointly reassign active separated card/action/count groups; inactive SKIP pool uses quiet unused neurons; card neurons absent from other groups; original context-slot routing retained with noncard sharing; unchanged quality gates; 32 covariance-only starts, four refinement rounds and at most 3000 annealing proposals per start",
        "controls": "fresh weights and zero exposure in both arms; matched independent learners and seeds; frozen evaluation of each trained learner and one identical unit-weight control per arm; no replayed actions",
        "scale_up_rule": SCALE_RULE, "final_test_seeds_used": False,
        "interpretation": "exploratory engineering pilot; three training replicates; mapping selection uses no reward or identity labels; new validation episodes; no confirmatory significance claim"}
    write_json(output / "declaration.json", declaration)
    return declaration


def check_sources(declaration):
    if implementation_hash() != declaration["implementation_sha256"]:
        raise ValueError("pilot implementation changed during execution")
    for row in declaration["sources"].values():
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("pilot source file changed")


def calibrate(base, output, declaration):
    check_sources(declaration)
    corpus = load_corpus(base)
    if set(corpus.environment_seeds) & set(declaration["evaluation_seeds"]):
        raise ValueError("evaluation seeds overlap calibration roots")
    stack = build_plastic_stack(base)
    before = plastic_hashes(stack.agent.plasticity.state)
    if not np.all(stack.agent.plasticity.state.state_dict()["efficacy"] == 1):
        raise ValueError("calibration must use fresh unit weights")
    outputs, indices, seeds = [], [], []
    exact = 0
    for repeat in range(declaration["calibration_repeats"]):
        for start in range(0, len(corpus), 4):
            rows = list(range(start, min(start + 4, len(corpus))))
            observations, _ = corpus.batch(rows)
            neural_seeds = tuple(derive_seed("separated-motor-calibration", declaration["calibration_seed"], repeat, i) for i in rows)
            efficacy = np.ones((len(rows), stack.agent.plasticity.topology.edge_count), dtype=np.float32)
            def sample():
                return np.asarray(stack.agent.processor.process(observations, fly_seeds=neural_seeds, efficacy=efficacy).output_activity).copy()
            values = sample()
            if repeat == 0:
                if not np.array_equal(values, sample()):
                    raise ValueError("reward-free calibration exact repeat failed")
                exact += len(rows)
            outputs.append(values); indices.extend(rows); seeds.extend(neural_seeds)
        print(json.dumps({"stage": "reward-free-calibration", "repeat": repeat + 1}), flush=True)
    activity = np.concatenate(outputs)
    if plastic_hashes(stack.agent.plasticity.state) != before:
        raise ValueError("reward-free calibration changed plastic state")
    activity_path = output / "calibration-activity.npz"
    np.savez_compressed(activity_path, output_hz=activity, corpus_indices=np.asarray(indices),
                        neural_seeds=np.asarray(seeds, dtype=np.int64), output_root_ids=stack.agent.processor.output_root_ids)
    metadata = {"corpus_sha256": corpus.sha256, "activity_sha256": sha256_file(activity_path),
        "duration_ms": 100, "initial_unit_weights": True, "plastic_state_unchanged": True,
        "exact_repeat_checks": exact, "calibration_repeats": declaration["calibration_repeats"]}
    calibration_masks = {key: value.copy() for key, value in corpus.masks.items()}
    calibration_masks["action_type_mask"][:, int(UpstreamActionType.SKIP_BLIND)] = False
    metadata["allow_skip_blinds"] = False
    baseline, separated, report = calibrate_separated(stack.agent.motor.mapping, activity,
        contexts=motor_context_windows(calibration_masks, order=indices), metadata=metadata)
    report["activity"] = metadata
    write_json(output / "calibration-report.json", report)
    (output / "configs").mkdir()
    common = replace(base,
        environment=replace(base.environment, num_envs=declaration["replicates"]),
        fly=replace(base.fly, artifact_path=str(resolve_path(base, base.fly.artifact_path))),
        calibration=replace(base.calibration, corpus_path=str(resolve_path(base, base.calibration.corpus_path))),
        training=replace(base.training, training_seed_offset=4000,
            max_environment_decisions=declaration["steps_per_learner"] * declaration["replicates"],
            checkpoint_every_decisions=min(1000, declaration["steps_per_learner"]) * declaration["replicates"],
            budget_basis="Predeclared separated motor pilot; fresh independent matched learners"))
    arms = {}
    configs = []
    for name, mapping in (("baseline", baseline), ("separated", separated)):
        motor_path = output / f"{name}-motor.json"
        mapping.save(motor_path)
        config = replace(common, name=f"separated-motor-{name}-v1",
                         fly=replace(common.fly, motor_mapping_path=str(motor_path.resolve())))
        path = output / "configs" / f"{name}.toml"
        document = asdict(config); document.pop("source_path")
        path.write_text(render_toml(document))
        configs.append(PlasticExperimentConfig.load(path))
        arms[name] = {"config": str(path.resolve()), "config_sha256": sha256_file(path),
            "motor": str(motor_path.resolve()), "motor_file_sha256": sha256_file(motor_path),
            "motor_mapping_sha256": mapping.structure_sha256, "separation": separation_summary(mapping)}
    validate_pair(*configs)
    plan = {"arms": arms, "declaration_sha256": sha256_file(output / "declaration.json"),
            "calibration_report_sha256": sha256_file(output / "calibration-report.json")}
    write_json(output / "training-plan.json", plan)
    return plan


def validate_evaluation(report, arm, index, declaration, *, initial=False):
    expected_source = "initial-efficacy-from-verified-checkpoint" if initial else "trained-checkpoint"
    if (report.get("learning_disabled") is not True or report.get("training_action_schedule_replayed") is not False
            or report.get("evaluation_weight_source") != expected_source
            or report.get("selected_learner_index") != index
            or report.get("action_policy") != "independent-fixed-motor-choices-no-training-action-replay"):
        raise ValueError("pilot requires independent frozen evaluation of the declared learner and weight source")
    if (report["seeds"] != declaration["evaluation_seeds"] or report["seed_stream"] != "validation"
            or report["win_ante"] != 1 or report["evaluation_batch_size"] != declaration["evaluation_batch_size"]
            or report["source_checkpoint_learners"] != declaration["replicates"]):
        raise ValueError("pilot evaluation schedule differs")
    components = report["components"]
    if (components["motor_mapping_sha256"] != arm["motor_mapping_sha256"]
            or components["action_constraints"] != {"allow_skip_blinds": False}
            or components["credit_assignment"]["version"] != "three-factor-choice-gated-v1"
            or components["training_seeds"] != declaration["training_seeds"]
            or components["exposure_budget_decisions"] != declaration["steps_per_learner"] * declaration["replicates"]
            or report["metrics"]["evaluation/mean_skipped_blinds"] != 0):
        raise ValueError("pilot evaluation model or exposure differs")
    if [row["seed"] for row in report["episodes"]] != declaration["evaluation_seeds"]:
        raise ValueError("pilot episode rows differ from declared seeds")


def outcomes(report):
    return {"blinds_beaten": sum(row["blinds_beaten"] for row in report["episodes"]),
            "wins": sum(int(row["won"]) for row in report["episodes"]),
            "mean_return": float(np.mean([row["episode_return"] for row in report["episodes"]]))}


def assess(output, declaration, plan):
    check_sources(declaration)
    for arm in plan["arms"].values():
        if sha256_file(Path(arm["config"])) != arm["config_sha256"] or sha256_file(Path(arm["motor"])) != arm["motor_file_sha256"]:
            raise ValueError("pilot arm artifact changed")
    initials = {name: json.loads((output / f"{name}-initial" / "summary.json").read_text()) for name in plan["arms"]}
    rows = []
    summaries = {name: json.loads((output / name / "run-summary.json").read_text()) for name in plan["arms"]}
    for name, report in initials.items():
        validate_evaluation(report, plan["arms"][name], 0, declaration, initial=True)
    for index in range(declaration["replicates"]):
        pair = {name: json.loads((output / f"{name}-eval-{index}" / "summary.json").read_text()) for name in plan["arms"]}
        for name, report in pair.items():
            validate_evaluation(report, plan["arms"][name], index, declaration)
            own = compare_frozen_evaluations(report, initials[name])
            write_json(output / f"{name}-learning-{index}.json", own)
        for field in set(MATCHED_FIELDS) - {"motor_mapping_sha256"}:
            if pair["baseline"]["components"][field] != pair["separated"]["components"][field]:
                raise ValueError(f"undeclared model difference: {field}")
        for field in ("motor_exploration", "action_constraints", "motor_routing_sha256"):
            if pair["baseline"]["components"][field] != pair["separated"]["components"][field]:
                raise ValueError(f"undeclared component difference: {field}")
        for left, right in zip(pair["baseline"]["episodes"], pair["separated"]["episodes"], strict=True):
            if left["balatro_seed"] != right["balatro_seed"]:
                raise ValueError("simulator seeds differ across arms")
        row = {"learner": index}
        for name, report in pair.items():
            trained, initial = outcomes(report), outcomes(initials[name])
            row[name] = {"trained": trained, "initial": initial,
                "blind_learning_gain": trained["blinds_beaten"]-initial["blinds_beaten"]}
        rows.append(row)
    totals = {name: {"trained_blinds": sum(row[name]["trained"]["blinds_beaten"] for row in rows),
                     "trained_wins": sum(row[name]["trained"]["wins"] for row in rows),
                     "blind_learning_gain": sum(row[name]["blind_learning_gain"] for row in rows)} for name in plan["arms"]}
    complete = all(summary["status"] == "completed" and summary["safety_stop"] is None
                   and summary["environment_decisions"] == declaration["steps_per_learner"]*declaration["replicates"] for summary in summaries.values())
    more_blinds = sum(row["separated"]["trained"]["blinds_beaten"] > row["baseline"]["trained"]["blinds_beaten"] for row in rows)
    more_gain = sum(row["separated"]["blind_learning_gain"] > max(0, row["baseline"]["blind_learning_gain"]) for row in rows)
    b, s = totals["baseline"], totals["separated"]
    alias = {}
    for name in plan["arms"]:
        metrics = [json.loads(line)["metrics"] for line in (output/name/"metrics.jsonl").read_text().splitlines()]
        alias[name] = {"mean": float(np.mean([row["credit/card_alias_edge_fraction_mean"] for row in metrics])),
                       "max": float(max(row["credit/card_alias_edge_fraction_mean"] for row in metrics))}
    if alias["separated"]["max"] != 0:
        raise ValueError("separated arm admitted credit to unchosen card outputs")
    signal = (complete and more_blinds > declaration["replicates"]/2 and more_gain > declaration["replicates"]/2
              and s["trained_blinds"] > b["trained_blinds"] and s["blind_learning_gain"] > max(0, b["blind_learning_gain"])
              and s["trained_wins"] >= b["trained_wins"])
    return {"version": VERSION, "comparability": "PASS", "all_budgets_complete_without_safety_stop": complete,
            "replicates": rows, "totals": totals, "training_card_alias": alias,
            "replicates_with_more_blind_clears": more_blinds, "replicates_with_larger_positive_learning_gain": more_gain,
            "predeclared_scale_up_signal": bool(signal), "interpretation": declaration["interpretation"]}


def run(output, declaration, base):
    suite = {"status": "running", "jobs": []}
    save = lambda: write_json(output / "suite.json", suite)
    def execute(name, command, accepted=(0,)):
        check_sources(declaration)
        job = {"name": name, "command": list(map(str, command)), "status": "running"}
        suite["jobs"].append(job); save()
        started = time.perf_counter()
        with (output/f"{name}.stdout").open("x") as stdout, (output/f"{name}.stderr").open("x") as stderr:
            process = subprocess.Popen(job["command"], stdout=stdout, stderr=stderr)
            job["pid"] = process.pid; save()
            job["returncode"] = process.wait()
        job.update(status="passed" if job["returncode"] == 0 else "safety_stop" if job["returncode"] in accepted else "failed",
                   wall_seconds=time.perf_counter()-started)
        save(); print(name, job["status"], flush=True)
        if job["returncode"] not in accepted:
            raise RuntimeError(f"{name} failed; inspect captured logs")
    def module(name, *args):
        return [sys.executable, "-m", name, *map(str, args)]
    try:
        plan = calibrate(base, output, declaration)
        width = declaration["replicates"]
        for name, arm in plan["arms"].items():
            execute(f"{name}-resume", module("flylatro.analysis.resume_cli", "--config", arm["config"],
                "--checkpoint", output/f"{name}-resume.pkl", "--prefix-decisions", 4*width,
                "--future-decisions", 4*width, "--output", output/f"{name}-resume.json", "--heavy"))
        for name, arm in plan["arms"].items():
            execute(f"{name}-training", module("flylatro.learning.cli", "--config", arm["config"],
                "--run-dir", output/name, "--heavy", "--dashboard", "--dashboard-refresh", 2, "--no-tensorboard"), (0, 2))
        for name, arm in plan["arms"].items():
            checkpoint = json.loads((output/name/"run-summary.json").read_text())["checkpoint"]
            for index in [-1, *range(width)]:
                label = f"{name}-initial" if index < 0 else f"{name}-eval-{index}"
                args = ["--initial-weights"] if index < 0 else []
                execute(label, module("flylatro.evaluation.plastic_cli", "--config", arm["config"],
                    "--checkpoint", checkpoint, "--learner-index", max(index, 0), "--episodes", declaration["evaluation_episodes"],
                    "--batch-size", declaration["evaluation_batch_size"], "--seed-stream", "validation",
                    "--seed-offset", declaration["evaluation_seed_offset"], "--win-ante", 1,
                    "--max-vector-steps", 1000, "--output-dir", output/label, "--heavy", *args))
        execute("checkpoint-audit", [sys.executable, "scripts/verify_cli_artifacts.py", "--run-dir", output/"baseline",
            "--run-dir", output/"separated", "--output", output/"checkpoint-audit.json"])
        write_json(output/"assessment.json", assess(output, declaration, plan))
        suite["status"] = "completed"
    except BaseException as error:
        suite.update(status="failed", error=f"{type(error).__name__}: {error}")
        if getattr(error, "report", None):
            suite["error_detail"] = error.report
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("artefacts/choice-credit-20261016-pilot-v1/configs/choice.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps-per-learner", type=int, default=10000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--evaluation-episodes", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    base = PlasticExperimentConfig.load(args.config)
    if not args.prepare_only:
        base.require_heavy_opt_in(args.heavy)
    output = args.output_dir.resolve()
    declaration = prepare(base, output, steps=args.steps_per_learner, replicates=args.replicates,
        episodes=args.evaluation_episodes, batch=args.evaluation_batch_size)
    if not args.prepare_only:
        run(output, declaration, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
