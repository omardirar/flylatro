"""Run a materialized protocol in audited stages, halting on failed health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from flylatro.analysis.corpus import CalibrationCorpus
from flylatro.analysis.preflight import REPORT_IDENTITY_FIELDS, run_preflight
from flylatro.analysis.provenance import EvidenceIdentity, describe_mismatches
from flylatro.analysis.readiness import EvidencePaths
from flylatro.analysis.reachability import REACHABILITY_REPORT_VERSION
from flylatro.analysis.representation import REPRESENTATION_REPORT_VERSION
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.protocol import ExperimentProtocol, assert_configuration_matches_arm, validate_control_group


STAGED_PROTOCOL_VERSION = "audited-staged-plastic-protocol-v1"


def validate_checkpoint_health(
    representation: Mapping[str, Any], reachability: Mapping[str, Any],
    expected: Mapping[str, Any], *, corpus_states: int, checkpoint: Path | None = None,
) -> None:
    reference = EvidenceIdentity.from_mapping(expected)
    if reference is None:
        raise ValueError("checkpoint health requires the current experiment identity")
    claimed_checkpoint = representation.get("checkpoint")
    if not isinstance(claimed_checkpoint, Mapping) or not claimed_checkpoint.get("path"):
        raise ValueError("checkpoint health requires an explicit checkpoint path")
    checkpoint_path = checkpoint or Path(claimed_checkpoint.get("path", ""))
    manifest = load_plastic_manifest(checkpoint_path)
    for report, kind, version, fields in (
        (representation, "representation_checkpoint", REPRESENTATION_REPORT_VERSION, REPORT_IDENTITY_FIELDS["representation_post"]),
        (reachability, "kc_reachability_checkpoint", REACHABILITY_REPORT_VERSION, REPORT_IDENTITY_FIELDS["kc_reachability"]),
    ):
        identity = EvidenceIdentity.from_mapping(report.get("evidence_identity"))
        if identity is None or identity.report_kind != kind:
            raise ValueError(f"checkpoint health requires {kind}, not initial calibration")
        if identity.report_version != version:
            raise ValueError(f"checkpoint health requires supported report version {version}")
        if report.get("gates", {}).get("status") != "PASS":
            raise ValueError(f"{kind} failed: {report.get('gates', {}).get('failed', [])}")
        if report.get("efficacy_policy") != "frozen-checkpoint-efficacy":
            raise ValueError("checkpoint health must measure frozen learned weights")
        mismatches = identity.mismatches(reference, required_fields=fields)
        if mismatches:
            raise ValueError("checkpoint health is stale: " + describe_mismatches(mismatches))
        if identity.plastic_weight_sha256 != manifest["plastic_weight_sha256"] or identity.checkpoint_sha256 != manifest["checkpoint_sha256"]:
            raise ValueError("checkpoint health weight or checkpoint SHA-256 differs")
    if representation.get("corpus_states_used") != corpus_states or representation.get("repeats_per_state", 0) < 2:
        raise ValueError("checkpoint health must cover the full corpus with independent repeats")
    if manifest["environment_decisions"] < 1:
        raise ValueError("checkpoint health requires a previously exposed fly")


def _write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="materialized protocol-plan.json")
    parser.add_argument("--health-report", type=Path, required=True, help="passed learned-checkpoint representation report")
    parser.add_argument("--reachability-report", type=Path, required=True, help="passed learned-checkpoint reachability report")
    parser.add_argument("--control-manifest", type=Path, action="append", required=True)
    parser.add_argument("--stages", default="100,250,500,1000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="run the stages; default only validates and writes the execution plan")
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"staged execution report already exists: {args.output}")
    stages = [int(value) for value in args.stages.split(",")]
    if not stages or sorted(set(stages)) != stages or stages[0] < 1 or args.repeats < 2 or args.batch_size < 1 or args.evaluation_episodes < 1:
        raise ValueError("stages must be positive and strictly increasing; repeats >= 2 and batch size >= 1")
    if args.execute and not args.heavy:
        raise ValueError("executing a staged protocol requires --heavy")
    plan = json.loads(args.plan.read_text())
    representation = json.loads(args.health_report.read_text())
    reachability = json.loads(args.reachability_report.read_text())
    report: dict[str, Any] = {
        "version": STAGED_PROTOCOL_VERSION, "status": "validating", "stages": stages,
        "protocol_sha256": plan["protocol_sha256"], "plan_sha256": sha256_file(args.plan),
        "health_report_sha256": sha256_file(args.health_report),
        "reachability_report_sha256": sha256_file(args.reachability_report),
        "arms": [], "completed_stages": [], "failed": [],
        "control_validations": [], "evaluation_episodes": args.evaluation_episodes,
    }
    report_root = args.output.parent / (args.output.stem + "-reports")
    try:
        protocol_path = Path(plan["protocol_path"]).resolve()
        protocol = ExperimentProtocol.load(protocol_path)
        protocol_payload = json.loads(protocol_path.read_text())
        if protocol.sha256 != plan["protocol_sha256"]:
            raise ValueError("plan protocol SHA-256 differs from its protocol file")
        planned_ids = [entry["arm_id"] for entry in plan["arms"]]
        if len(set(planned_ids)) != len(planned_ids) or set(planned_ids) != {arm.arm_id for arm in protocol.arms}:
            raise ValueError("staged plan must contain every protocol arm exactly once")
        if set(protocol.conditions) != {"plastic_real", "no_plasticity"}:
            raise ValueError("staged protocol requires paired plastic_real/no_plasticity arms")
        planned_runs = [Path(entry["run_dir"]).resolve() for entry in plan["arms"]]
        if len(set(planned_runs)) != len(planned_runs):
            raise ValueError("each staged arm must have its own run directory")
        seen_arms: set[str] = set()
        for entry in plan["arms"]:
            config_path = Path(entry["config"]).resolve()
            config = PlasticExperimentConfig.load(config_path)
            if config.protocol.arm_id != entry["arm_id"] or config.protocol.protocol_sha256 != plan["protocol_sha256"]:
                raise ValueError("materialized arm or protocol binding differs from the plan")
            assert_configuration_matches_arm(protocol_payload, config)
            protocol_arm = protocol.arm(entry["arm_id"])
            dependencies = [protocol_arm.reference_arm_id] if config.training.condition == "no_plasticity" else []
            if entry["depends_on"] != dependencies or any(dependency not in seen_arms for dependency in dependencies):
                raise ValueError("staged arm dependencies must match the protocol and precede their control")
            if dependencies:
                source_dir = next(Path(item["run_dir"]).resolve() for item in plan["arms"] if item["arm_id"] == dependencies[0])
                if resolve_path(config, config.training.matched_source_run_dir).resolve() != source_dir or resolve_path(config, config.training.action_schedule_path).resolve() != source_dir / "training-actions.jsonl":
                    raise ValueError("matched control must replay its protocol reference run")
            seen_arms.add(entry["arm_id"])
            if config.training.max_environment_decisions != stages[-1] or any(stage % config.environment.num_envs for stage in stages):
                raise ValueError("stages must align with each arm and end at its unchanged declared exposure budget")
            if not config.safety.enabled:
                raise ValueError("staged experiments require default-on training safety")
            if config.training.condition not in {"plastic_real", "no_plasticity"} or not config.curriculum.enabled or config.curriculum.ladder != (1,):
                raise ValueError("this staged runner supports paired plastic_real/no_plasticity fixed-Ante-1 protocols")
            config.require_heavy_opt_in(args.heavy)
            run_dir = Path(entry["run_dir"]).resolve()
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"staged run already contains data: {run_dir}")
            arm_report = {"arm_id": entry["arm_id"], "condition": config.training.condition,
                          "config": str(config_path), "run_dir": str(run_dir), "commands": []}
            report["arms"].append(arm_report)
            if config.training.condition == "plastic_real":
                evidence = EvidencePaths.from_config(config).merge(EvidencePaths(
                    protocol=resolve_path(config, config.protocol.protocol_path),
                    control_manifests=tuple(args.control_manifest),
                ))
                preflight = run_preflight(config, profile="ante1", **evidence.as_reports())
                _write(report_root / entry["arm_id"].replace(":", "-") / "preflight.json", preflight)
                if preflight["status"] != "PASS":
                    raise ValueError(f"strict stage-start preflight failed: {preflight['failed']}")
                corpus_states = len(CalibrationCorpus.load(resolve_path(config, config.calibration.corpus_path)))
                validate_checkpoint_health(representation, reachability, preflight["expected_identity"], corpus_states=corpus_states)
                arm_report["expected_identity"] = preflight["expected_identity"]
                arm_report["corpus_states"] = corpus_states
            previous_checkpoint = None
            for stage in stages:
                command = [sys.executable, "-m", "flylatro.learning.cli", "--config", str(config_path),
                           "--run-dir", str(run_dir), "--stop-after-decisions", str(stage), "--no-tensorboard", "--heavy"]
                if previous_checkpoint is not None:
                    command += ["--resume", str(previous_checkpoint)]
                checkpoint = run_dir / ("plastic-checkpoint-final.pkl" if stage == stages[-1] else f"plastic-checkpoint-stage-{stage:012d}.pkl")
                arm_report["commands"].append({"stage": stage, "training": command, "checkpoint": str(checkpoint)})
                previous_checkpoint = checkpoint
            evaluation_dir = report_root / entry["arm_id"].replace(":", "-") / "evaluation"
            arm_report["evaluation_dir"] = str(evaluation_dir)
            arm_report["evaluation_command"] = [
                sys.executable, "-m", "flylatro.evaluation.plastic_cli", "--config", str(config_path),
                "--checkpoint", str(previous_checkpoint), "--condition", config.training.condition,
                "--win-ante", "1", "--episodes", str(args.evaluation_episodes), "--seed-stream", "validation",
                "--output-dir", str(evaluation_dir), "--progress", "--heavy",
            ]
        report["status"] = "running" if args.execute else "planned"
        _write(args.output, report)
        if not args.execute:
            print(json.dumps({"output": str(args.output), "status": report["status"], "arms": len(report["arms"])}))
            return 0
        # Generated order is primary then its exact control. Each checkpoint is
        # audited before the next primary stage; no further arm runs on failure.
        completed_arms: set[str] = set()
        for arm in report["arms"]:
            entry = next(item for item in plan["arms"] if item["arm_id"] == arm["arm_id"])
            if any(dependency not in completed_arms for dependency in entry["depends_on"]):
                raise ValueError("stage dependency is incomplete")
            for stage_plan in arm["commands"]:
                report["active_stage"] = {"arm_id": arm["arm_id"], "stage": stage_plan["stage"], "phase": "training", "checkpoint": stage_plan["checkpoint"]}
                _write(args.output, report)
                result = subprocess.run(stage_plan["training"], check=False)
                if result.returncode:
                    raise ValueError(f"training stage stopped: {arm['arm_id']} at {stage_plan['stage']} (exit {result.returncode})")
                if arm["condition"] == "plastic_real":
                    report["active_stage"]["phase"] = "checkpoint_health"
                    _write(args.output, report)
                    directory = report_root / arm["arm_id"].replace(":", "-")
                    rep_path = directory / f"representation-{stage_plan['stage']}.json"
                    reach_path = directory / f"reachability-{stage_plan['stage']}.json"
                    diagnostic = [sys.executable, "-m", "flylatro.analysis.representation_cli",
                                  "--config", arm["config"], "--stage", "post", "--checkpoint", stage_plan["checkpoint"],
                                  "--repeats", str(args.repeats), "--batch-size", str(args.batch_size),
                                  "--output", str(rep_path), "--reachability-output", str(reach_path), "--heavy"]
                    if subprocess.run(diagnostic, check=False).returncode:
                        raise ValueError(f"checkpoint neural health failed: {arm['arm_id']} at {stage_plan['stage']}")
                    validate_checkpoint_health(json.loads(rep_path.read_text()), json.loads(reach_path.read_text()),
                                               arm["expected_identity"], corpus_states=arm["corpus_states"], checkpoint=Path(stage_plan["checkpoint"]))
                report["completed_stages"].append({"arm_id": arm["arm_id"], "stage": stage_plan["stage"]})
                _write(args.output, report)
                print(json.dumps({"arm_id": arm["arm_id"], "stage": stage_plan["stage"], "status": "stage_passed"}), flush=True)
            completed_arms.add(arm["arm_id"])
            if arm["condition"] == "no_plasticity":
                report["active_stage"]["phase"] = "control_validation"
                _write(args.output, report)
                source = next(item for item in report["arms"] if item["arm_id"] in entry["depends_on"])
                validation = validate_control_group([
                    json.loads((Path(source["run_dir"]) / "run-manifest.json").read_text()),
                    json.loads((Path(arm["run_dir"]) / "run-manifest.json").read_text()),
                ])
                validation_path = report_root / arm["arm_id"].replace(":", "-") / "control-validation.json"
                _write(validation_path, validation)
                report["control_validations"].append({"arm_id": arm["arm_id"], "path": str(validation_path), "status": validation["gates"]["status"]})
                if validation["gates"]["status"] != "PASS":
                    raise ValueError(f"exact control validation failed: {validation['gates']['failed']}")
            report["active_stage"]["phase"] = "held_out_evaluation"
            _write(args.output, report)
            if subprocess.run(arm["evaluation_command"], check=False).returncode:
                raise ValueError(f"held-out evaluation failed: {arm['arm_id']}")
            if arm["condition"] == "no_plasticity":
                report["active_stage"]["phase"] = "paired_comparison"
                _write(args.output, report)
                comparison_path = report_root / arm["arm_id"].replace(":", "-") / "paired-evaluation.json"
                comparison = [sys.executable, "-m", "flylatro.evaluation.compare", "--trained", str(Path(source["evaluation_dir"]) / "summary.json"),
                              "--control", str(Path(arm["evaluation_dir"]) / "summary.json"), "--output", str(comparison_path)]
                if subprocess.run(comparison, check=False).returncode:
                    raise ValueError(f"held-out comparison failed: {arm['arm_id']}")
            _write(args.output, report)
        report["status"] = "completed"
        report["active_stage"] = None
    except (ValueError, OSError, KeyError) as error:
        report["status"] = "halted"
        report["failed"].append(str(error))
    _write(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "failed": report["failed"]}), flush=True)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
