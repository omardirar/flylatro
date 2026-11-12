"""Reproduce and record the first genuine frozen final-test win, if one exists."""
import argparse
import json
from pathlib import Path
import subprocess

from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.evaluation.compare import MATCHED_FIELDS
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.replay.bundle import ReplayBundle
from flylatro.replay.package import package_showcase
from flylatro.replay.verify import verify_replay
from run_lr_pilots import ROOT, gpu_command, module, write


def select_win(lock):
    for pair in lock["pairs"]:
        summary_path = Path(pair["trained_summary"])
        summary = json.loads(summary_path.read_text())
        if summary["seed_stream"] != "final_test" or summary["seeds"] != lock["final_seeds"] or summary["win_ante"] != 1:
            raise ValueError("showcase source differs from final-test declaration")
        if not summary["learning_disabled"] or summary["training_action_schedule_replayed"]:
            raise ValueError("showcase source is not an independent frozen evaluation")
        if summary["action_policy"] != "independent-fixed-motor-choices-no-training-action-replay":
            raise ValueError("showcase source uses an incompatible action policy")
        if [episode["seed"] for episode in summary["episodes"]] != lock["final_seeds"]:
            raise ValueError("showcase source episode rows differ from the declared seed order")
        if {key: summary["components"][key] for key in MATCHED_FIELDS} != lock["model_identity"]:
            raise ValueError("showcase source fixed model differs from final lock")
        if summary["plastic_weight_sha256"] != pair["trained_weight_sha256"] or summary["checkpoint_manifest"]["checkpoint_sha256"] != pair["trained_checkpoint_sha256"]:
            raise ValueError("showcase source differs from locked checkpoint")
        if summary["config_sha256"] != pair["trained_config_sha256"]:
            raise ValueError("showcase source differs from locked configuration")
        for key in ("checkpoint", "config"):
            expected = pair["trained_checkpoint_sha256"] if key == "checkpoint" else pair["trained_config_file_sha256"]
            if sha256_file(Path(pair["trained_" + key])) != expected:
                raise ValueError("showcase checkpoint/configuration file differs from final lock")
        for episode in summary["episodes"]:
            if episode["won"]:
                return pair, episode, summary_path
    return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    lock = json.loads(args.lock.read_text())
    selected = select_win(lock)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "showcase-evidence.json"
    report = {"version": "genuine-heldout-win-showcase-v1", "lock_sha256": sha256_file(args.lock),
        "selection": "first trained replicate, then first declared final seed with a recorded win; no new seed search or model tuning",
        "learning_claim": "A reproduced win alone does not establish statistical improvement."}
    if selected is None:
        report.update(status="no_genuine_final_test_win", genuine_win_reproduced=False)
        write(report_path, report); print(json.dumps(report)); return
    pair, episode, summary_path = selected
    source = summary_path.parent / "replay-candidates" / f"seed-{episode['seed']}"
    bundle = ReplayBundle.open(source)
    metadata = bundle.manifest.get("metadata", {})
    if metadata.get("episode") != episode or metadata.get("win_ante") != 1:
        raise ValueError("candidate bundle differs from the recorded held-out win")
    if bundle.identity.checkpoint_sha256 != pair["trained_checkpoint_sha256"] or bundle.identity.plastic_weight_hash != pair["trained_weight_sha256"] or bundle.identity.simulator_seed != episode["seed"]:
        raise ValueError("candidate bundle differs from the frozen fly/seed identity")
    env = BalatroSimAdapter(1, win_ante=1)
    assert verify_replay(bundle, env) == ()
    report.update(status="verified_candidate", genuine_win_reproduced=True,
        trained_summary=str(summary_path), trained_summary_sha256=sha256_file(summary_path),
        episode=episode, source_bundle=str(source), source_manifest_sha256=sha256_file(source / "manifest.json"))
    neural_output = args.output_dir / "recorded-evaluation"
    command = module("flylatro.evaluation.plastic_cli", "--config", pair["trained_config"],
        "--checkpoint", pair["trained_checkpoint"], "--win-ante", 1, "--episodes", 1,
        "--seed-stream", "final_test", "--seed-offset", lock["final_seeds"].index(episode["seed"]),
        "--unlock-final-test", "--record-neural", "--record-spikes", "--output-dir", neural_output, "--heavy")
    report["recording_command"] = command
    write(report_path, report)
    if not args.execute:
        print(json.dumps(report)); return
    try:
        gpu_command(command, args.output_dir.name + "-heldout-win-recording")
        summary = json.loads((neural_output / "summary.json").read_text())
        if summary["episodes"] != [episode] or summary["plastic_weight_sha256"] != pair["trained_weight_sha256"]:
            raise ValueError("recorded neural showcase does not reproduce the frozen final-test win")
        recorded = neural_output / "replay-candidates" / f"seed-{episode['seed']}"
        recorded_bundle = ReplayBundle.open(recorded)
        if recorded_bundle.decisions != bundle.decisions:
            raise ValueError("recording changed the held-out winning trajectory")
        assert verify_replay(recorded_bundle, BalatroSimAdapter(1, win_ante=1)) == ()
        package = package_showcase(recorded, args.output_dir / "reproducible-bundle",
            checkpoint=Path(pair["trained_checkpoint"]), config=Path(pair["trained_config"]))
        config = PlasticExperimentConfig.load(Path(pair["trained_config"]))
        artifact = Path(config.fly.artifact_path)
        if not artifact.is_absolute():
            from flylatro.learning.config import resolve_path
            artifact = resolve_path(config, config.fly.artifact_path)
        frames = args.output_dir / "brain-frames"
        result = subprocess.run(module("flylatro.visualization.cli", "--bundle", package.path,
            "--artifact", artifact, "--output-dir", frames, "--fps", 10, "--full"), cwd=ROOT,
            capture_output=True, text=True, check=True)
        rendering = json.loads(result.stdout.splitlines()[-1])
        video_command = rendering["video_command"]
        subprocess.run(video_command, cwd=ROOT, capture_output=True, text=True, check=True)
        report.update(status="completed", recorded_summary_sha256=sha256_file(neural_output / "summary.json"),
            packaged_bundle=str(package.path), neural_recording_sha256=sha256_file(package.neural_path),
            visualization=rendering, video_sha256=sha256_file(frames / "brain.mp4"))
        write(report_path, report)
    except Exception as error:
        report.update(status="halted", error=str(error)); write(report_path, report)
        raise
    print(json.dumps({"status": report["status"], "seed": episode["seed"], "genuine_win_reproduced": True}))


if __name__ == "__main__":
    main()
