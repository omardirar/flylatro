"""Reconstruct one historical real trajectory, then compare updates on fixed activity.

Run from the repository root. This is a diagnostic, never an action policy.
No stopped checkpoint is resumed. Every efficacy vector starts at one.
"""
from dataclasses import replace
from pathlib import Path
import json
import pickle
import time

import numpy as np

from flylatro.analysis.evidence import experiment_identity, write_report
from flylatro.analysis.plasticity import plasticity_diagnostics, synaptic_change_report
from flylatro.fly.flywire_artifact import FlyWireArtifact, sha256_file
from flylatro.fly.mushroom_body.plasticity import ThreeFactorPlasticity
from flylatro.fly.mushroom_body.state import PlasticEdgeState, state_numpy
from flylatro.fly.mushroom_body.topology import PlasticEdgeTopology
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from flylatro.learning.reward_schedule import load_action_schedule
from flylatro.learning.safety import TrainingSafetyStop


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/ante1-guarded-1000-20261009/mapping-000-replicate-000-plastic_real"
CONFIG = ROOT / "artefacts/ante1-guarded-1000-arms/mapping-000-replicate-000-plastic_real.toml"
OUTPUT = ROOT / "artefacts/lr-sensitivity-20261009"


def forensic():
    config = PlasticExperimentConfig.load(CONFIG)
    artifact = FlyWireArtifact.load(ROOT / config.fly.artifact_path)
    topology = PlasticEdgeTopology.from_artifact(artifact)
    for name in ("plastic-checkpoint-000000000160.pkl", "plastic-checkpoint-safety-stop.pkl"):
        path = SOURCE / name
        manifest = load_plastic_manifest(path)
        payload = pickle.loads(path.read_bytes())
        state = PlasticEdgeState.from_state_dict(payload["trainer"]["plastic_state"])
        assert state.weight_sha256 == manifest["plastic_weight_sha256"]
        assert topology.sha256 == payload["components"]["plastic_topology_sha256"]
        dest = OUTPUT / (name.removesuffix(".pkl") + "-corrected-synapses.json")
        if dest.exists():
            raise FileExistsError(dest)
        write_report(dest, {
            "version": "checkpoint-synaptic-audit-v2",
            "checkpoint": str(path), "read_only_forensic_audit": True,
            "quarantined": "safety-stop" in name,
            "synaptic_changes": synaptic_change_report(topology, state),
            "diagnostics": plasticity_diagnostics(state, config.plasticity),
        }, experiment_identity(config, {**payload["components"],
            "checkpoint_sha256": sha256_file(path), "plastic_weight_sha256": state.weight_sha256},
            report_kind="checkpoint_synaptic_audit", report_version="checkpoint-synaptic-audit-v2"))
    metrics = [json.loads(row)["metrics"] for row in (SOURCE / "metrics.jsonl").read_text().splitlines()]
    pulses = [json.loads(row)["synthetic_reinforcement_channels"][0]
              for row in (SOURCE / "synthetic-reinforcement-events.jsonl").read_text().splitlines()]
    from collections import Counter
    actions = [json.loads(row)["actions"] for row in (SOURCE / "training-actions.jsonl").read_text().splitlines()]
    result = {
        "decisions": len(metrics), "source_log_hashes": {
            name: sha256_file(SOURCE / name) for name in
            ("metrics.jsonl", "training-actions.jsonl", "synthetic-reinforcement-events.jsonl")},
        "events": dict(Counter(e for pulse in pulses for e in pulse["events"])),
        "appetitive_total": sum(p["synthetic_appetitive"] for p in pulses),
        "aversive_total": sum(p["synthetic_aversive"] for p in pulses),
        "action_type_counts": dict(Counter(a["action_type"][0] for a in actions)),
        "played_card_counts": dict(Counter(a["n_cards"][0] for a in actions if a["action_type"][0] == 0)),
    }
    for label, sign in (("positive", 1), ("negative", -1)):
        rows = [m for m, p in zip(metrics, pulses) if sign * (p["synthetic_appetitive"] - p["synthetic_aversive"]) > 0]
        result[label] = {"events": len(rows),
            "absolute_update_sum": sum(m["plasticity/event_absolute_change"] for m in rows),
            "maximum_edge_update": max(m["plasticity/event_max_absolute_change"] for m in rows)}
    (OUTPUT / "reward-forensics.json").write_text(json.dumps(result, indent=2) + "\n")


def reconstruct():
    started = time.monotonic()
    config = PlasticExperimentConfig.load(CONFIG)
    stack = build_plastic_stack(config)
    trainer = stack.trainer
    trainer.env.set_win_ante(1)
    schedule, _ = load_action_schedule(SOURCE / "training-actions.jsonl")
    trainer.action_schedule = schedule
    source_metrics = [json.loads(row)["metrics"] for row in (SOURCE / "metrics.jsonl").read_text().splitlines()]
    source_pulses = [json.loads(row)["synthetic_reinforcement_channels"][0]
                     for row in (SOURCE / "synthetic-reinforcement-events.jsonl").read_text().splitlines()]
    rules = {rate: ThreeFactorPlasticity(stack.agent.plasticity.topology,
        PlasticEdgeState.initialize(stack.agent.plasticity.topology.edge_count),
        replace(config.plasticity, learning_rate=rate)) for rate in (0.002, 0.005, 0.02)}
    records, pres, posts, pulses, terminals = [], [], [], [], []
    math_error = 0.0
    historical_error = 0.0
    for step in range(len(schedule)):
        before_episodes = trainer.state.completed_episodes
        measured = trainer.step()
        neural, learning = trainer.last_decision.neural, trainer.last_learning
        pre, post = state_numpy(neural.edge_pre_activity), state_numpy(neural.edge_post_activity)
        pres.append(pre.copy()); posts.append(post.copy())
        pulse = learning.pulses[0]
        expected = source_pulses[step]
        assert pulse.appetitive == expected["synthetic_appetitive"]
        assert pulse.aversive == expected["synthetic_aversive"]
        pulses.append((pulse.appetitive, pulse.aversive))
        terminal = trainer.state.completed_episodes > before_episodes
        terminals.append(terminal)
        row = {"decision": step + 1, "terminal": terminal, "rates": {}}
        for rate, rule in rules.items():
            # Independent equation, separate from implementation methods.
            expected_eligibility = np.clip(
                rule.config.eligibility_decay * rule.state.eligibility
                + np.clip(pre / rule.config.kc_reference_hz, 0, 1)
                * np.clip(post / rule.config.mbon_reference_hz, 0, 1),
                -rule.config.max_eligibility, rule.config.max_eligibility)
            expected_weights = np.clip(rule.state.efficacy
                + rate * np.float32(pulse.appetitive - pulse.aversive) * expected_eligibility,
                rule.config.min_efficacy, rule.config.max_efficacy)
            rule.record_activity(pre, post)
            event = rule.apply_dopamine(pulse.appetitive, pulse.aversive)[0]
            np.testing.assert_array_equal(rule.state.eligibility, expected_eligibility)
            np.testing.assert_allclose(rule.state.efficacy, expected_weights, atol=2e-7, rtol=0)
            math_error = max(math_error, float(np.max(np.abs(rule.state.efficacy - expected_weights))))
            row["rates"][str(rate)] = {
                "absolute_update": event.absolute_change,
                "maximum_edge_update": event.maximum_absolute_change,
                "eligibility_mean_before_reset": event.mean_absolute_eligibility,
                "eligibility_max_before_reset": event.max_absolute_eligibility,
                "lower_bound_fraction": rule.routine_metrics()["lower_bound_fraction"],
                "mean_absolute_drift": rule.routine_metrics()["mean_absolute_drift"],
            }
            if terminal:
                rule.state.episode_count[0] += 1
                rule.state.reset_fast_traces(learners=(0,), eligibility=True)
        math_error = max(math_error, float(np.max(np.abs(
            rules[0.02].state.efficacy - state_numpy(stack.agent.plasticity.state.efficacy)))))
        for key in ("plasticity/event_absolute_change", "plasticity/mean_efficacy", "plasticity/lower_bound_fraction"):
            historical_error = max(historical_error, abs(measured[key] - source_metrics[step][key]))
            np.testing.assert_allclose(measured[key], source_metrics[step][key], atol=1e-5, rtol=1e-6)
        records.append(row)
        if (step + 1) % 20 == 0:
            print(json.dumps({"reconstructed_decisions": step + 1, "equation_max_error": math_error}), flush=True)
        failure = trainer.safety.observe(measured)
        if failure is not None:
            assert step + 1 == len(schedule), "unexpected early guard failure"
            break
    final_manifest = load_plastic_manifest(SOURCE / "plastic-checkpoint-safety-stop.pkl")
    assert stack.agent.plasticity.state.weight_sha256 == final_manifest["plastic_weight_sha256"]
    elapsed = time.monotonic() - started
    np.savez_compressed(OUTPUT / "fixed-neural-trajectory.npz", pre=np.asarray(pres), post=np.asarray(posts),
                        pulses=np.asarray(pulses), terminals=np.asarray(terminals))
    report = {"version": "fixed-trajectory-lr-sensitivity-v1", "decisions": len(records),
        "source_quarantined_state_resumed": False, "historical_final_weight_exact": True,
        "independent_equation_max_error": math_error, "historical_metric_max_error": historical_error,
        "trajectory_sha256": sha256_file(OUTPUT / "fixed-neural-trajectory.npz"),
        "gpu_wall_seconds": elapsed, "peak_cuda_allocated_bytes": __import__('torch').cuda.max_memory_allocated(),
        "rates": {str(rate): {"rule_sha256": rule.config.sha256,
                              "final": plasticity_diagnostics(rule.state, rule.config)} for rate, rule in rules.items()},
        "decisions_detail": records,
        "interpretation": "Identical activity/outcomes/resets isolate update magnitude. This is not independent-action learning evidence."}
    write_report(OUTPUT / "fixed-trajectory-comparison.json", report,
                 experiment_identity(config, stack.components, report_kind="fixed_trajectory_lr_sensitivity",
                                     report_version=report["version"]))
    print(json.dumps({"status": "PASS", "gpu_wall_seconds": elapsed, "equation_max_error": math_error}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--forensic-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.forensic_only:
        forensic()
    else:
        if (OUTPUT / "fixed-neural-trajectory.npz").exists():
            raise FileExistsError("trajectory already exists")
        reconstruct()
