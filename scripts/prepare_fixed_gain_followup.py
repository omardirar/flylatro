"""Persist the bounded reward-free gain follow-up before observing its results."""
from dataclasses import asdict, replace
import json

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig
from flylatro.learning.protocol_materialize import render_toml
from run_lr_pilots import OUT, ROOT


def main():
    target = OUT / "fixed-gain-followup-predeclaration.json"
    if target.exists():
        raise FileExistsError(target)
    previous = OUT / "fixed-dynamics-execution.json"
    if json.loads(previous.read_text())["status"] != "no_ready_candidate":
        raise ValueError("previous experiment must be terminal with no ready candidate")
    configurations = []
    # First regenerate the three failed audits using the corrected Pearson
    # estimator and their verified actual-GPU neural activity, under new names.
    for duration in (100, 250, 500):
        original_tag = f"source-complete-gain0p18-{duration}ms"
        path = ROOT / f"configs/plastic-v783-{original_tag}.toml"
        configurations.append({"tag": original_tag + "-pearson-v5",
            "duration_ms": duration, "synapse_scale_mv": 0.18,
            "config": str(path.relative_to(ROOT)), "config_sha256": sha256_file(path),
            "activity_cache": str((OUT / f"{original_tag}-unit-activity.npz").relative_to(ROOT))})
    base = PlasticExperimentConfig.load(ROOT / "configs/plastic-v783-source-complete-calibration.toml")
    for gain in (0.20, 0.225, 0.25, 0.275):
        tag = "source-complete-gain" + str(gain).replace(".", "p") + "-100ms-pearson-v5"
        config = replace(base, name=tag, fly=replace(base.fly, duration_ms=100, synapse_scale_mv=gain))
        path = ROOT / f"configs/plastic-v783-{tag}.toml"
        payload = asdict(config); payload.pop("source_path")
        with path.open("x") as stream:
            stream.write(render_toml(payload))
        configurations.append({"tag": tag, "duration_ms": 100, "synapse_scale_mv": gain,
            "config": str(path.relative_to(ROOT)), "config_sha256": sha256_file(path)})
    plan = {
        "version": "fixed-gain-followup-predeclaration-v1",
        "prior_execution_sha256": sha256_file(previous),
        "reason": "All fixed-gain 0.18 windows passed representation but failed action-type signal diversity. Test higher fixed drive at the shortest declared window; no rewards or game performance enter selection.",
        "estimator_repair": "Population std was paired with N-1 covariance, inflating correlations. v5 motor and v4 representation use Pearson correlation with consistent denominator. The 0.50 diversity gate and every other threshold remain unchanged.",
        "configurations": configurations,
        "selection": "First configuration in declared order passing full-corpus pre/post representation, reachability, motor, integrated plasticity, specificity and initial preflight with no FAILs. No readiness failure permits training.",
        "sampling": {"corpus_states": 96, "repeats": 3, "batch_size": 8, "activity_seed": 12000000},
        "unchanged": ["complete anatomical source", "primary plasticity architecture", "LR 0.002", "eligibility", "reinforcement", "sensory mapping", "all gates", "three 1000-decision pairs", "held-out statistical declaration"],
        "budget_seconds": 21600,
        "implementation_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            ROOT / "scripts/run_dynamics_sensitivity.py",
            ROOT / "src/flylatro/interface/motor_calibration.py",
            ROOT / "src/flylatro/analysis/representation.py")},
    }
    with target.open("x") as stream:
        json.dump(plan, stream, indent=2); stream.write("\n")
    print(json.dumps({"predeclaration": str(target), "configurations": len(configurations)}))


if __name__ == "__main__":
    main()
