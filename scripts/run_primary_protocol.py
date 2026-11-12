"""Run the declared three audited primary/exact-control pairs after pilot PASS."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.interface.motor import MotorMapping
from flylatro.learning.config import PlasticExperimentConfig, resolve_path
from flylatro.learning.protocol import ExperimentProtocol
from flylatro.learning.protocol_materialize import materialize_protocol, render_toml
from run_lr_pilots import OUT, ROOT, gpu_command, module, write


def save_config(path, config):
    payload = asdict(config); payload.pop("source_path")
    with path.open("x") as stream:
        stream.write(render_toml(payload))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot-execution", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    pilot = json.loads(args.pilot_execution.read_text())
    if pilot["status"] != "passed":
        raise ValueError("current-source stability, exact control and resume audits must all PASS")
    tag = args.tag
    destination = OUT / f"{tag}-primary-driver.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = {"status": "benchmarking", "pilot_execution_sha256": sha256_file(args.pilot_execution)}
    write(destination, result)
    try:
        base = PlasticExperimentConfig.load(Path(pilot["config"]))
        benchmark = OUT / f"{tag}-benchmark-v3.json"
        bench_command = [sys.executable, "-c", "from flylatro.benchmarks.cli import plastic_end_to_end_main; raise SystemExit(plastic_end_to_end_main())"]
        gpu_command([*bench_command, "--config", str(base.source_path), "--steps", "10",
            "--profile-components", "--output", str(benchmark), "--heavy"], f"{tag}-benchmark-v3")
        batched_path = ROOT / f"configs/plastic-v783-{tag}-batch4-benchmark.toml"
        batched = replace(base, name=tag + "-batch4-benchmark",
            environment=replace(base.environment, num_envs=4),
            training=replace(base.training, max_environment_decisions=20))
        save_config(batched_path, batched)
        batch_report = OUT / f"{tag}-batch4-benchmark-v3.json"
        gpu_command([*bench_command, "--config", str(batched_path), "--steps", "5",
            "--profile-components", "--compare-sequential", "--output", str(batch_report), "--heavy"],
            f"{tag}-batch4-benchmark-v3")
        # Primary arms remain one independently seeded fly each. A batch
        # disagreement is reported; it cannot silently change their execution.
        result["batched_neural_comparison"] = json.loads(batch_report.read_text())["neural_execution_comparison"]
        base = replace(base, name=tag + "-primary-base", calibration=replace(base.calibration, benchmark_report=str(benchmark)))
        base_path = ROOT / f"configs/plastic-v783-{tag}-primary-base.toml"
        save_config(base_path, base)
        base = PlasticExperimentConfig.load(base_path)
        protocol_path = OUT / f"{tag}-1000-protocol.json"
        if protocol_path.exists():
            raise FileExistsError(protocol_path)
        motor = MotorMapping.load(resolve_path(base, base.fly.motor_mapping_path))
        protocol = ExperimentProtocol.create(name=tag + "-1000-20261009", base_seed=910009,
            replicate_count=3, conditions=("plastic_real", "no_plasticity"),
            exposure_budget_decisions=1000, curriculum_ladder=(1,),
            motor_mapping_id=motor.structure_sha256, sensory_mapping_seed=base.fly.sensory_mapping_seed)
        protocol.save(protocol_path)
        arms = OUT / f"{tag}-1000-arms"
        materialize_protocol(protocol, base, protocol_path=protocol_path, output_dir=arms,
            run_root=str(ROOT / "runs"), budget_basis=f"measured pilot and {benchmark}; declared three x 1000 pairs; shared six-hour ledger",
            checkpoint_every_decisions=50)
        command = module("flylatro.learning.staged_cli", "--plan", arms / "protocol-plan.json",
            "--health-report", pilot["representation"], "--reachability-report", pilot["reachability"],
            "--control-manifest", Path(pilot["primary"]) / "run-manifest.json",
            "--control-manifest", Path(pilot["control"]) / "run-manifest.json",
            "--stages", "100,250,500,1000", "--batch-size", 8, "--repeats", 3,
            "--evaluation-episodes", 32, "--heavy")
        planned = OUT / f"{tag}-1000-validated-plan.json"
        gpu_command([*command, "--output", str(planned)], f"{tag}-1000-validation")
        execution = OUT / f"{tag}-1000-execution.json"
        result.update(status="primary_training", protocol=str(protocol_path),
            plan=str(arms / "protocol-plan.json"), execution=str(execution), benchmark=str(benchmark))
        write(destination, result)
        code = gpu_command([*command, "--output", str(execution), "--execute"], f"{tag}-1000-staged", allowed=(0, 2))
        result.update(status="passed" if code == 0 else "primary_halted")
        write(destination, result)
    except Exception as error:
        result.update(status="halted", error=str(error)); write(destination, result)
        raise


if __name__ == "__main__":
    main()
