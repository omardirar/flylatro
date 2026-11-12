"""Continue the preserved passing pilot/control after repairing resume audit."""
import argparse
import json
from pathlib import Path

from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_manifest
from flylatro.learning.protocol import validate_control_group
from flylatro.learning.staged_cli import validate_checkpoint_health
from run_lr_pilots import OUT, gpu_command, module, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot-execution", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    original = json.loads(args.pilot_execution.read_text())
    if original["status"] != "halted" or "resume-audit failed with exit 2" not in original["error"]:
        raise ValueError("only the preserved, diagnosed resume-audit failure can be continued here")
    tag = args.tag
    destination = OUT / f"{tag}-pilot-validation-v2.json"
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = Path(original["primary"]) / "plastic-checkpoint-final.pkl"
    manifest = load_plastic_manifest(checkpoint)
    if manifest["environment_decisions"] != 250 or original["summary"]["status"] != "completed":
        raise ValueError("pilot must have completed its declared exposure")
    representation = OUT / f"{tag}-pilot-representation.json"
    reachability = OUT / f"{tag}-pilot-reachability.json"
    expected = json.loads((OUT / f"{tag}-pilot-preflight.json").read_text())["expected_identity"]
    validate_checkpoint_health(json.loads(representation.read_text()), json.loads(reachability.read_text()),
        expected, corpus_states=96, checkpoint=checkpoint)
    manifests = [Path(original[role]) / "run-manifest.json" for role in ("primary", "control")]
    validation = validate_control_group([json.loads(path.read_text()) for path in manifests])
    if validation["gates"]["status"] != "PASS":
        raise ValueError("preserved exact control evidence no longer validates")
    result = {**original, "status": "resume_audit_v2", "error": None,
        "preserved_execution_sha256": sha256_file(args.pilot_execution),
        "verified_input_sha256": {str(path): sha256_file(path) for path in (*manifests, representation, reachability)},
        "checkpoint": str(checkpoint), "representation": str(representation), "reachability": str(reachability),
        "control_validation": str(OUT / f"{tag}-pilot-control-validation.json")}
    write(destination, result)
    try:
        resume = OUT / f"{tag}-resume-audit-v2.json"
        gpu_command(module("flylatro.analysis.resume_cli", "--config", original["config"],
            "--checkpoint", OUT / f"{tag}-resume-probe-v2.pkl", "--prefix-decisions", 3,
            "--future-decisions", 13, "--output", resume, "--heavy"), f"{tag}-resume-audit-v2")
        result.update(status="passed", resume_audit=str(resume)); write(destination, result)
    except Exception as error:
        result.update(status="halted", error=str(error)); write(destination, result)
        raise


if __name__ == "__main__":
    main()
