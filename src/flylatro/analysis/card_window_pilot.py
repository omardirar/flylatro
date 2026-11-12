"""Predeclare and run a frozen 100 ms versus longer-window card-response study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from flylatro.analysis.card_information import prepare_cases
from flylatro.analysis.card_window import FOLLOW_UP_RULE, VERSION, audit_reference, compare_windows
from flylatro.analysis.evidence import load_corpus
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.choice_credit_pilot import implementation_hash, validate_pair
from flylatro.learning.config import PlasticExperimentConfig, _lif_parameters


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_sources(declaration):
    for source in declaration["sources"].values():
        for key in ("config", "checkpoint"):
            if sha256_file(Path(source[key])) != source[key + "_sha256"]:
                raise ValueError("declared source changed during window study")
    if implementation_hash() != declaration["implementation_sha256"]:
        raise ValueError("declared implementation changed during window study")


def prepare(pilot, output, *, durations, states=20, cards_per_state=2, repeats=4, seed=24_200_000, reference=None):
    pilot, output = pilot.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if min(states, cards_per_state) < 1 or repeats < 2:
        raise ValueError("positive state/card counts and at least two repeats required")
    if json.loads((pilot / "assessment.json").read_text())["comparability"] != "PASS":
        raise ValueError("source matched pilot comparability must pass")
    configs = {label: PlasticExperimentConfig.load(pilot / "configs" / f"{label}.toml")
               for label in ("baseline", "choice")}
    validate_pair(configs["baseline"], configs["choice"])
    config = configs["choice"]
    dt = _lif_parameters(config).dt_ms
    durations = [float(value) for value in durations]
    if (len(durations) < 2 or any(not math.isfinite(value) or value <= 0
            or not math.isclose(value / dt, round(value / dt), abs_tol=1e-7, rel_tol=0) for value in durations)
            or durations != sorted(set(durations)) or durations[0] != config.fly.duration_ms):
        raise ValueError("durations must start at source duration and increase in whole simulation steps")
    if config.environment.allow_skip_blinds or config.reinforcement.run_failure_pulse != .01:
        raise ValueError("window study retains disabled skips and failure pulse 0.01")
    corpus = load_corpus(config)
    cases = prepare_cases(corpus, states=states, cards_per_state=cards_per_state)
    sources = {label: {"config": str(config.source_path), "config_sha256": sha256_file(config.source_path),
        "checkpoint": str(pilot / label / "plastic-checkpoint-final.pkl"),
        "checkpoint_sha256": sha256_file(pilot / label / "plastic-checkpoint-final.pkl")}
        for label, config in configs.items()}
    windows = []
    for duration in durations:
        name = f"window-{duration:g}ms"
        command = [sys.executable, "-m", "flylatro.analysis.card_information_cli",
            "--pilot-dir", str(pilot), "--analysis-duration-ms", str(duration),
            "--states", str(states), "--cards-per-state", str(cards_per_state),
            "--repeats", str(repeats), "--seed", str(seed), "--output-dir", str(output / name), "--heavy"]
        windows.append({"name": name, "duration_ms": duration, "command": command, "output_dir": str(output / name)})
    declaration = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen duration intervention only; checkpoint verification at original training duration; no training or normalization recalibration",
        "implementation_sha256": implementation_hash(), "sources": sources,
        "corpus_sha256": corpus.sha256, "case_plan": [{"state_index": case["state_index"], "slot": case["slot"], "state_hash": case["state_hash"]} for case in cases],
        "states_requested": states, "cards_per_state": cards_per_state, "neural_repeats": repeats,
        "seed": seed, "primary_arm": "choice", "criterion": FOLLOW_UP_RULE,
        "health_criterion": "finite nonnegative rates; per-sample KC/card-pool active fractions meet source safety minima; fraction above 200 Hz at most 0.25; all plastic state unchanged",
        "windows": windows,
        "reference": str(reference.resolve()) if reference else None,
        "reference_report_sha256": sha256_file(reference / "report.json") if reference else None,
        "interpretation": "descriptive follow-up screen; no confirmatory inference, gameplay or learning claim"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "declaration.json", declaration)
    return declaration


def run(output, declaration):
    suite = {"status": "running", "jobs": []}
    save = lambda: write_json(output / "suite.json", suite)
    save()
    try:
        for index, window in enumerate(declaration["windows"]):
            check_sources(declaration)
            row = {"name": window["name"], "command": window["command"], "status": "running"}
            suite["jobs"].append(row)
            save()
            started = time.perf_counter()
            with (output / (window["name"] + ".stdout")).open("x") as stdout, (output / (window["name"] + ".stderr")).open("x") as stderr:
                process = subprocess.Popen(window["command"], stdout=stdout, stderr=stderr)
                row["pid"] = process.pid
                save()
                row["returncode"] = process.wait()
            row.update(status="passed" if row["returncode"] == 0 else "failed",
                       wall_seconds=time.perf_counter() - started)
            save()
            if row["returncode"]:
                raise RuntimeError(window["name"] + " failed; inspect captured output")
            check_sources(declaration)
            if index == 0 and declaration["reference"]:
                reference = Path(declaration["reference"])
                if sha256_file(reference / "report.json") != declaration["reference_report_sha256"]:
                    raise ValueError("reference report changed")
                write_json(output / "reference-audit.json", audit_reference(Path(window["output_dir"]), reference))
            print(json.dumps({"window": window["duration_ms"], "status": "completed"}), flush=True)
        base = Path(declaration["windows"][0]["output_dir"])
        assessments = []
        for window in declaration["windows"][1:]:
            assessment = compare_windows(base, Path(window["output_dir"]))
            assessment["declaration_sha256"] = sha256_file(output / "declaration.json")
            assessments.append(assessment)
        write_json(output / "assessment.json", {"version": VERSION, "comparisons": assessments})
        check_sources(declaration)
        suite["status"] = "completed"
    except BaseException as error:
        suite.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=Path("artefacts/choice-credit-20261016-pilot-v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--durations-ms", type=float, nargs="+", default=[100, 250])
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--cards-per-state", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=24_200_000)
    parser.add_argument("--reference", type=Path, help="audit baseline against the earlier information-flow experiment")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.heavy:
        raise ValueError("--heavy is required for a real neural window study")
    declaration = prepare(args.pilot_dir, args.output_dir, durations=args.durations_ms,
        states=args.states, cards_per_state=args.cards_per_state, repeats=args.repeats, seed=args.seed, reference=args.reference)
    if not args.prepare_only:
        run(args.output_dir.resolve(), declaration)
    print(json.dumps({"output_dir": str(args.output_dir), "prepared_only": args.prepare_only}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
