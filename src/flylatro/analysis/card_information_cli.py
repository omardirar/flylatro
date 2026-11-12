"""Trace single-card changes through frozen sensory, KC, MBON and card readouts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np

from flylatro.analysis.card_information import (
    CONDITIONS, STAGE_UNITS, VERSION, measure_learner, prepare_cases, sensory_effects, summarize,
    verify_erased_input_controls,
)
from flylatro.analysis.evidence import load_corpus
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint, load_plastic_manifest
from flylatro.learning.config import (
    PlasticExperimentConfig, build_plastic_stack, fly_dynamics_payload, payload_sha256,
)
from flylatro.analysis.card_window import integration_window


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--learner-index", type=int, action="append")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--cards-per-state", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=24_200_000)
    parser.add_argument("--analysis-duration-ms", type=float,
                        help="declared frozen timing intervention after strict source-checkpoint verification; source identity stays unchanged")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if min(args.states, args.cards_per_state) < 1 or args.repeats < 2:
        raise ValueError("positive state/card counts and at least two neural repeats required")
    if args.analysis_duration_ms is not None and (not np.isfinite(args.analysis_duration_ms) or args.analysis_duration_ms <= 0):
        raise ValueError("analysis duration must be positive and finite")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.pilot_dir:
        if args.config or args.checkpoint:
            parser.error("--pilot-dir cannot be combined with --config/--checkpoint")
        if json.loads((args.pilot_dir / "assessment.json").read_text())["comparability"] != "PASS":
            raise ValueError("matched pilot comparability must pass")
        sources = {label: (args.pilot_dir / "configs" / f"{label}.toml",
                          args.pilot_dir / label / "plastic-checkpoint-final.pkl")
                   for label in ("baseline", "choice")}
        from flylatro.learning.choice_credit_pilot import validate_pair
        validate_pair(*(PlasticExperimentConfig.load(paths[0]) for paths in sources.values()))
    else:
        if not args.config or not args.checkpoint:
            parser.error("provide --pilot-dir or both --config and --checkpoint")
        sources = {"checkpoint": (args.config, args.checkpoint)}
    configs = {label: PlasticExperimentConfig.load(paths[0]) for label, paths in sources.items()}
    for config in configs.values():
        config.require_heavy_opt_in(args.heavy)
        if config.fly.backend != "flywire" or config.fly.mode != "mbon_direct":
            raise ValueError("information-flow experiment requires the FlyWire MBON-direct stack")
        indices = args.learner_index if args.learner_index is not None else range(config.environment.num_envs)
        if (not indices or len(set(indices)) != len(indices)
                or any(not 0 <= index < config.environment.num_envs for index in indices)):
            raise ValueError("learner indices must be unique and inside checkpoint configuration")
    corpus = load_corpus(next(iter(configs.values())), args.corpus)
    cases = prepare_cases(corpus, states=args.states, cards_per_state=args.cards_per_state)
    started = time.perf_counter()
    sources_before = {label: {"config": sha256_file(paths[0]), "checkpoint": sha256_file(paths[1])}
                      for label, paths in sources.items()}
    implementation = [Path(__file__), Path(__file__).with_name("card_information.py"),
                      Path(__file__).with_name("card_window.py")]
    implementation_before = {path.name: sha256_file(path) for path in implementation}
    report = {"version": VERSION, "status": "running", "learning_disabled": True,
        "source_hashes": sources_before, "implementation_sha256": implementation_before,
        "corpus_sha256": corpus.sha256, "seed": args.seed, "repeats": args.repeats,
        "conditions": list(CONDITIONS), "population_units": STAGE_UNITS,
        "selection": "distinct competing PLAY states spaced in corpus order; visible legal cards spaced across occupied slots; no outcomes",
        "intervention": "one visible rank or suit one-hot changed to next cyclic category; all other observation fields and legality bits fixed",
        "controls": "common neural seeds for original/rank/suit and initial/trained weights; independent seeds for repeat variability; exact duplicates at first seed; no reward or simulator outcomes used",
        "interpretation": "descriptive rate sensitivity, not information-theoretic loss or a decoding/significance test; unchanged drive proves that input distinction is absent from this encoding",
        "limitations": "curated states; isolated visible-input edits may be inconsistent with deck aggregates or ordinary hand ordering; no simulator-state mutation; spike timing excluded; slots/repeats are not independent states; population distance cannot establish usable rank/suit representations or localize a causal learning bottleneck",
        "cases": [{"state_index": case["state_index"], "state_hash": case["state_hash"], "slot": case["slot"],
                   "legal_card_mask": case["masks"]["card_select_mask"][0].tolist(), "changes": case["changes"]}
                  for case in cases], "arms": {}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "report.json"
    def save():
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
    save()
    try:
        for label, config in configs.items():
            stack = build_plastic_stack(config)
            load_plastic_checkpoint(sources[label][1], stack.trainer, expected_components=stack.components,
                                    require_matching_exposure_budget=False)
            encoder = stack.agent.processor.encoder
            if "sensory" not in report:
                records, vectors = sensory_effects(encoder, cases)
                path = args.output_dir / "sensory-activity.npz"
                np.savez_compressed(path, alpn_drive_hz=vectors,
                                    alpn_root_ids=encoder.mapping.available_alpn_root_ids)
                report["sensory"] = {"mapping_sha256": encoder.mapping.sha256,
                    "records": records, "activity_file": path.name, "activity_sha256": sha256_file(path),
                    "array_axes": ["case", "condition", "available_ALPN_column"]}
            elif encoder.mapping.sha256 != report["sensory"]["mapping_sha256"]:
                raise ValueError("pilot arms have different sensory mappings")
            arm = {"config": str(sources[label][0].resolve()), "checkpoint": str(sources[label][1].resolve()),
                   "manifest": load_plastic_manifest(sources[label][1]), "components": stack.components, "learners": []}
            analysis_duration = config.fly.duration_ms if args.analysis_duration_ms is None else args.analysis_duration_ms
            dynamics = fly_dynamics_payload(replace(config, fly=replace(config.fly, duration_ms=analysis_duration)))
            arm["analysis_window"] = {"duration_ms": analysis_duration, "training_duration_ms": config.fly.duration_ms,
                "dynamics": dynamics, "dynamics_sha256": payload_sha256(dynamics),
                "normalization_policy": "fixed source motor pools and normalization; no recalibration",
                "checkpoint_policy": "strict verification at original training duration before declared frozen intervention"}
            report["arms"][label] = arm
            save()
            indices = args.learner_index if args.learner_index is not None else range(config.environment.num_envs)
            for index in indices:
                window = nullcontext() if args.analysis_duration_ms is None else integration_window(stack.agent.processor, analysis_duration)
                with window:
                    result = measure_learner(stack.agent, index, cases, repeats=args.repeats, seed=args.seed,
                        output_dir=args.output_dir, label=label,
                        progress=lambda weights, state, slot: print(json.dumps({"arm": label, "learner": index,
                            "analysis_duration_ms": analysis_duration,
                            "weights": weights, "completed_state": state, "slot": slot}), flush=True))
                result["identical_encoded_input_checks"] = verify_erased_input_controls(result, report["sensory"]["records"])
                for weights in result["weights"].values():
                    weights["summary"] = summarize(weights["rows"])
                arm["learners"].append(result)
                save()
            del stack, encoder
            gc.collect()
        for label, paths in sources.items():
            if sources_before[label] != {"config": sha256_file(paths[0]), "checkpoint": sha256_file(paths[1])}:
                raise ValueError("source config/checkpoint changed during diagnostic")
        if implementation_before != {path.name: sha256_file(path) for path in implementation}:
            raise ValueError("diagnostic implementation changed during execution")
        report.update(status="completed", sources_unchanged=True,
                      wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(json.dumps({"report": str(destination), "status": report["status"], "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
