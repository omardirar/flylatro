"""Diagnose frozen card selection, content permutation and immediate play quality."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np

from flylatro.analysis.card_selection import VERSION, diagnose_learner, score_plays
from flylatro.analysis.evidence import load_corpus
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.checkpoints import load_plastic_checkpoint, load_plastic_manifest
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_environment, build_plastic_stack


def prepare_cases(config, corpus, count):
    if count < 1 or corpus.snapshots is None:
        raise ValueError("positive state count and a corpus with verified snapshots required")
    env = build_plastic_environment(replace(config, environment=replace(config.environment, num_envs=1)))
    if str(env.simulator_version) != corpus.simulator_version:
        raise ValueError("corpus and scoring simulator versions differ")
    env.reset((0,))
    eligible = np.flatnonzero(corpus.masks["action_type_mask"][:, 0]
                             & (corpus.masks["card_select_mask"].sum(axis=1) > 1)
                             & (corpus.observations["hand_len"] >= 2))
    unique = {}
    for index in eligible.tolist():
        unique.setdefault(corpus.state_hashes[index], index)
    indices = list(unique.values())
    if not indices:
        raise ValueError("corpus has no competing legal PLAY states")
    indices = [indices[int(i)] for i in np.linspace(0, len(indices) - 1, min(count, len(indices)), dtype=int)]
    cases = []
    for index in indices:
        obs, masks = corpus.row(index)
        if not config.environment.allow_skip_blinds:
            masks["action_type_mask"][:, 3] = False
        scores = score_plays(env, corpus.snapshots[index], obs, masks)
        cases.append({"index": index, "observations": obs, "masks": masks, "scores": scores})
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, help="diagnose both completed matched-pilot arms and all learner rows")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--learner-index", type=int, action="append", help="repeat to select rows; default all rows separately")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=24_100_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    if min(args.states, args.repeats, args.permutations) < 1:
        raise ValueError("state, repeat and permutation counts must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.pilot_dir:
        if args.config or args.checkpoint:
            parser.error("--pilot-dir cannot be combined with --config/--checkpoint")
        assessment = json.loads((args.pilot_dir / "assessment.json").read_text())
        if assessment["comparability"] != "PASS":
            raise ValueError("matched pilot comparability must pass")
        sources = {label: (args.pilot_dir / "configs" / f"{label}.toml",
                          args.pilot_dir / label / "plastic-checkpoint-final.pkl") for label in ("baseline", "choice")}
        from flylatro.learning.choice_credit_pilot import validate_pair
        validate_pair(*(PlasticExperimentConfig.load(paths[0]) for paths in sources.values()))
    else:
        if not args.config or not args.checkpoint:
            parser.error("provide --pilot-dir or both --config and --checkpoint")
        sources = {"checkpoint": (args.config, args.checkpoint)}
    configs = {label: PlasticExperimentConfig.load(paths[0]) for label, paths in sources.items()}
    for config in configs.values():
        config.require_heavy_opt_in(args.heavy)
        indices = args.learner_index if args.learner_index is not None else range(config.environment.num_envs)
        if not indices or any(not 0 <= index < config.environment.num_envs for index in indices):
            raise ValueError("learner index outside checkpoint configuration")
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate learner indices")
    config = next(iter(configs.values()))
    corpus = load_corpus(config, args.corpus)
    started = time.perf_counter()
    source_hashes = {label: {"config": sha256_file(paths[0]), "checkpoint": sha256_file(paths[1])}
                     for label, paths in sources.items()}
    cases = prepare_cases(config, corpus, args.states)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"version": VERSION, "status": "running", "learning_disabled": True,
              "corpus_sha256": corpus.sha256, "simulator_version": corpus.simulator_version,
              "source_hashes": source_hashes, "seed": args.seed, "repeats": args.repeats,
              "requested_permutations": args.permutations,
              "selection": "distinct legal PLAY states spaced in corpus order; no outcome-based selection",
              "intervention": "occupied observation slots and legal bits permuted; choices mapped to original snapshot; forced PLAY head diagnostic",
              "scoring": "real simulator scalar immediate reward; canonical original-slot card order; every legal subset; snapshot restored before each play",
              "limitations": "curated calibration corpus, conditional PLAY choices and observation interventions; repeated trials are not independent game states; immediate reward is not long-term value; no win-rate or confirmatory learning claim",
              "implementation_sha256": {path.name: sha256_file(path) for path in (Path(__file__), Path(__file__).with_name("card_selection.py"))},
              "states": [{"index": case["index"], "state_hash": corpus.state_hashes[case["index"]],
                          "snapshot_sha256": corpus.snapshot_sha256[case["index"]],
                          "hand_len": int(case["observations"]["hand_len"][0]),
                          "legal_subsets": [{"cards": list(cards), "immediate_reward": value}
                                            for cards, value in case["scores"].items()]}
                         for case in cases], "arms": {}}
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
            arm = {"config": str(sources[label][0].resolve()), "checkpoint": str(sources[label][1].resolve()),
                   "manifest": load_plastic_manifest(sources[label][1]), "components": stack.components, "learners": []}
            report["arms"][label] = arm
            indices = args.learner_index if args.learner_index is not None else range(config.environment.num_envs)
            for index in indices:
                result = diagnose_learner(stack.agent, index, cases, repeats=args.repeats,
                    permutation_count=args.permutations, seed=args.seed,
                    progress=lambda state: print(json.dumps({"arm": label, "learner": index, "completed_state": state}), flush=True))
                arm["learners"].append(result)
                save()
            del stack
            gc.collect()
        for label, paths in sources.items():
            if source_hashes[label] != {"config": sha256_file(paths[0]), "checkpoint": sha256_file(paths[1])}:
                raise ValueError("source config/checkpoint changed during diagnostic")
        report.update(status="completed", sources_unchanged=True, wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(json.dumps({"report": str(destination), "status": report["status"], "states": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
