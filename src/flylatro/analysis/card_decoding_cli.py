"""Collect frozen responses and fit diagnostic-only, run-heldout card probes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from flylatro.analysis.card_decoding import (
    VERSION, ALPHAS, check_partitions, labels_for, primary_assessment, probe, select_states,
)
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.analysis.card_window_pilot import write_json
from flylatro.analysis.corpus import build_calibration_corpus
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.choice_credit_pilot import implementation_hash
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_environment, build_plastic_stack, resolve_path
from flylatro.seeds import SeedPlan, derive_seed


def prepare(config, checkpoint, output, *, counts=(192, 64, 64), repeats=4, permutations=199):
    if output.exists():
        raise FileExistsError(output)
    if min(counts) < 2 or repeats < 2 or permutations < 1:
        raise ValueError("positive counts, at least two runs/repeats and one permutation required")
    if (config.fly.backend != "flywire" or config.fly.mode != "mbon_direct"
            or config.environment.backend != "balatro_sim" or config.environment.num_envs != 3
            or config.fly.duration_ms != 100 or config.environment.allow_skip_blinds
            or config.reinforcement.run_failure_pulse != .01
            or config.plasticity.version != "three-factor-choice-gated-v1"):
        raise ValueError("probe requires original three-learner choice-credit 100 ms, no-skip, failure 0.01 source")
    env = build_plastic_environment(replace(config, environment=replace(config.environment, num_envs=1)))
    output.mkdir(parents=True)
    records, observations, corpora = {}, {}, {}
    for part_index, (part, count) in enumerate(zip(("train", "validation", "test"), counts)):
        # Root streams also remain disjoint. These are validation-range seeds;
        # 'test' means probe-heldout, never the locked final-test stream.
        seeds = SeedPlan().seeds("validation", 48 if part == "train" else 20,
                                 offset=20_000 + part_index * 100)
        corpus = build_calibration_corpus(env, environment_seeds=seeds, states_per_seed=64,
            navigation_seed=26_000_000, store_snapshots=False)
        records[part], observations[part] = select_states(corpus, count, seed=26_001_000 + part_index,
            excluded_runs=[row["run_seed"] for rows in records.values() for row in rows],
            excluded_hashes=[row["state_hash"] for rows in records.values() for row in rows])
        corpus.save(output / f"{part}-corpus.npz")
        corpora[part] = {"sha256": corpus.sha256, "environment_seeds": list(seeds)}
        print(json.dumps({"stage": "prepared", "partition": part, "selected_runs": count}), flush=True)
    check_partitions(records)
    sources = {"config": config.source_path.resolve(), "checkpoint": checkpoint.resolve(),
               "motor": resolve_path(config, config.fly.motor_mapping_path),
               "artifact": resolve_path(config, config.fly.artifact_path)}
    declaration = {"version": VERSION, "implementation_sha256": implementation_hash(),
        "sources": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in sources.items()},
        "corpora": corpora, "records": records, "repeats": repeats, "permutations": permutations,
        "duration_ms": 100, "alphas": list(ALPHAS), "activity_seed": 26_010_000,
        "probe_seed": 26_020_000, "final_test_seeds_used": False,
        "fit": "train-only standardization and slot-specific multi-output ridge; alpha selected on validation balanced accuracy separately by task/population; repeats averaged before fitting; no refit on validation",
        "null": "shuffle whole eight-card label blocks across training runs; repeat validation alpha selection for each null; common permutations across populations/learners",
        "primary": "six Holm-corrected tests: rank/suit at sensory, KC, MBON; KC/MBON statistics averaged across three trained learners; above chance and training-only slot-majority baseline",
        "controls": "direct visible-identity one-hot positive control; uniform chance; train-only slot majority; run-label permutations; initial unit weights; same neural seeds across initial/trained; exact response duplicates",
        "limitations": "one permuted eight-card PLAY observation per game run; visible-input intervention can disagree with simulator order and aggregates; probe has privileged diagnostic labels and is never an agent; mean rates only; no gameplay claims; three trained learners"}
    write_json(output / "declaration.json", declaration)
    return declaration, observations


def check_sources(declaration):
    if implementation_hash() != declaration["implementation_sha256"]:
        raise ValueError("probe implementation changed during execution")
    for source in declaration["sources"].values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("probe source changed during execution")


def run(config, checkpoint, output, declaration, observations):
    started = time.perf_counter()
    report = {"version": VERSION, "status": "running", "learning_disabled": True,
              "diagnostic_only": True, "declaration_sha256": sha256_file(output / "declaration.json"),
              "responses": {}, "probes": {}}
    save = lambda: write_json(output / "report.json", report)
    save()
    try:
        check_sources(declaration)
        stack = build_plastic_stack(config)
        load_plastic_checkpoint(checkpoint, stack.trainer, expected_components=stack.components,
                                require_matching_exposure_budget=False)
        state, processor = stack.agent.plasticity.state, stack.agent.processor
        before = plastic_hashes(state)
        if not np.all(state_numpy(state.initial_efficacy) == 1):
            raise ValueError("initial positive control requires unit initial efficacies")
        report.update(source_components=stack.components, plastic_state_hashes_before=before)
        labels, direct, sensory = {}, {}, {}
        for part, obs_list in observations.items():
            labels[part], direct[part] = labels_for(obs_list)
            sensory[part] = np.stack([np.asarray(processor.encoder.encode(obs).rates_hz)[0,
                processor.encoder.mapping.available_alpn_indices].copy() for obs in obs_list])
        label_file = output / "labels-and-inputs.npz"
        np.savez_compressed(label_file, **{f"{part}_{task}": values for part, tasks in labels.items() for task, values in tasks.items()},
                            **{f"{part}_{stage}": values for stage, parts in (("direct", direct), ("sensory", sensory)) for part, values in parts.items()})
        report["inputs"] = {"file": label_file.name, "sha256": sha256_file(label_file),
                            "sensory_mapping_sha256": processor.encoder.mapping.sha256}
        cache = {}
        exact = 0
        for weight_label, learner in (("initial", 0), ("trained", 0), ("trained", 1), ("trained", 2)):
            key = f"{weight_label}-{learner}"
            efficacy = (state.initial_efficacy if weight_label == "initial" else state.efficacy)[learner:learner + 1]
            arrays = {}
            for part, obs_list in observations.items():
                samples = {"kc": [], "mbon": []}
                seeds = []
                for index, obs in enumerate(obs_list):
                    responses = []
                    for repeat in range(declaration["repeats"]):
                        seed = derive_seed("offline-card-probe", declaration["activity_seed"], part, index, repeat)
                        def sample():
                            response = processor.process(obs, fly_seeds=(seed,), efficacy=efficacy)
                            return {"kc": np.asarray(response.kc_activity)[0].copy(),
                                    "mbon": np.asarray(response.mbon_activity)[0].copy()}
                        response = sample()
                        if repeat == 0:
                            duplicate = sample()
                            if any(not np.array_equal(response[name], duplicate[name]) for name in response):
                                raise ValueError("exact neural duplicate failed")
                            exact += 1
                        if any(not np.isfinite(a).all() or (a < 0).any() for a in response.values()):
                            raise ValueError("invalid frozen neural rates")
                        responses.append(response)
                        seeds.append(seed)
                    for stage in samples:
                        samples[stage].append(np.stack([row[stage] for row in responses]))
                    if (index + 1) % 16 == 0 or index + 1 == len(obs_list):
                        print(json.dumps({"stage": "neural", "weights": key, "partition": part,
                                          "completed": index + 1, "total": len(obs_list)}), flush=True)
                arrays.update({f"{part}_{stage}": np.stack(values) for stage, values in samples.items()})
                arrays[part + "_neural_seeds"] = np.asarray(seeds, dtype=np.int64).reshape(len(obs_list), -1)
            path = output / f"{key}-responses.npz"
            np.savez_compressed(path, **arrays, kc_root_ids=processor.kc_root_ids, mbon_root_ids=processor.mbon_root_ids)
            report["responses"][key] = {"file": path.name, "sha256": sha256_file(path),
                "axes": ["game_run", "neural_repeat", "neuron"],
                "health": {stage: {"active_fraction": float(np.mean(arrays["test_" + stage] > 0)),
                                   "high_rate_fraction": float(np.mean(arrays["test_" + stage] > 200))}
                           for stage in ("kc", "mbon")}}
            cache[key] = {stage: {part: arrays[f"{part}_{stage}"].mean(axis=1) for part in observations}
                          for stage in ("kc", "mbon")}
            if plastic_hashes(state) != before:
                raise ValueError("frozen probe changed plastic state")
            save()
        report.update(exact_repeat_checks=exact, plastic_state_hashes_after=plastic_hashes(state),
                      plastic_state_unchanged=True)
        fitted = {}
        datasets = {"direct": direct, "sensory": sensory,
                    **{f"{key}-{stage}": parts for key, stages in cache.items() for stage, parts in stages.items()}}
        for key, parts in datasets.items():
            print(json.dumps({"stage": "offline-probe", "population": key}), flush=True)
            metrics, predictions = probe(*(parts[part] for part in ("train", "validation", "test")), labels,
                permutations=declaration["permutations"], seed=declaration["probe_seed"])
            path = output / f"{key}-probe.npz"
            np.savez_compressed(path, **predictions)
            report["probes"][key] = {"metrics": metrics, "file": path.name, "sha256": sha256_file(path)}
            fitted[key] = {"metrics": metrics, "arrays": predictions}
            save()
        assessment = primary_assessment(fitted)
        assessment["direct_positive_control_pass"] = all(fitted["direct"]["metrics"][task]["accuracy"] >= .98 for task in ("rank", "suit"))
        assessment["comparability"] = "PASS" if assessment["direct_positive_control_pass"] else "FAIL"
        if not assessment["direct_positive_control_pass"]:
            for row in assessment["primary_tests"]:
                row["above_controls"] = False
        write_json(output / "assessment.json", assessment)
        check_sources(declaration)
        report.update(status="completed", sources_unchanged=True, wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("artefacts/choice-credit-20261016-pilot-v1/configs/choice.toml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artefacts/choice-credit-20261016-pilot-v1/choice/plastic-checkpoint-final.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-states", type=int, default=192)
    parser.add_argument("--validation-states", type=int, default=64)
    parser.add_argument("--test-states", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    config = PlasticExperimentConfig.load(args.config)
    if not args.prepare_only:
        config.require_heavy_opt_in(args.heavy)
    declaration, observations = prepare(config, args.checkpoint, args.output_dir,
        counts=(args.train_states, args.validation_states, args.test_states), repeats=args.repeats, permutations=args.permutations)
    if not args.prepare_only:
        run(config, args.checkpoint, args.output_dir, declaration, observations)
    print(json.dumps({"output_dir": str(args.output_dir), "prepared_only": args.prepare_only}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
