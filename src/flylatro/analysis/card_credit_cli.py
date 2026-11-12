"""Trace cached card signals through anatomy, coactivity and motor credit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from flylatro.analysis.card_credit import VERSION, STAGES, PathwayLayout, feature_matrices, weight_summary
from flylatro.analysis.card_decoding import check_partitions, labels_for, probe
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.analysis.card_window_pilot import write_json
from flylatro.analysis.corpus import CalibrationCorpus
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.learning.checkpoints import load_plastic_checkpoint
from flylatro.learning.choice_credit_pilot import implementation_hash
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack, payload_sha256


def verified_sources(source):
    report = json.loads((source / "report.json").read_text())
    declaration = json.loads((source / "declaration.json").read_text())
    if (report["status"] != "completed" or not report["plastic_state_unchanged"]
            or sha256_file(source / "declaration.json") != report["declaration_sha256"]):
        raise ValueError("completed, frozen, declaration-bound decoding cache required")
    if json.loads((source / "assessment.json").read_text())["comparability"] != "PASS":
        raise ValueError("source decoding positive control did not pass")
    check_partitions(declaration["records"])
    paths = [source / "report.json", source / "declaration.json", source / "assessment.json"]
    for row in [report["inputs"], *report["responses"].values()]:
        path = source / row["file"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("cached response or input hash mismatch")
        paths.append(path)
    for entry in declaration["sources"].values():
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("source model file changed")
        paths.append(path)
    masks = {}
    with np.load(source / report["inputs"]["file"]) as cached_labels:
        for part, rows in declaration["records"].items():
            path = source / f"{part}-corpus.npz"
            corpus = CalibrationCorpus.load(path)
            if corpus.sha256 != declaration["corpora"][part]["sha256"]:
                raise ValueError("source corpus identity mismatch")
            obs_list = []
            masks[part] = {key: [] for key in corpus.masks}
            for row in rows:
                i = row["state_index"]
                if (row["state_hash"] != corpus.state_hashes[i]
                        or row["run_seed"] != corpus.source_run_seed[i]):
                    raise ValueError("source selection provenance mismatch")
                obs, legal = corpus.row(i)
                obs = {key: value.copy() for key, value in obs.items()}
                obs["hand"][0, :8] = obs["hand"][0, row["permutation"]].copy()
                obs_list.append(obs)
                for key in masks[part]:
                    masks[part][key].append(legal[key][0].copy())
            labels, _ = labels_for(obs_list)
            for task in ("rank", "suit"):
                if not np.array_equal(labels[task], cached_labels[f"{part}_{task}"]):
                    raise ValueError("cached identity labels do not reproduce selected input")
            masks[part] = {key: np.stack(value) for key, value in masks[part].items()}
            paths.extend((path, path.with_suffix(path.suffix + ".manifest.json")))
    return report, declaration, masks, paths


def assess(probes):
    tests = []
    for stage in STAGES:
        for task in ("rank", "suit"):
            rows = [probes[f"trained-{i}-{stage}"] for i in range(3)]
            observed = float(np.mean([row["metrics"][task]["balanced_accuracy"] for row in rows]))
            baseline = float(np.mean([row["metrics"][task]["slot_majority"]["balanced_accuracy"] for row in rows]))
            null = np.mean([row["arrays"][task + "_null"] for row in rows], axis=0)
            tests.append({"stage": stage, "task": task, "balanced_accuracy": observed,
                "slot_majority_balanced_accuracy": baseline,
                "p": float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))})
    previous = 0.
    for index, row in enumerate(sorted(tests, key=lambda row: row["p"])):
        previous = max(previous, min(1., row["p"] * (len(tests) - index)))
        row["holm_p"] = previous
        row["above_controls"] = bool(previous <= .05 and row["balanced_accuracy"] > max(
            1 / (13 if row["task"] == "rank" else 4), row["slot_majority_balanced_accuracy"]))
    return {"exploratory_tests": tests,
            "limitations": "follow-up on reused probe-heldout contexts; not independent confirmation. Static afferent averages omit temporal integration, other inputs and inhibition. Fresh eligibility increments omit historical traces, actual reward, exploration and gameplay outcomes. No causal learning bottleneck established."}


def run(source, output, *, permutations=499, heavy=False):
    if output.exists():
        raise FileExistsError(output)
    if permutations < 1:
        raise ValueError("at least one permutation required")
    cached, original, masks, paths = verified_sources(source)
    config = PlasticExperimentConfig.load(Path(original["sources"]["config"]["path"]))
    config.require_heavy_opt_in(heavy)
    checkpoint = Path(original["sources"]["checkpoint"]["path"])
    identity = {str(path.resolve()): sha256_file(path) for path in paths}
    output.mkdir(parents=True)
    declaration = {"version": VERSION, "implementation_sha256": implementation_hash(),
        "source_files": identity, "source_dir": str(source.resolve()), "permutations": permutations,
        "stages": list(STAGES), "fit_policy": "same train/tune/test run partitions and ridge method as source; run-label-shuffle fits; Holm over 16 exploratory stage/task tests averaged over three trained learners",
        "card_population": "all ten fixed card pools, including two currently unavailable slots; labels cover the eight visible slots; exposure reports include all ten slots",
        "coactivity": "per-repeat clip(KC/20,0,1)*clip(MBON/20,0,1), averaged across repeats after applying max-eligibility bound; new increments from zero traces, with and without the existing union-of-selected-heads gate",
        "weights": "original strictly verified checkpoint; initial and all three trained learners; no updates or forward simulation",
        "scope": "exploratory mechanistic cache analysis; original held-out contexts reused; no new independent confirmation; no agent modifications"}
    write_json(output / "declaration.json", declaration)
    report = {"version": VERSION, "status": "running", "learning_disabled": True,
              "declaration_sha256": sha256_file(output / "declaration.json"), "learners": {}, "probes": {}}
    save = lambda: write_json(output / "report.json", report)
    save()
    started = time.perf_counter()
    try:
        stack = build_plastic_stack(config)
        load_plastic_checkpoint(checkpoint, stack.trainer, expected_components=stack.components,
                                require_matching_exposure_budget=False)
        if payload_sha256(stack.components) != payload_sha256(cached["source_components"]):
            raise ValueError("source response components differ from loaded stack")
        state = stack.agent.plasticity.state
        before = plastic_hashes(state)
        if before != cached["plastic_state_hashes_after"]:
            raise ValueError("cached responses do not match loaded frozen plastic state")
        report["plastic_state_hashes_before"] = before
        state_initial, state_trained = state_numpy(state.initial_efficacy), state_numpy(state.efficacy)
        labels = {}
        with np.load(source / cached["inputs"]["file"]) as data:
            for part in masks:
                labels[part] = {task: data[f"{part}_{task}"].copy() for task in ("rank", "suit")}
        fitted = {}
        for key, entry in cached["responses"].items():
            with np.load(source / entry["file"]) as data:
                layout = PathwayLayout(stack.agent.plasticity.topology, stack.agent.motor,
                                       data["kc_root_ids"], data["mbon_root_ids"])
                if layout.credit.manifest != stack.agent.credit.manifest:
                    raise ValueError("offline credit assignment differs from agent")
                if "anatomy" not in report:
                    report["anatomy"] = layout.structural_summary()
                index = int(key.split("-")[-1])
                efficacy = state_initial[index] if key.startswith("initial") else state_trained[index]
                features, diagnostics, derived, diagnostic_arrays = {}, {}, {}, {}
                for part in masks:
                    values, diagnostics[part], by_run, actions = feature_matrices(layout,
                        data[f"{part}_kc"], data[f"{part}_mbon"], efficacy, masks[part], config.plasticity)
                    for stage, value in values.items():
                        features.setdefault(stage, {})[part] = value
                        derived[f"{part}_{stage}"] = value
                    diagnostic_arrays.update({f"{part}_{name}": value for name, value in by_run.items()})
                    diagnostic_arrays.update({f"{part}_action_{name}": value for name, value in actions.items()})
            feature_path = output / f"{key}-derived.npz"
            np.savez_compressed(feature_path, **derived, **diagnostic_arrays,
                                card_edge_indices=layout.topology.edge_indices[layout.card_edges],
                                connected_kc_columns=layout.connected_kc, card_mbon_columns=layout.card_posts)
            report["learners"][key] = {"diagnostics": diagnostics, "features_file": feature_path.name,
                "features_sha256": sha256_file(feature_path), "weight_changes": weight_summary(layout, state_initial[index], efficacy)}
            save()
            for stage in STAGES:
                print(json.dumps({"weights": key, "stage": stage}), flush=True)
                parts = features[stage]
                metrics, arrays = probe(*(parts[part] for part in ("train", "validation", "test")), labels,
                    permutations=permutations, seed=26_020_000)
                name = f"{key}-{stage}"
                path = output / f"{name}-probe.npz"
                np.savez_compressed(path, **arrays)
                report["probes"][name] = {"metrics": metrics, "file": path.name, "sha256": sha256_file(path)}
                fitted[name] = {"metrics": metrics, "arrays": arrays}
                save()
            if plastic_hashes(state) != before:
                raise ValueError("cached credit audit changed plastic state")
        assessment = assess(fitted)
        write_json(output / "assessment.json", assessment)
        if implementation_hash() != declaration["implementation_sha256"]:
            raise ValueError("credit-audit implementation changed during execution")
        if any(sha256_file(Path(path)) != digest for path, digest in identity.items()):
            raise ValueError("credit-audit source changed during execution")
        report.update(status="completed", source_files_unchanged=True, plastic_state_unchanged=True,
                      plastic_state_hashes_after=plastic_hashes(state), wall_seconds=time.perf_counter() - started)
        save()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("artefacts/card-decoding-20261024-v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--heavy", action="store_true")
    args = parser.parse_args(argv)
    run(args.source_dir, args.output_dir, permutations=args.permutations, heavy=args.heavy)
    print(json.dumps({"output_dir": str(args.output_dir), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
