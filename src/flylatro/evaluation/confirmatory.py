"""Predeclared seed-block paired inference for three frozen training replicates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from flylatro.evaluation.compare import compare_frozen_evaluations
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.seeds import SeedPlan


def paired_seed_block_analysis(differences: np.ndarray, *, bootstrap_seed: int = 910010,
                              primary_fly_indices: np.ndarray | None = None) -> dict[str, Any]:
    """Rows are independent trained flies; columns are identical held-out seeds.

    Inference conditions on these trained flies. Seeds, rather than individual
    fly/seed outcomes, are resampling blocks, so a shared seed is never counted
    as three independent observations. Before observing outcomes, assign one
    trained fly to each seed. Its binary paired difference supports the exact
    McNemar test for equal win probability without a symmetry assumption about
    differences between a three-fly average and a shared binary baseline.
    """
    values = np.asarray(differences)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 1:
        raise ValueError("analysis requires exactly three flies and at least one seed")
    if not np.isfinite(values).all() or not np.isin(values, (-1, 0, 1)).all():
        raise ValueError("paired win differences must be -1, 0 or 1")
    if primary_fly_indices is None:
        primary_fly_indices = np.random.default_rng(910011).integers(0, 3, size=values.shape[1])
    indices = np.asarray(primary_fly_indices)
    if indices.shape != (values.shape[1],) or not np.issubdtype(indices.dtype, np.integer) or not np.isin(indices, (0, 1, 2)).all():
        raise ValueError("primary fly assignment must select one of three flies per seed")
    primary = values[indices, np.arange(values.shape[1])].astype(np.int64)
    observed = int(primary.sum())
    distribution = {0: 1.0}
    for weight in np.abs(primary[primary != 0]):
        next_distribution: dict[int, float] = {}
        for total, probability in distribution.items():
            for signed in (-int(weight), int(weight)):
                next_distribution[total + signed] = next_distribution.get(total + signed, 0.0) + probability / 2
        distribution = next_distribution
    pvalue = sum(probability for total, probability in distribution.items() if total >= observed)
    blocks = values.mean(axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    samples = blocks[rng.integers(0, len(blocks), size=(20000, len(blocks)))].mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    primary_samples = primary[rng.integers(0, len(primary), size=(20000, len(primary)))].mean(axis=1)
    primary_lower, primary_upper = np.quantile(primary_samples, (0.025, 0.975))
    if not np.any(primary):
        # Zero observed discordance does not imply zero uncertainty. Exact
        # 95% upper bound on discordance probability also bounds |win delta|.
        radius = 1 - 0.05 ** (1 / len(primary))
        primary_lower, primary_upper = -radius, radius
    estimate = float(blocks.mean())
    passed = bool(estimate > 0 and primary.mean() > 0 and pvalue < 0.05 and primary_lower > 0)
    return {
        "flies": values.shape[0], "seed_blocks": values.shape[1],
        "paired_mean_win_difference": estimate,
        "per_fly_win_differences": values.mean(axis=1).tolist(),
        "primary_fly_assignment_seed": 910011,
        "primary_fly_indices": indices.tolist(),
        "primary_paired_mean_win_difference": float(primary.mean()),
        "primary_paired_95_ci": [float(primary_lower), float(primary_upper)],
        "primary_trained_only_wins": int(np.count_nonzero(primary > 0)),
        "primary_control_only_wins": int(np.count_nonzero(primary < 0)),
        "one_sided_exact_sign_permutation_p": float(pvalue),
        "paired_seed_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_draws": 20000, "bootstrap_seed": bootstrap_seed,
        "nonzero_seed_blocks": int(np.count_nonzero(primary)),
        "statistical_improvement_demonstrated": passed,
        "inference_scope": "held-out seed distribution conditional on these three trained flies; no population-level training-replicate claim",
        "uncertainty_limit": "Percentile bootstrap can degenerate for constant effects. Zero primary discordance uses an exact 95% discordance bound and p=1; it cannot establish improvement.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trained", action="append", type=Path, required=True)
    parser.add_argument("--control", action="append", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = json.loads(args.lock.read_text())
    if lock["analysis_module_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("confirmatory analysis code differs from the pre-evaluation lock")
    for path, digest in lock["evaluation_implementation_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("frozen evaluation code differs from the pre-evaluation lock")
    if "batch_validation" in lock:
        from flylatro.evaluation.batch_validation import qualified_batch_size
        certificate = Path(lock["batch_validation"])
        if sha256_file(certificate) != lock["batch_validation_sha256"]:
            raise ValueError("batch qualification differs from the pre-evaluation lock")
        if qualified_batch_size(certificate, lock["pairs"]) != lock["evaluation_batch_size"]:
            raise ValueError("batch qualification differs from locked execution size")
    pairing_path = Path(lock["pairing_predeclaration"])
    if sha256_file(pairing_path) != lock["pairing_predeclaration_sha256"]:
        raise ValueError("pairing declaration differs from its locked identity")
    pairing = json.loads(pairing_path.read_text())
    if lock["primary_fly_indices"] != pairing["primary_fly_indices"]:
        raise ValueError("primary fly assignment differs from the prospective declaration")
    if len(args.trained) != 3 or len(args.control) != 3:
        raise ValueError("exactly three declared primary/control pairs required")
    expected_seeds = list(SeedPlan().seeds("final_test", 128))
    comparisons = []
    for index, (trained_path, control_path) in enumerate(zip(args.trained, args.control, strict=True)):
        trained, control = json.loads(trained_path.read_text()), json.loads(control_path.read_text())
        pair = compare_frozen_evaluations(trained, control)
        if pair["seed_stream"] != "final_test" or pair["seeds"] != expected_seeds:
            raise ValueError("evaluation seeds differ from the locked final-test declaration")
        if pair["model_identity"] != lock["model_identity"]:
            raise ValueError("fixed model differs from the pre-evaluation lock")
        declared = lock["pairs"][index]
        for summary, role in ((trained, "trained"), (control, "control")):
            if summary.get("evaluation_batch_size", 1) != lock["evaluation_batch_size"]:
                raise ValueError(f"{role} execution batch size differs from the lock")
            if sha256_file(Path(declared[role + "_config"])) != declared[role + "_config_file_sha256"]:
                raise ValueError(f"{role} configuration file differs from its locked identity")
            if summary["config_sha256"] != declared[role + "_config_sha256"]:
                raise ValueError(f"{role} evaluation configuration differs from the lock")
            if summary["plastic_weight_sha256"] != declared[role + "_weight_sha256"]:
                raise ValueError(f"{role} weights differ from locked checkpoint")
            if summary["checkpoint_manifest"]["checkpoint_sha256"] != declared[role + "_checkpoint_sha256"]:
                raise ValueError(f"{role} evaluation checkpoint differs from locked checkpoint")
            if sha256_file(Path(declared[role + "_checkpoint"])) != declared[role + "_checkpoint_sha256"]:
                raise ValueError(f"{role} checkpoint differs from locked checkpoint")
        if trained["components"]["plasticity_rule_sha256"] != lock["plasticity_rule_sha256"]:
            raise ValueError("learning rule differs from locked configuration")
        comparisons.append(pair)
    if any(pair["model_identity"] != comparisons[0]["model_identity"] for pair in comparisons[1:]):
        raise ValueError("replicates must share the exact fixed model identity")
    effects = np.asarray([[episode["win_delta"] for episode in pair["paired_episodes"]] for pair in comparisons])
    result = paired_seed_block_analysis(effects, primary_fly_indices=np.asarray(lock["primary_fly_indices"]))
    result.update(version="confirmatory-frozen-ante1-analysis-v1", lock_sha256=sha256_file(args.lock),
                  comparisons=comparisons, summary_hashes={str(path): sha256_file(path) for path in (*args.trained, *args.control)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("statistical_improvement_demonstrated",
        "paired_mean_win_difference", "one_sided_exact_sign_permutation_p", "paired_seed_bootstrap_95_ci")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
