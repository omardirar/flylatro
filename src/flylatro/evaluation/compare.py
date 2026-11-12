"""Descriptive paired held-out comparisons, separate from replay controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from flylatro.fly.flywire_artifact import sha256_file


PAIRED_EVALUATION_VERSION = "paired-frozen-plastic-evaluation-v1"
MATCHED_FIELDS = (
    "artifact_sha256", "population_sha256", "plastic_topology_sha256",
    "sensory_mapping_sha256", "motor_mapping_sha256", "fly_dynamics_sha256",
    "reinforcement_mapping_sha256", "plasticity_rule_sha256",
    "canonical_motor_candidate_set_sha256", "simulator_version", "output_mode",
    "fly_backend",
)


def compare_frozen_evaluations(trained: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    for name, summary in (("trained", trained), ("control", control)):
        if summary.get("version") != "frozen-plastic-evaluation-v2":
            raise ValueError(f"{name} needs an explicit v2 frozen evaluation summary")
        if summary.get("learning_disabled") is not True or summary.get("training_action_schedule_replayed") is not False:
            raise ValueError(f"{name} must use frozen weights and independent actions")
        if summary.get("action_policy") != "independent-fixed-motor-choices-no-training-action-replay":
            raise ValueError(f"{name} has an incompatible action policy")
    for key in ("win_ante", "seed_stream", "seeds"):
        if trained.get(key) != control.get(key):
            raise ValueError(f"paired evaluation {key} differs")
    for key in MATCHED_FIELDS:
        if key not in trained["components"] or key not in control["components"]:
            raise ValueError(f"paired evaluation lacks model field {key}")
        if trained["components"][key] != control["components"][key]:
            raise ValueError(f"paired evaluation model field {key} differs")
    if trained["components"].get("action_constraints", {}) != control["components"].get("action_constraints", {}):
        raise ValueError("paired evaluation action constraints differ")
    seeds = trained["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("paired evaluation needs nonempty unique seeds")
    paired = []
    for name, summary in (("trained", trained), ("control", control)):
        episodes = summary["episodes"]
        if len(episodes) != len(seeds) or [episode["seed"] for episode in episodes] != seeds:
            raise ValueError(f"{name} episode rows do not match the declared seeds")
    for left, right in zip(trained["episodes"], control["episodes"], strict=True):
        if left.get("balatro_seed") != right.get("balatro_seed"):
            raise ValueError("paired evaluation simulator run seeds differ")
        numeric = [left[key] for key in ("ante", "decisions", "episode_return")]
        numeric += [right[key] for key in ("ante", "decisions", "episode_return")]
        numeric += [row["score"] for row in (left, right) if row.get("score") is not None]
        if not np.isfinite(numeric).all():
            raise ValueError("paired evaluation episode metrics must be finite")
        paired.append({
            "seed": left["seed"], "balatro_seed": left.get("balatro_seed"),
            "trained_won": bool(left["won"]), "control_won": bool(right["won"]),
            "win_delta": int(left["won"]) - int(right["won"]),
            "ante_delta": left["ante"] - right["ante"],
            "return_delta": left["episode_return"] - right["episode_return"],
            "decisions_delta": left["decisions"] - right["decisions"],
            "score_delta": (left["score"] - right["score"]
                            if left.get("score") is not None and right.get("score") is not None else None),
        })
    return {
        "version": PAIRED_EVALUATION_VERSION,
        "comparison_semantics": "same held-out seeds; frozen weights; each fly chooses its own actions",
        "win_ante": trained["win_ante"], "seed_stream": trained["seed_stream"], "seeds": seeds,
        "episodes": len(seeds), "paired_episodes": paired,
        "trained_win_rate": float(np.mean([row["trained_won"] for row in paired])),
        "control_win_rate": float(np.mean([row["control_won"] for row in paired])),
        "win_rate_delta": float(np.mean([row["win_delta"] for row in paired])),
        "mean_return_delta": float(np.mean([row["return_delta"] for row in paired])),
        "mean_ante_delta": float(np.mean([row["ante_delta"] for row in paired])),
        "mean_episode_length_delta": float(np.mean([row["decisions_delta"] for row in paired])),
        "scored_episode_pairs": sum(row["score_delta"] is not None for row in paired),
        "trained_only_wins": sum(row["trained_won"] and not row["control_won"] for row in paired),
        "control_only_wins": sum(row["control_won"] and not row["trained_won"] for row in paired),
        "interpretation": "descriptive pilot comparison; PASS means comparability, not learning improvement",
        "trained_plastic_weight_sha256": trained["plastic_weight_sha256"],
        "control_plastic_weight_sha256": control["plastic_weight_sha256"],
        "model_identity": {key: trained["components"][key] for key in MATCHED_FIELDS},
        "gates": {"status": "PASS", "failed": []},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path, required=True, help="trained evaluation summary.json")
    parser.add_argument("--control", type=Path, required=True, help="untrained evaluation summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"comparison output already exists: {args.output}")
    provenance = {
        "trained_summary_sha256": sha256_file(args.trained),
        "control_summary_sha256": sha256_file(args.control),
        "trained_summary": str(args.trained.resolve()), "control_summary": str(args.control.resolve()),
    }
    try:
        report = compare_frozen_evaluations(json.loads(args.trained.read_text()), json.loads(args.control.read_text()))
    except (ValueError, KeyError) as error:
        report = {"version": PAIRED_EVALUATION_VERSION, "gates": {"status": "FAIL", "failed": ["incomparable_evaluations"]}, "error": str(error)}
    report["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["gates"]["status"], "win_rate_delta": report.get("win_rate_delta")}))
    return 0 if report["gates"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
