"""Declared frozen integration-window interventions and paired comparisons."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from flylatro.analysis.card_information import CONDITIONS
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import payload_sha256


VERSION = "frozen-card-window-comparison-v1"
FOLLOW_UP_RULE = (
    "healthy unchanged plastic state in both arms; in a strict majority of choice learners, "
    "both rank and suit have lower KC/card-contrast repeat RMS and positive card-contrast "
    "effect/noise power ratios exceeding baseline; mean choice ratio improves for both edits "
    "and mean unchanged-input top-card agreement does not decrease"
)


@contextmanager
def integration_window(processor, duration_ms):
    """Call only after strict checkpoint verification at its original duration."""
    dt = float(processor.backend.parameters.dt_ms)
    duration = float(duration_ms)
    if (not math.isfinite(duration) or duration <= 0 or not math.isfinite(dt) or dt <= 0
            or round(duration / dt) < 1
            or not math.isclose(duration / dt, round(duration / dt), rel_tol=0, abs_tol=1e-7)):
        raise ValueError("analysis duration must be a positive finite whole number of simulation steps")
    original = processor.duration_ms
    processor.duration_ms = duration
    try:
        yield
    finally:
        processor.duration_ms = original


def _require_equal(left, right, label):
    if left != right:
        raise ValueError(f"window reports differ in {label}")


def artifact_summary(directory, report, arm, learner):
    """Audit saved arrays and summarize health and unchanged-input preferences."""
    result = {}
    cases = report["cases"]
    for label, weights in learner["weights"].items():
        path = directory / weights["activity_file"]
        if sha256_file(path) != weights["activity_sha256"]:
            raise ValueError("neural activity hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            health = {}
            for stage, minimum in (("kc", arm["manifest"]["experiment_config"]["safety"]["minimum_kc_active_fraction"]),
                                   ("card_pool_neurons", arm["manifest"]["experiment_config"]["safety"]["minimum_motor_active_fraction"])):
                rates = arrays[stage]
                if not rates.size or not np.isfinite(rates).all() or np.any(rates < 0):
                    raise ValueError("invalid neural rates in window activity")
                active = np.mean(rates > 0, axis=-1)
                high = np.mean(rates > 200, axis=-1)
                health[stage] = {"minimum_active_fraction": float(active.min()),
                                 "maximum_high_rate_fraction": float(high.max()),
                                 "mean_rate_hz": float(rates.mean()),
                                 "healthy": bool(active.min() >= minimum and high.max() <= .25)}
            scores = arrays["card_scores"]
            if scores.shape[:3] != (len(cases), len(CONDITIONS), report["repeats"]) or not np.isfinite(scores).all():
                raise ValueError("invalid card-score array shape or values")
            agreements = []
            seen_states = set()
            for index, case in enumerate(cases):
                if case["state_index"] in seen_states:
                    continue
                seen_states.add(case["state_index"])
                mask = np.asarray(case["legal_card_mask"])
                preferred = np.argmax(np.where(mask, scores[index, 0], -np.inf), axis=1)
                agreements.append(float(np.mean([preferred[a] == preferred[b]
                                  for a, b in combinations(range(len(preferred)), 2)])))
        result[label] = {"health": health, "healthy": all(row["healthy"] for row in health.values()),
                         "same_input_top_card_repeat_agreement": float(np.mean(agreements)),
                         "summary": weights["summary"]}
    return result


def validate_pair(base, candidate):
    for report in (base, candidate):
        if report.get("status") != "completed" or not report.get("learning_disabled") or not report.get("sources_unchanged"):
            raise ValueError("window comparison requires completed frozen reports")
    for key in ("version", "source_hashes", "corpus_sha256", "seed", "repeats", "conditions", "population_units", "cases"):
        _require_equal(base[key], candidate[key], key)
    _require_equal(list(base["arms"]), list(candidate["arms"]), "arms")
    for key in ("mapping_sha256", "records", "array_axes"):
        _require_equal(base["sensory"][key], candidate["sensory"][key], "sensory inputs")
    for report in (base, candidate):
        if len({arm["analysis_window"]["duration_ms"] for arm in report["arms"].values()}) != 1:
            raise ValueError("arms within a window have different durations")
    for label, left in base["arms"].items():
        right = candidate["arms"][label]
        for key in ("config", "checkpoint", "manifest", "components"):
            _require_equal(left[key], right[key], key)
        for arm in (left, right):
            window = arm["analysis_window"]
            if payload_sha256(window["dynamics"]) != window["dynamics_sha256"]:
                raise ValueError("analysis dynamics hash mismatch")
            if window["dynamics"]["decision_duration_ms"] != window["duration_ms"]:
                raise ValueError("analysis duration differs from its dynamics")
        old, new = left["analysis_window"], right["analysis_window"]
        if new["duration_ms"] <= old["duration_ms"]:
            raise ValueError("candidate duration must exceed baseline")
        for key in ("training_duration_ms", "normalization_policy", "checkpoint_policy"):
            _require_equal(old[key], new[key], key)
        old_dynamics, new_dynamics = dict(old["dynamics"]), dict(new["dynamics"])
        old_dynamics.pop("decision_duration_ms")
        new_dynamics.pop("decision_duration_ms")
        _require_equal(old_dynamics, new_dynamics, "non-duration dynamics")
        _require_equal([l["learner_index"] for l in left["learners"]],
                       [l["learner_index"] for l in right["learners"]], "learner indices")
        for first, second in zip(left["learners"], right["learners"]):
            for result in (first, second):
                if (not result.get("plastic_state_unchanged")
                        or result["plastic_state_hashes_before"] != result["plastic_state_hashes_after"]):
                    raise ValueError("window diagnostic changed plastic state")
            for key in ("plastic_state_hashes_before", "population_root_ids", "exact_repeat_checks"):
                _require_equal(first[key], second[key], key)
            for weight_label in ("initial", "trained"):
                first_rows = first["weights"][weight_label]["rows"]
                second_rows = second["weights"][weight_label]["rows"]
                _require_equal(len(first_rows), len(second_rows), "trial count")
                for a, b in zip(first_rows, second_rows):
                    for key in ("case_index", "state_index", "slot", "condition", "neural_seeds"):
                        _require_equal(a[key], b[key], key)


def compare_windows(base_dir, candidate_dir):
    base = json.loads((base_dir / "report.json").read_text())
    candidate = json.loads((candidate_dir / "report.json").read_text())
    validate_pair(base, candidate)
    paths = [directory / report["sensory"]["activity_file"] for directory, report in ((base_dir, base), (candidate_dir, candidate))]
    for path, report in zip(paths, (base, candidate)):
        if sha256_file(path) != report["sensory"]["activity_sha256"]:
            raise ValueError("sensory activity hash mismatch")
    with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
        _require_equal(a.files, b.files, "sensory array keys")
        if any(not np.array_equal(a[key], b[key]) for key in a.files):
            raise ValueError("window sensory activity differs")
    comparisons, healthy = {}, True
    for label, arm in base["arms"].items():
        rows = []
        for first, second in zip(arm["learners"], candidate["arms"][label]["learners"]):
            a = artifact_summary(base_dir, base, arm, first)
            b = artifact_summary(candidate_dir, candidate, candidate["arms"][label], second)
            healthy &= all(weights["healthy"] for weights in (*a.values(), *b.values()))
            row = {"learner_index": first["learner_index"], "baseline": a, "candidate": b, "edit_signals": {}}
            for kind in ("rank", "suit"):
                x, y = [results["trained"]["summary"][kind]["populations"] for results in (a, b)]
                old_ratio = x["card_contrasts"]["ratio_of_mean_effect_to_noise_power"]
                new_ratio = y["card_contrasts"]["ratio_of_mean_effect_to_noise_power"]
                row["edit_signals"][kind] = bool(new_ratio is not None and old_ratio is not None
                    and new_ratio > max(0, old_ratio)
                    and y["kc"]["baseline_repeat_rms"] < x["kc"]["baseline_repeat_rms"]
                    and y["card_contrasts"]["baseline_repeat_rms"] < x["card_contrasts"]["baseline_repeat_rms"])
            row["both_edit_signal"] = all(row["edit_signals"].values())
            rows.append(row)
        comparisons[label] = rows
    primary = comparisons["choice"]
    majority = sum(row["both_edit_signal"] for row in primary) > len(primary) / 2
    def mean_improves(kind):
        values = [[row[side]["trained"]["summary"][kind]["populations"]["card_contrasts"]["ratio_of_mean_effect_to_noise_power"]
                   for row in primary] for side in ("baseline", "candidate")]
        return all(value is not None for side in values for value in side) and np.mean(values[1]) > np.mean(values[0])
    mean_signal_improves = all(mean_improves(kind) for kind in ("rank", "suit"))
    mean_agreement = [float(np.mean([row[side]["trained"]["same_input_top_card_repeat_agreement"] for row in primary]))
                      for side in ("baseline", "candidate")]
    duration = lambda report: next(iter(report["arms"].values()))["analysis_window"]["duration_ms"]
    return {"version": VERSION, "comparability": "PASS", "baseline_duration_ms": duration(base),
            "candidate_duration_ms": duration(candidate), "all_health_checks_pass": bool(healthy),
            "primary_arm": "choice", "learners_with_both_edit_signal": sum(row["both_edit_signal"] for row in primary),
            "mean_signal_improves_for_both_edits": bool(mean_signal_improves),
            "mean_original_top_card_agreement": {"baseline": mean_agreement[0], "candidate": mean_agreement[1]},
            "predeclared_follow_up_signal": bool(healthy and majority and mean_signal_improves and mean_agreement[1] >= mean_agreement[0]),
            "criterion": FOLLOW_UP_RULE, "arms": comparisons,
            "interpretation": "descriptive screen for a further experiment; no hand-quality, held-out gameplay or learning claim"}


def audit_reference(current_dir, reference_dir):
    """Reproduce the earlier 100 ms experiment without depending on ZIP metadata."""
    current = json.loads((current_dir / "report.json").read_text())
    reference = json.loads((reference_dir / "report.json").read_text())
    if current["status"] != "completed" or reference["status"] != "completed":
        raise ValueError("reference audit requires completed reports")
    for key in ("version", "source_hashes", "corpus_sha256", "seed", "repeats", "cases", "conditions"):
        _require_equal(current[key], reference[key], key)
    checked = 0
    for label, arm in current["arms"].items():
        if arm["analysis_window"]["duration_ms"] != arm["analysis_window"]["training_duration_ms"]:
            raise ValueError("reference audit requires the original training duration")
        original = reference["arms"][label]
        _require_equal(arm["components"], original["components"], "reference components")
        _require_equal([row["learner_index"] for row in arm["learners"]],
                       [row["learner_index"] for row in original["learners"]], "reference learners")
        for first, second in zip(arm["learners"], original["learners"]):
            _require_equal(first["plastic_state_hashes_before"], second["plastic_state_hashes_before"], "reference plastic state")
            for weight in ("initial", "trained"):
                paths = []
                for directory, result in ((current_dir, first), (reference_dir, second)):
                    entry = result["weights"][weight]
                    path = directory / entry["activity_file"]
                    if sha256_file(path) != entry["activity_sha256"]:
                        raise ValueError("reference neural activity hash mismatch")
                    paths.append(path)
                with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
                    _require_equal(a.files, b.files, "reference array keys")
                    if any(not np.array_equal(a[key], b[key]) for key in a.files):
                        raise ValueError("original-duration responses differ from reference")
                checked += 1
    return {"status": "PASS", "neural_files_compared": checked,
            "reference_report_sha256": sha256_file(reference_dir / "report.json"),
            "interpretation": "all original-duration neural arrays exactly reproduce prior experiment"}
