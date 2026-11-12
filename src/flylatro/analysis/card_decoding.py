"""Offline linear probes. These fitted objects never enter an agent or reward path."""

from __future__ import annotations

import numpy as np

from flylatro.analysis.card_information import visible_card
from flylatro.analysis.corpus import UNAVAILABLE_RUN_SEED
from flylatro.seeds import derive_seed

VERSION = "offline-run-heldout-card-decoding-v1"
TASKS = {"rank": 13, "suit": 4}
ALPHAS = (.01, .1, 1., 10.)


def select_states(corpus, count, *, seed, excluded_runs=(), excluded_hashes=()):
    """One original state per actual run; randomize eight visible card slots."""
    if count < 1 or not corpus.records_run_provenance:
        raise ValueError("positive count and actual run provenance required")
    runs, hashes = set(excluded_runs), set(excluded_hashes)
    candidates = []
    for index in range(len(corpus)):
        run, digest = corpus.source_run_seed[index], corpus.state_hashes[index]
        if run == UNAVAILABLE_RUN_SEED or run in runs or digest in hashes:
            continue
        obs, masks = corpus.row(index)
        if (int(obs["hand_len"][0]) != 8 or not masks["action_type_mask"][0, 0]
                or not masks["card_select_mask"][0, :8].all()
                or not all(visible_card(card) for card in obs["hand"][0, :8])):
            continue
        candidates.append(index)
        runs.add(run)
        hashes.add(digest)
    if len(candidates) < count:
        raise ValueError(f"need {count} distinct visible eight-card runs, found {len(candidates)}")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(candidates, count, replace=False).tolist())
    records, observations = [], []
    for index in chosen:
        obs, _ = corpus.row(index)
        obs = {key: value.copy() for key, value in obs.items()}
        permutation = np.random.default_rng(derive_seed("card-probe-permutation", seed, index)).permutation(8)
        obs["hand"][0, :8] = obs["hand"][0, permutation].copy()
        records.append({"state_index": index, "state_hash": corpus.state_hashes[index],
                        "run_seed": corpus.source_run_seed[index], "root_seed": corpus.source_seed[index],
                        "permutation": permutation.tolist()})
        observations.append(obs)
    return records, observations


def labels_for(observations):
    hands = np.stack([obs["hand"][0, :8, :17] for obs in observations])
    return {"rank": hands[:, :, :13].argmax(axis=2),
            "suit": hands[:, :, 13:17].argmax(axis=2)}, hands.reshape(len(hands), -1)


def check_partitions(records):
    for key in ("run_seed", "state_hash", "root_seed"):
        seen = set()
        for rows in records.values():
            values = {row[key] for row in rows}
            if seen & values:
                raise ValueError(f"probe partitions overlap in {key}")
            if key != "root_seed" and len(values) != len(rows):
                raise ValueError(f"probe repeats a {key}")
            seen.update(values)


def scores(labels, prediction, classes):
    truth, predicted = np.asarray(labels), np.asarray(prediction)
    if truth.shape != predicted.shape or truth.ndim != 2:
        raise ValueError("labels and predictions must be [run, slot]")
    support = np.bincount(truth.ravel(), minlength=classes)
    correct = np.bincount(truth.ravel(), weights=(truth == predicted).ravel(), minlength=classes)
    present = support > 0
    return {"accuracy": float(np.mean(truth == predicted)),
            "balanced_accuracy": float(np.mean(correct[present] / support[present])),
            "class_support": support.tolist(),
            "slot_accuracy": np.mean(truth == predicted, axis=0).tolist()}


class RidgeReadout:
    """Train-only standardization and dual ridge; validation chooses regularization."""

    def __init__(self, train, validation, test, alphas=ALPHAS):
        train, validation, test = [np.asarray(x, dtype=np.float64) for x in (train, validation, test)]
        if (any(x.ndim != 2 or not np.isfinite(x).all() for x in (train, validation, test))
                or len(train) < 2 or any(x.shape[1] != train.shape[1] for x in (validation, test))
                or not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas)):
            raise ValueError("finite matrices, matching features and positive ridge penalties required")
        mean, scale = train.mean(axis=0), train.std(axis=0)
        keep = scale > 1e-8
        self.features = int(keep.sum())
        # Zero-information populations still produce an honest intercept-only probe.
        x, v, t = [(a[:, keep] - mean[keep]) / scale[keep] for a in (train, validation, test)]
        width = max(self.features, 1)
        kernel = x @ x.T / width
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        self.maps = []
        self.alphas = tuple(alphas)
        for alpha in alphas:
            inverse = (eigenvectors / (np.maximum(eigenvalues, 0) + alpha)) @ eigenvectors.T
            self.maps.append((v @ x.T / width @ inverse, t @ x.T / width @ inverse))

    def predict(self, train_labels, validation_labels, classes):
        target = np.eye(classes)[train_labels].reshape(len(train_labels), -1)
        intercept = target.mean(axis=0)
        centered = target - intercept
        candidates = []
        for alpha, (validation_map, test_map) in zip(self.alphas, self.maps):
            prediction = (validation_map @ centered + intercept).reshape(-1, train_labels.shape[1], classes).argmax(axis=2)
            metric = scores(validation_labels, prediction, classes)["balanced_accuracy"]
            candidates.append((metric, alpha, test_map))
        # Deterministic tie break follows declared alpha order.
        _, alpha, test_map = max(candidates, key=lambda item: item[0])
        prediction = (test_map @ centered + intercept).reshape(-1, train_labels.shape[1], classes).argmax(axis=2)
        return prediction, alpha


def probe(train, validation, test, labels, *, permutations=199, seed=26_020_000):
    if permutations < 1:
        raise ValueError("at least one run-label permutation required")
    model = RidgeReadout(train, validation, test)
    result, arrays = {}, {}
    for task, classes in TASKS.items():
        y, v, t = [labels[part][task] for part in ("train", "validation", "test")]
        prediction, alpha = model.predict(y, v, classes)
        metrics = scores(t, prediction, classes)
        majority = np.asarray([np.bincount(y[:, slot], minlength=classes).argmax() for slot in range(y.shape[1])])
        baseline = np.broadcast_to(majority, t.shape)
        rng = np.random.default_rng(derive_seed("card-probe-label-null", seed, task))
        null = []
        for _ in range(permutations):
            # One whole hand label block moves per run; slots stay together.
            shuffled = y[rng.permutation(len(y))]
            shuffled_prediction, _ = model.predict(shuffled, v, classes)
            null.append(scores(t, shuffled_prediction, classes)["balanced_accuracy"])
        baseline_score = scores(t, baseline, classes)
        bootstrap = np.random.default_rng(derive_seed("card-probe-run-bootstrap", seed, task))
        excess = []
        for _ in range(1000):
            indices = bootstrap.integers(0, len(t), len(t))
            excess.append(scores(t[indices], prediction[indices], classes)["balanced_accuracy"]
                          - scores(t[indices], baseline[indices], classes)["balanced_accuracy"])
        result[task] = {**metrics, "uniform_chance": 1 / classes, "slot_majority": baseline_score,
                       "selected_alpha": alpha, "variable_training_features": model.features,
                       "null_mean": float(np.mean(null)), "null_95_interval": np.quantile(null, [.025, .975]).tolist(),
                       "permutation_p_uncorrected": float((1 + np.count_nonzero(np.asarray(null) >= metrics["balanced_accuracy"])) / (permutations + 1)),
                       "excess_over_slot_majority_95_run_bootstrap": np.quantile(excess, [.025, .975]).tolist()}
        arrays.update({task + "_prediction": prediction, task + "_null": np.asarray(null)})
    return result, arrays


def primary_assessment(reports):
    """Six primary tests: rank/suit at sensory/KC/MBON, mean across trained learners."""
    tests = []
    for stage in ("sensory", "kc", "mbon"):
        for task in TASKS:
            selected = ([reports["sensory"]] if stage == "sensory" else
                        [reports[f"trained-{index}-{stage}"] for index in range(3)])
            observed = np.mean([row["metrics"][task]["balanced_accuracy"] for row in selected])
            baseline = np.mean([row["metrics"][task]["slot_majority"]["balanced_accuracy"] for row in selected])
            null = np.mean([row["arrays"][task + "_null"] for row in selected], axis=0)
            tests.append({"stage": stage, "task": task, "balanced_accuracy": float(observed),
                          "slot_majority_balanced_accuracy": float(baseline),
                          "p": float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))})
    previous = 0.
    for index, row in enumerate(sorted(tests, key=lambda row: row["p"])):
        previous = max(previous, min(1., row["p"] * (len(tests) - index)))
        row["holm_p"] = previous
        row["above_controls"] = bool(previous <= .05 and row["balanced_accuracy"] > max(
            1 / TASKS[row["task"]], row["slot_majority_balanced_accuracy"]))
    return {"primary_tests": tests,
            "interpretation": "Evidence concerns linear decoding of shuffled visible-card inputs from mean rates; failed probes do not prove absence of identity or timing information. Decodability is not causal use or gameplay skill."}
