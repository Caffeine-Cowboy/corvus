"""Fixed-threshold metrics and cluster bootstrap uncertainty."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _validated(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or not len(y) or len(y) != len(s):
        raise ValueError("Labels and scores must be nonempty aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Labels must be binary and scores finite")
    return y.astype(np.int64), s


def source_threshold(labels, scores, fpr=0.05):
    if not 0 < fpr < 1:
        raise ValueError("FPR must be between 0 and 1")
    y, s = _validated(labels, scores)
    negatives = s[y == 0]
    if negatives.size < 2 or not np.isfinite(negatives).all():
        raise ValueError("Need at least two finite source validation negatives")
    # Decisions use strict >; ties at the selected quantile remain negative.
    return float(np.quantile(negatives, 1 - fpr, method="higher"))


def binary_metrics(labels, scores, threshold):
    y, s = _validated(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    pred = s > threshold
    tp, tn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    fp, fn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    both = len(np.unique(y)) == 2
    return {"n": len(y), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, s)) if both else None,
            "average_precision": float(average_precision_score(y, s)) if both else None,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2 if both else None,
            "threshold": float(threshold), "decision_rule": "score > threshold",
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def cluster_bootstrap(labels, scores, groups, threshold, iterations=500, seed=42):
    y, s = _validated(labels, scores)
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) != len(y) or any(str(x).strip() == "" for x in g):
        raise ValueError("Groups must align with labels and cannot be empty")
    binary_metrics(y, s, threshold)
    names = np.unique(g)
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    members = {name: np.flatnonzero(g == name) for name in names}
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ("auroc", "average_precision", "sensitivity", "specificity", "balanced_accuracy")}
    valid = 0
    if len(names) >= 2:
        for _ in range(iterations):
            indices = np.concatenate([members[n] for n in rng.choice(names, len(names), replace=True)])
            if len(np.unique(y[indices])) < 2:
                continue
            metrics = binary_metrics(y[indices], s[indices], threshold)
            valid += 1
            for key in values:
                values[key].append(metrics[key])
    return {"unit": "group_id", "groups": len(names), "iterations": iterations,
            "valid_replicates": valid, "skipped_replicates": iterations - valid,
            "intervals_95_percent": {k: np.quantile(v, [0.025, 0.975]).tolist() if v else None
                                      for k, v in values.items()},
            "scope": "Sampling uncertainty conditional on this frozen model and selected cohort; not training-seed uncertainty."}
