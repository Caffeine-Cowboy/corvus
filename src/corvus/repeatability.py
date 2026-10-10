"""Frozen-seed repeatability of the supervised ECG baseline.

This rerun reuses an already-observed test partition. It measures training-seed
variation conditional on that cohort and cannot establish a new generalization
claim. Work directories may contain identifying provenance; export only the
aggregate report. No scores, waveforms, records, or weights enter that report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from corvus.data import read_manifest, require_binary, sha256
from corvus.experiment import environment, write_json
from corvus.lab.train import predict_tensor, tensorize, train_ecg
from corvus.metrics import binary_metrics, cluster_bootstrap
from corvus.model import TransientCNN

SEEDS = (17, 42, 173)


def code_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def summarize_runs(runs):
    """Describe training-seed variability; do not masquerade it as a CI."""
    keys = ("auroc", "average_precision", "sensitivity", "specificity", "balanced_accuracy")
    return {key: {"mean": float(np.mean([r["test"][key] for r in runs])),
                  "sample_standard_deviation": float(np.std([r["test"][key] for r in runs], ddof=1)),
                  "min": float(min(r["test"][key] for r in runs)),
                  "max": float(max(r["test"][key] for r in runs))}
            for key in keys}


def run(manifest, work, aggregate, *, epochs=30, seeds=SEEDS, bootstrap=2000):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Need at least two distinct predeclared seeds")
    if epochs < 1 or bootstrap < 1:
        raise ValueError("Epochs and bootstrap repetitions must be positive")
    manifest, work, aggregate = Path(manifest), Path(work), Path(aggregate)
    # Never silently replace a published result or re-open a completed holdout.
    if work.exists() or aggregate.exists():
        raise FileExistsError("Choose fresh work and aggregate paths; this protocol is immutable")
    records = read_manifest(manifest, "ecg")
    splits = {s: [r for r in records if r.split == s] for s in ("train", "val", "test")}
    for split, rows in splits.items():
        require_binary(rows, split)
    work.mkdir(parents=True)
    protocol = {
        "study": "supervised CNN training-seed repeatability on previously observed holdout",
        "seeds": list(seeds), "epochs": epochs, "batch_size": 32,
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.01,
        "gradient_clip_norm": 5.0, "input_length": 1024, "architecture": "TransientCNN",
        "checkpoint_selection": "lowest validation log loss, earliest epoch on exact ties",
        "threshold_selection": "validation balanced accuracy; specificity and high threshold tie-break",
        "test_used_for_selection": False, "cpu_threads": 2,
        "manifest_sha256": sha256(manifest), "source_code_sha256": code_hash(),
        "bootstrap_repetitions": bootstrap, "bootstrap_seed": 20261010,
        "split_counts": {s: {"records": len(rows), "groups": len({r.group_id for r in rows}),
                               "positives": sum(r.label for r in rows)} for s, rows in splits.items()},
        "holdout_status": "already inspected in the historical one-seed lab; no new holdout claim",
        "software_environment": environment(),
    }
    write_json(work / "protocol-frozen-before-training.json", protocol)
    frozen_hash = sha256(work / "protocol-frozen-before-training.json")
    start = time.monotonic()
    training = []
    # All weights and thresholds are selected before any test scores are computed.
    for seed in seeds:
        result = train_ecg(manifest, work / f"seed-{seed}", epochs=epochs, seed=seed)
        training.append(result)
    if sha256(manifest) != protocol["manifest_sha256"] or code_hash() != protocol["source_code_sha256"]:
        raise RuntimeError("Manifest or code changed after freezing; refusing evaluation")
    if sha256(work / "protocol-frozen-before-training.json") != frozen_hash:
        raise RuntimeError("Frozen protocol changed; refusing evaluation")
    test_x = tensorize(splits["test"])
    y = np.array([r.label for r in splits["test"]])
    groups = [r.group_id for r in splits["test"]]
    runs, scores = [], []
    for seed, fitted in zip(seeds, training):
        model = TransientCNN()
        checkpoint = work / f"seed-{seed}" / "weights.pt"
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        values = predict_tensor(model, test_x)
        scores.append(values)
        threshold = fitted["threshold"]
        runs.append({"seed": seed, "checkpoint_sha256": sha256(checkpoint),
                     "best_epoch": fitted["best_epoch"], "threshold": threshold,
                     "validation": fitted["validation"],
                     "test": binary_metrics(y, values, threshold),
                     "patient_bootstrap": cluster_bootstrap(y, values, groups, threshold, bootstrap, 20261010)})
    # Sampling CI for the mean AUROC of the frozen models: common resampled
    # patients preserve model pairing; seeds are fixed, not resampled.
    names = np.unique(groups)
    members = {name: np.flatnonzero(np.asarray(groups) == name) for name in names}
    rng = np.random.default_rng(20261010)
    values = []
    for _ in range(bootstrap):
        ix = np.concatenate([members[n] for n in rng.choice(names, len(names), replace=True)])
        if len(np.unique(y[ix])) == 2:
            values.append(np.mean([binary_metrics(y[ix], s[ix], t["threshold"])["auroc"]
                                   for s, t in zip(scores, runs)]))
    report = {
        "evidence_type": "new_local_repeatability_rerun_on_previously_observed_holdout",
        "protocol": protocol, "frozen_protocol_sha256": frozen_hash,
        "runs": runs, "training_seed_variation": summarize_runs(runs),
        "mean_auroc_patient_bootstrap_95_percent": np.quantile(values, [.025, .975]).tolist(),
        "mean_auroc_bootstrap_valid_replicates": len(values),
        "mean_auroc_bootstrap_scope": "Patient sampling uncertainty conditional on three frozen seeds and this selected cohort; excludes new-population and training-seed uncertainty",
        "wall_seconds": time.monotonic() - start,
        "limitations": ["Historical holdout was already observed before this protocol was frozen.",
                        "Three seeds give a descriptive sample SD, not a precise training-seed confidence interval.",
                        "Machine-report weak labels; balanced convenience sampling; one ten-second lead-II recording per patient.",
                        "No external, prospective, clinical, or multimodal validation.",
                        "Exact reruns need locally acquired records; no raw records, record IDs, or individual predictions are distributed."],
    }
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    write_json(aggregate, report)
    write_json(work / "evaluation-complete.json", {"aggregate_sha256": sha256(aggregate)})
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--aggregate", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--bootstrap", type=int, default=2000)
    args = vars(p.parse_args(argv))
    result = run(**args)
    print(json.dumps({"training_seed_variation": result["training_seed_variation"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
