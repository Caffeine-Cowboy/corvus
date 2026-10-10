"""Source-only selection/calibration, followed by immutable target scoring."""
from __future__ import annotations

import csv
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import WaveformDataset, read_manifest, require_binary, sha256, waveform_inventory
from .metrics import binary_metrics, cluster_bootstrap, source_threshold
from .model import TransientCNN


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def empty_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "device": "cpu"}


@torch.inference_mode()
def scores(model, records, length, batch_size=64):
    model.eval()
    values = []
    for x, _ in DataLoader(WaveformDataset(records, length), batch_size=batch_size):
        values.extend(torch.sigmoid(model(x)).cpu().numpy().tolist())
    return np.asarray(values)


def crest_scores(records, length):
    values = []
    for x, _ in WaveformDataset(records, length):
        values.append(float(x.abs().max() / (x.square().mean().sqrt() + 1e-8)))
    return np.asarray(values)


def train(manifest, run_dir, *, epochs=30, length=1024, seed=42, batch_size=16,
          learning_rate=0.001, source_fpr=0.05, shuffle_labels=False):
    if epochs < 1 or batch_size < 1 or length < 32 or not 0 < learning_rate < 1:
        raise ValueError("Invalid training hyperparameters")
    if not 0 < source_fpr < 1:
        raise ValueError("source_fpr must lie in (0,1)")
    records = read_manifest(manifest, domain="astro")
    splits = {s: [r for r in records if r.split == s] for s in ("train", "val", "test")}
    for name in ("train", "val"):
        require_binary(splits[name], f"Source {name}")
    if sum(r.label == 0 for r in splits["val"]) < 2:
        raise ValueError("Source validation needs >=2 negatives for threshold calibration")
    if splits["test"]:
        require_binary(splits["test"], "Source test")
    run_dir = empty_directory(run_dir)
    seed_all(seed)
    model = TransientCNN()
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    labels = np.array([r.label for r in splits["train"]])
    if shuffle_labels:
        labels = np.random.default_rng(seed).permutation(labels)
    loader = DataLoader(WaveformDataset(splits["train"], length, labels),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    weight = float((labels == 0).sum() / (labels == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    val_labels = np.array([r.label for r in splits["val"]])
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        val_scores = scores(model, splits["val"], length)
        clipped = np.clip(val_scores, 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(val_labels * np.log(clipped) + (1 - val_labels) * np.log(1 - clipped)))
        history.append({"epoch": epoch, "train_loss": total_loss / len(labels), "source_val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val_scores = scores(model, splits["val"], length)
    threshold = source_threshold(val_labels, val_scores, source_fpr)
    random_model = TransientCNN()
    random_model.load_state_dict(initial_state)
    random_threshold = source_threshold(val_labels, scores(random_model, splits["val"], length), source_fpr)
    crest_threshold = source_threshold(val_labels, crest_scores(splits["val"], length), source_fpr)
    config = {"epochs": epochs, "length": length, "seed": seed, "batch_size": batch_size,
              "learning_rate": learning_rate, "source_fpr": source_fpr, "shuffle_labels": shuffle_labels,
              "preprocessing": "dimensionless polyphase resampling; per-window median/MAD; clip +/-8",
              "source_manifest_sha256": sha256(manifest), "source_waveforms": waveform_inventory(records),
              "source_domain": "astro", "threshold_domain": "astro", "target_data_used": False,
              "environment": environment()}
    checkpoint = {"state_dict": best_state, "random_state_dict": initial_state,
                  "threshold": threshold, "random_threshold": random_threshold,
                  "crest_threshold": crest_threshold, "config": config, "best_epoch": best_epoch}
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    source_summary = {"status": "source-only training complete", "config": config,
                      "best_epoch": best_epoch, "history": history,
                      "source_validation": binary_metrics(val_labels, val_scores, threshold),
                      "split_counts": {s: {"windows": len(rs), "groups": len({r.group_id for r in rs})}
                                       for s, rs in splits.items()},
                      "caution": "Validation calibration is empirical; source FPR does not imply ECG FPR."}
    if splits["test"]:
        source_summary["source_test"] = binary_metrics([r.label for r in splits["test"]],
                                                       scores(model, splits["test"], length), threshold)
    write_json(run_dir / "training.json", source_summary)
    return source_summary


def evaluate(checkpoint_path, manifest, output, *, bootstrap=500):
    records = read_manifest(manifest, domain="ecg")
    if any(r.split != "test" for r in records):
        raise ValueError("Strict zero-shot target manifest must contain only test records")
    require_binary(records, "ECG test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["source_domain"] != "astro" or config["target_data_used"] is not False:
        raise ValueError("Checkpoint does not satisfy source-only protocol")
    target_inventory = waveform_inventory(records)
    source_digests = {item["signal_sha256"] for item in config["source_waveforms"]}
    if source_digests.intersection(item["signal_sha256"] for item in target_inventory):
        raise ValueError("Source-target waveform overlap; strict zero-shot test rejected")
    output = empty_directory(output)
    seed_all(config["seed"])
    model, random_model = TransientCNN(), TransientCNN()
    model.load_state_dict(checkpoint["state_dict"])
    random_model.load_state_dict(checkpoint["random_state_dict"])
    model.requires_grad_(False)
    random_model.requires_grad_(False)
    length = config["length"]
    # Score all records before labels are used for metrics. No fitting occurs here.
    trained = scores(model, records, length)
    random_values = scores(random_model, records, length)
    crest = crest_scores(records, length)
    y, groups = [r.label for r in records], [r.group_id for r in records]
    report = {"protocol": "strict zero-shot; checkpoint, orientation and thresholds fixed on astro only",
              "endpoint": "Supplied ECG record-level binary labels; consult cohort label provenance",
              "checkpoint_sha256": sha256(checkpoint_path), "target_manifest_sha256": sha256(manifest),
              "target_waveforms": target_inventory, "training_config": config,
              "environment": environment(), "models": {},
              "limitations": ["Source probabilities are not calibrated ECG diagnosis probabilities.",
                              "Scores are not inverted or thresholds tuned using ECG outcomes.",
                              "This experiment does not demonstrate a shared latent representation by itself."]}
    for name, values, threshold in [("astro_cnn", trained, checkpoint["threshold"]),
                                     ("random_cnn", random_values, checkpoint["random_threshold"]),
                                     ("crest_factor", crest, checkpoint["crest_threshold"])]:
        report["models"][name] = {
            **binary_metrics(y, values, threshold),
            "bootstrap": cluster_bootstrap(y, values, groups, threshold, bootstrap, config["seed"])}
    write_json(output / "metrics.json", report)
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "group_id", "label", "astro_score", "random_score", "crest_score", "astro_prediction"])
        writer.writerows((r.record_id, r.group_id, r.label, float(a), float(b), float(c),
                          int(a > checkpoint["threshold"])) for r, a, b, c in zip(records, trained, random_values, crest))
    return report
