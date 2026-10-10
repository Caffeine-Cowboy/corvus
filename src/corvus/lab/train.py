"""Supervised ECG baseline, selected exclusively on ECG validation data."""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from corvus.data import read_manifest, require_binary, sha256, transform, waveform_inventory
from corvus.experiment import empty_directory, environment, seed_all, write_json
from corvus.metrics import binary_metrics
from corvus.model import TransientCNN


def tensorize(records, length=1024):
    return np.stack([transform(np.load(r.path, allow_pickle=False), length) for r in records])


@torch.inference_mode()
def predict_tensor(model, x, batch_size=32):
    model.eval()
    return np.concatenate([torch.sigmoid(model(torch.from_numpy(batch))).numpy()
                           for batch in np.array_split(x, max(1, int(np.ceil(len(x) / batch_size))))])


def validation_threshold(y, scores):
    """Maximum balanced accuracy; prefer specificity and then a higher threshold."""
    unique = np.unique(scores)
    candidates = np.r_[0.0, (unique[:-1] + unique[1:]) / 2, 1.0]
    best = None
    for threshold in candidates:
        metrics = binary_metrics(y, scores, threshold)
        key = (metrics["balanced_accuracy"], metrics["specificity"], float(threshold))
        if best is None or key > best:
            best = key
    return best[2]


def train_ecg(manifest, output, epochs=30, seed=42):
    records = read_manifest(manifest, domain="ecg")
    splits = {s: [r for r in records if r.split == s] for s in ("train", "val", "test")}
    for name, rows in splits.items():
        require_binary(rows, name)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output = empty_directory(output)
    seed_all(seed)
    train_x, val_x = tensorize(splits["train"]), tensorize(splits["val"])
    train_y = np.array([r.label for r in splits["train"]], dtype=np.float32)
    val_y = np.array([r.label for r in splits["val"]])
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = TransientCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, best_weights, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
        val_scores = predict_tensor(model, val_x)
        val_loss = float(log_loss(val_y, val_scores, labels=[0, 1]))
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_y), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}; validation loss {val_loss:.4f}", flush=True)
    model.load_state_dict(best_weights)
    val_scores = predict_tensor(model, val_x)
    threshold = validation_threshold(val_y, val_scores)
    torch.save(best_weights, output / "weights.pt")
    report = {"experiment": "supervised ECG baseline; separate from strict zero-shot study",
              "seed": seed, "epochs": epochs, "best_epoch": best_epoch, "length": 1024,
              "threshold": threshold, "threshold_selection": "validation balanced accuracy, specificity tie-break",
              "manifest_sha256": sha256(manifest), "manifest": str(Path(manifest).resolve()),
              "split_counts": {s: len(rs) for s, rs in splits.items()},
              "validation": binary_metrics(val_y, val_scores, threshold), "history": history,
              "training_waveforms": waveform_inventory(splits["train"]),
              "validation_waveforms": waveform_inventory(splits["val"]),
              "environment": environment(), "test_used_for_selection": False}
    write_json(output / "training.json", report)
    return report
