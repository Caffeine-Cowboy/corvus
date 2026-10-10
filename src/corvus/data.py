"""Canonical manifests and identical, label-independent window transforms."""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

FIELDS = ["record_id", "path", "group_id", "parent_id", "domain", "label", "split", "sample_rate_hz"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signal_digest(path):
    """Hash numerical content independent of NPY headers and storage dtype."""
    original = np.load(path, allow_pickle=False)
    if original.dtype.kind not in "fiu":
        raise ValueError("Waveforms must contain real numeric values")
    x = np.ascontiguousarray(original, dtype="<f4")
    h = hashlib.sha256(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Record:
    record_id: str
    path: Path
    group_id: str
    parent_id: str
    domain: str
    label: int
    split: str
    sample_rate_hz: float


def read_manifest(path, domain=None):
    path = Path(path).resolve()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    result, ids, assignments, content_splits = [], set(), {}, {}
    for row in rows:
        if None in row or any(row[k] is None or not str(row[k]).strip() for k in FIELDS):
            raise ValueError("Manifest values cannot be empty")
        if row["record_id"] in ids:
            raise ValueError("Duplicate record_id")
        ids.add(row["record_id"])
        if row["domain"] not in {"astro", "ecg"} or (domain and row["domain"] != domain):
            raise ValueError(f"Expected exclusively {domain or 'astro/ecg'} records")
        if row["label"] not in {"0", "1"} or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Expected binary labels and train/val/test splits")
        rate = float(row["sample_rate_hz"])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid sample_rate_hz")
        signal_path = (path.parent / row["path"]).resolve()
        if not signal_path.is_file() or signal_path.suffix != ".npy":
            raise ValueError(f"Missing .npy signal: {signal_path}")
        for kind in ("group_id", "parent_id"):
            key = (row["domain"], kind, row[kind])
            if key in assignments and assignments[key] != row["split"]:
                raise ValueError(f"Split leakage through {kind}: {row[kind]}")
            assignments[key] = row["split"]
        digest = signal_digest(signal_path)
        key = (row["domain"], digest)
        if key in content_splits and content_splits[key] != row["split"]:
            raise ValueError("Identical waveform content crosses splits")
        content_splits[key] = row["split"]
        result.append(Record(row["record_id"], signal_path, row["group_id"], row["parent_id"],
                             row["domain"], int(row["label"]), row["split"], rate))
    return result


def transform(signal, length=1024):
    """Map one complete window to dimensionless time; no population fitting.

    This deliberately changes physical frequency and must not be interpreted as
    resampling both domains to a shared physical sample rate.
    """
    original = np.asarray(signal)
    if original.dtype.kind not in "fiu":
        raise ValueError("Waveforms must contain real numeric values")
    x = np.asarray(original, dtype=np.float64)
    if x.ndim != 1 or x.size < 16 or length < 32:
        raise ValueError("Expected a 1D waveform >=16 samples and length >=32")
    finite = np.isfinite(x)
    if finite.mean() < 0.99:
        raise ValueError("More than 1% missing/nonfinite samples")
    if not finite.all():
        x = np.interp(np.arange(x.size), np.flatnonzero(finite), x[finite])
    if np.std(x) < 1e-8:
        raise ValueError("Flat waveform")
    # Remove DC before filtering so its zero-padded boundary cannot create pulses.
    x = x - np.median(x)
    # Polyphase resampling applies an anti-aliasing filter when shortening.
    divisor = math.gcd(length, x.size)
    x = resample_poly(x, length // divisor, x.size // divisor)
    x -= np.median(x)
    scale = 1.4826 * np.median(np.abs(x))
    if scale < 1e-8:
        scale = np.std(x)
    if scale < 1e-8:
        raise ValueError("Flat waveform")
    return np.clip(x / scale, -8, 8).astype(np.float32)[None, :]


class WaveformDataset(Dataset):
    def __init__(self, records, length=1024, labels=None):
        self.records, self.length = records, length
        self.labels = [r.label for r in records] if labels is None else labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            x = transform(np.load(record.path, allow_pickle=False), self.length)
        except ValueError as e:
            raise ValueError(f"{record.record_id}: {e}") from e
        return torch.from_numpy(x), torch.tensor(float(self.labels[index]))


def require_binary(records, name):
    if {r.label for r in records} != {0, 1}:
        raise ValueError(f"{name} must contain both classes")


def waveform_inventory(records):
    return [{"record_id": r.record_id, "sha256": sha256(r.path),
             "signal_sha256": signal_digest(r.path)} for r in records]
