import csv
import json

import numpy as np
import pytest
import torch

from corvus.data import FIELDS, read_manifest, sha256, transform
from corvus.experiment import evaluate, train
from corvus.metrics import binary_metrics, cluster_bootstrap, source_threshold
from corvus.synthetic import generate


def test_transform_affine_invariance_and_input_rejection():
    rng = np.random.default_rng(4)
    x = rng.normal(size=128)
    np.testing.assert_allclose(transform(x, 128), transform(7 * x + 12, 128), atol=1e-5)
    assert transform(x, 64).shape == (1, 64)
    with pytest.raises(ValueError, match="Flat"):
        transform(np.ones(128))
    with pytest.raises(ValueError, match="1D"):
        transform(np.ones((2, 100)))
    with pytest.raises(ValueError, match="missing"):
        transform(np.full(128, np.nan))
    with pytest.raises(ValueError, match="real numeric"):
        transform(np.arange(128) + 1j)


def test_no_alias_or_group_split_leakage(tmp_path):
    for i in range(2):
        np.save(tmp_path / f"{i}.npy", np.arange(32) + i)
    rows = [["a", "0.npy", "same", "p1", "astro", 0, "train", 500],
            ["b", "1.npy", "same", "p2", "astro", 1, "val", 500]]
    manifest = tmp_path / "manifest.csv"
    def write():
        with manifest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows(rows)
    write()
    with pytest.raises(ValueError, match="Split leakage"):
        read_manifest(manifest)
    rows[1][2] = "different"
    np.save(tmp_path / "1.npy", np.arange(32, dtype=np.float32))
    write()
    with pytest.raises(ValueError, match="Identical waveform"):
        read_manifest(manifest)


def test_threshold_ties_and_cluster_ci():
    threshold = source_threshold([0, 0, 1, 1], [0.3, 0.3, 0.5, 0.6])
    result = binary_metrics([0, 0, 1, 1], [0.3, 0.3, 0.5, 0.6], threshold)
    assert result["fp"] == 0 and result["tp"] == 2
    assert result["auroc"] == 1
    ci = cluster_bootstrap([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "a", "b", "b"], 0.5, 30)
    assert ci["groups"] == 2
    assert ci["skipped_replicates"] > 0
    assert ci["intervals_95_percent"]["auroc"] == [1.0, 1.0]


def test_source_only_frozen_end_to_end(tmp_path):
    data = generate(tmp_path / "synthetic")
    source, target = data / "astro/manifest.csv", data / "ecg/manifest.csv"
    with pytest.raises(ValueError, match="exclusively astro"):
        train(target, tmp_path / "invalid", epochs=1)
    train(source, tmp_path / "run", epochs=2, length=128)
    checkpoint = tmp_path / "run/checkpoint.pt"
    before = sha256(checkpoint)
    result = evaluate(checkpoint, target, tmp_path / "evaluation", bootstrap=20)
    assert sha256(checkpoint) == before
    assert result["models"]["astro_cnn"]["n"] == 48
    assert result["training_config"]["target_data_used"] is False
    assert set(result["models"]) == {"astro_cnn", "random_cnn", "crest_factor"}
    assert json.loads((tmp_path / "evaluation/metrics.json").read_text()) == result
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["threshold"] == result["models"]["astro_cnn"]["threshold"]
    copied = list(csv.DictReader(source.open()))
    for row in copied:
        row["domain"], row["split"] = "ecg", "test"
    copied_manifest = source.parent / "copied_target.csv"
    with copied_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(copied)
    with pytest.raises(ValueError, match="Source-target waveform overlap"):
        evaluate(checkpoint, copied_manifest, tmp_path / "overlap", bootstrap=20)
    # Flipping only target labels must never change model scores or decisions.
    rows = list(csv.DictReader(target.open()))
    for row in rows:
        row["label"] = str(1 - int(row["label"]))
    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    evaluate(checkpoint, target, tmp_path / "flipped", bootstrap=20)
    predictions = list(csv.DictReader((tmp_path / "evaluation/predictions.csv").open()))
    flipped = list(csv.DictReader((tmp_path / "flipped/predictions.csv").open()))
    for original, changed in zip(predictions, flipped):
        assert original["astro_score"] == changed["astro_score"]
        assert original["astro_prediction"] == changed["astro_prediction"]
