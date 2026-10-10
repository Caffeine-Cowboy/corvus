"""Selection independence and strict statistical contracts, with artificial data."""
import csv

import numpy as np
import pytest
import torch

from corvus.data import FIELDS, read_manifest, sha256
from corvus.lab.train import train_ecg
from corvus.metrics import binary_metrics, cluster_bootstrap
from corvus.repeatability import run, summarize_runs
from corvus.synthetic import generate


def supervised_fixture(tmp_path):
    manifest = generate(tmp_path / "artificial") / "ecg/manifest.csv"
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = sorted({r["group_id"] for r in rows})
    for row in rows:
        i = groups.index(row["group_id"])
        row["split"] = "train" if i < 14 else "val" if i < 19 else "test"
    write_rows(manifest, rows)
    return manifest, rows


def write_rows(manifest, rows):
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_supervised_selection_invariant_to_test_outcomes(tmp_path):
    manifest, rows = supervised_fixture(tmp_path)
    before = train_ecg(manifest, tmp_path / "before", epochs=2, seed=17)
    for row in rows:
        if row["split"] == "test":
            row["label"] = str(1 - int(row["label"]))
    write_rows(manifest, rows)
    after = train_ecg(manifest, tmp_path / "after", epochs=2, seed=17)
    assert before["history"] == after["history"]
    assert before["threshold"] == after["threshold"]
    assert before["best_epoch"] == after["best_epoch"]
    a = torch.load(tmp_path / "before/weights.pt", weights_only=True)
    b = torch.load(tmp_path / "after/weights.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("labels,scores", [([0, 2], [.1, .9]), ([0, .5], [.1, .9]),
                                            ([[0], [1]], [.1, .9]), ([0, 1], [[.1], [.9]]),
                                            ([0], [.1, .9]), ([0, 1], [.1, np.nan]), ([], [])])
def test_metrics_reject_silent_coercion_and_broadcasting(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores, .5)


def test_bootstrap_rejects_misaligned_groups():
    with pytest.raises(ValueError, match="align"):
        cluster_bootstrap([0, 1], [.1, .9], ["only-one"], .5)
    with pytest.raises(ValueError, match="finite"):
        binary_metrics([0, 1], [.1, .9], np.nan)


def test_repeatability_exports_only_aggregate_and_is_immutable(tmp_path):
    manifest, _ = supervised_fixture(tmp_path)
    report = run(manifest, tmp_path / "run", tmp_path / "aggregate.json", epochs=1, seeds=(17, 42), bootstrap=20)
    assert report["protocol"]["test_used_for_selection"] is False
    assert len(report["runs"]) == 2
    assert report["mean_auroc_bootstrap_valid_replicates"] == 20
    assert report["runs"][0]["test"]["n"] == 10
    assert "record_id" not in (tmp_path / "aggregate.json").read_text()
    assert "signal_sha256" not in (tmp_path / "aggregate.json").read_text()
    assert "manifest\"" not in (tmp_path / "aggregate.json").read_text()
    with pytest.raises(FileExistsError):
        run(manifest, tmp_path / "run", tmp_path / "aggregate.json", epochs=1, seeds=(17, 42), bootstrap=20)


def test_malformed_manifest_rows_fail_explicitly(tmp_path):
    manifest, rows = supervised_fixture(tmp_path)
    rows[0]["sample_rate_hz"] = None
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="empty"):
        read_manifest(manifest)
