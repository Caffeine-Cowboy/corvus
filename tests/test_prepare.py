"""Data-adapter checks: labels, physical sampling, groups, and safe failure."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from corvus import prepare


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    filename = tmp_path / "observation.h5"
    with h5py.File(filename, "w") as handle:
        handle["frb/ts"] = np.arange(20, dtype=float)
        handle["frb/model_ts"] = np.full(20, 900.0)
        handle["frb/plot_time"] = np.arange(20, dtype=float) * 0.5
    return filename


def annotation_rows(hdf5):
    return [
        dict(record_id="obs/background", path=hdf5.name, group_id="source-1", parent_id="obs-1", label="0", split="train", start="0", stop="8"),
        dict(record_id="obs/pulse", path=hdf5.name, group_id="source-1", parent_id="obs-1", label="1", split="train", start="8", stop="16"),
    ]


def test_chime_extracts_observed_signal_and_explicit_rate(tmp_path, hdf5):
    annotations = write_csv(tmp_path / "annotations.csv", annotation_rows(hdf5))
    manifest = prepare.prepare_chime(annotations, tmp_path / "prepared", "ms")
    rows = read_manifest(manifest)
    assert len(rows) == 2
    assert list(rows[0]) == list(prepare.MANIFEST_FIELDS)
    assert rows[0]["sample_rate_hz"] == "2000.0"
    assert rows[0]["group_id"] == rows[1]["group_id"] == "source-1"
    assert rows[0]["parent_id"] == "obs-1"
    assert rows[0]["domain"] == "astro"
    assert np.array_equal(np.load(manifest.parent / rows[1]["path"]), np.arange(8, 16))
    assert not Path(rows[0]["path"]).is_absolute()
    audit = json.loads((manifest.parent / "preparation-audit.json").read_text())
    assert audit["class_counts"] == {"0": 1, "1": 1}
    assert audit["signal_dataset"] == "frb/ts"
    assert audit["group_count"] == 1


@pytest.mark.parametrize("unit,rate", [("s", 2.0), ("ms", 2000.0), ("us", 2000000.0)])
def test_chime_time_units_are_explicit(tmp_path, hdf5, unit, rate):
    annotations = write_csv(tmp_path / "annotations.csv", annotation_rows(hdf5))
    manifest = prepare.prepare_chime(annotations, tmp_path / "prepared", unit)
    assert float(read_manifest(manifest)[0]["sample_rate_hz"]) == rate


@pytest.mark.parametrize("change,match", [
    ({"split": "val"}, "crosses splits"),
    ({"group_id": "another-source"}, "inconsistent"),
    ({"parent_id": "another-observation"}, "inconsistent"),
    ({"record_id": "obs/background"}, "Duplicate record_id"),
    ({"start": "7"}, "Overlapping"),
    ({"stop": "30"}, "exceeds waveform"),
    ({"label": "unknown"}, "exactly 0 or 1"),
    ({"label": "0"}, "reviewed label=0 and label=1"),
])
def test_chime_rejects_invalid_annotations_without_partial_output(tmp_path, hdf5, change, match):
    rows = annotation_rows(hdf5)
    rows[1].update(change)
    annotations = write_csv(tmp_path / "annotations.csv", rows)
    output = tmp_path / "prepared"
    with pytest.raises(prepare.PreparationError, match=match):
        prepare.prepare_chime(annotations, output, "ms")
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared-*"))


def test_chime_repeater_group_cannot_cross_observations_and_splits(tmp_path, hdf5):
    second = tmp_path / "other.h5"
    second.write_bytes(hdf5.read_bytes())
    rows = annotation_rows(hdf5)
    rows[1].update(path=second.name, parent_id="obs-2", split="test")
    annotations = write_csv(tmp_path / "annotations.csv", rows)
    with pytest.raises(prepare.PreparationError, match="crosses splits"):
        prepare.prepare_chime(annotations, tmp_path / "prepared", "ms")


@pytest.mark.parametrize("times", [
    np.array([0., 1., 2., 4.] + list(range(5, 21))),
    np.arange(20, dtype=float)[::-1],
    np.full(20, np.nan),
])
def test_chime_rejects_nonuniform_or_invalid_time_axis(tmp_path, hdf5, times):
    import h5py
    with h5py.File(hdf5, "r+") as handle:
        handle["frb/plot_time"][:] = times
    annotations = write_csv(tmp_path / "annotations.csv", annotation_rows(hdf5))
    with pytest.raises(prepare.PreparationError, match="cadence|nonfinite"):
        prepare.prepare_chime(annotations, tmp_path / "prepared", "ms")


def test_chime_does_not_replace_existing_output(tmp_path, hdf5):
    annotations = write_csv(tmp_path / "annotations.csv", annotation_rows(hdf5))
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(prepare.PreparationError, match="already exists"):
        prepare.prepare_chime(annotations, output, "ms")
    assert (output / "sentinel").read_text() == "keep"


def mimic_inputs(tmp_path, labels=None, records=None):
    labels = labels if labels is not None else [
        dict(study_id="100", label="1", label_provenance="cardiologist consensus"),
        dict(study_id="101", label="0", label_provenance="cardiologist consensus"),
    ]
    records = records if records is not None else [
        dict(subject_id="10", study_id="100", path="files/p10/s100/100"),
        dict(subject_id="10", study_id="101", path="files/p10/s101/101"),
        dict(subject_id="11", study_id="102", path="files/p11/s102/102"),
    ]
    return (
        write_csv(tmp_path / "record_list.csv", records),
        write_csv(tmp_path / "labels.csv", labels),
    )


def mock_wfdb(monkeypatch, missing_lead=False, bad_fs=False):
    calls = []

    def rdheader(path):
        return SimpleNamespace(sig_name=["I", "V1"] if missing_lead and path.endswith("101") else ["I", "II"])

    def rdrecord(path, *, channels, physical):
        calls.append((path, channels, physical))
        return SimpleNamespace(fs=float("nan") if bad_fs and path.endswith("101") else 250.0, p_signal=np.arange(16, dtype=float)[:, None] * 0.1)

    original = prepare._optional_module
    monkeypatch.setattr(prepare, "_optional_module", lambda name: SimpleNamespace(rdheader=rdheader, rdrecord=rdrecord) if name == "wfdb" else original(name))
    return calls


def test_mimic_joins_explicit_labels_preserves_patient_and_fs(tmp_path, monkeypatch):
    record_list, labels = mimic_inputs(tmp_path)
    calls = mock_wfdb(monkeypatch)
    manifest = prepare.prepare_mimic(tmp_path, record_list, labels, tmp_path / "prepared")
    rows = read_manifest(manifest)
    assert len(rows) == 2
    assert {row["group_id"] for row in rows} == {"patient-10"}
    assert {row["parent_id"] for row in rows} == {"study-100", "study-101"}
    assert {row["split"] for row in rows} == {"test"}
    assert {row["domain"] for row in rows} == {"ecg"}
    assert {row["sample_rate_hz"] for row in rows} == {"250.0"}
    assert all(channels == [1] and physical is True for _, channels, physical in calls)
    assert np.load(manifest.parent / rows[0]["path"]).shape == (16,)
    audit = json.loads((manifest.parent / "preparation-audit.json").read_text())
    assert audit["exclusion_counts"] == {"no_explicit_label": 1}
    assert audit["provided_label_provenance_counts"] == {"cardiologist consensus": 2}


@pytest.mark.parametrize("options,reason", [
    ({"missing_lead": True}, "missing_or_ambiguous_lead"),
    ({"bad_fs": True}, "invalid_or_unreadable_waveform"),
])
def test_mimic_excludes_and_audits_invalid_waveforms(tmp_path, monkeypatch, options, reason):
    record_list, labels = mimic_inputs(tmp_path)
    mock_wfdb(monkeypatch, **options)
    manifest = prepare.prepare_mimic(tmp_path, record_list, labels, tmp_path / "prepared")
    rows = read_manifest(manifest)
    assert len(rows) == 1
    audit = json.loads((manifest.parent / "preparation-audit.json").read_text())
    assert audit["excluded_records"] == 2
    assert audit["exclusion_counts"][reason] == 1
    assert audit["waveform_exclusions"][0]["study_id"] == "101"


@pytest.mark.parametrize("label_rows,match", [
    ([dict(study_id="100", label="unknown")], "exactly 0 or 1"),
    ([dict(study_id="100", label="0"), dict(study_id="100", label="1")], "Duplicate study_id"),
    ([dict(study_id="999", label="1")], "missing from record_list"),
])
def test_mimic_rejects_ambiguous_labels(tmp_path, label_rows, match):
    record_list, labels = mimic_inputs(tmp_path, labels=label_rows)
    with pytest.raises(prepare.PreparationError, match=match):
        prepare.prepare_mimic(tmp_path, record_list, labels, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("records,match", [
    ([dict(subject_id="10", study_id="100")], "missing CSV columns: path"),
    ([dict(subject_id="10", study_id="100", path="../outside")], "relative to --root"),
    ([dict(subject_id="10", study_id="100", path="x"), dict(subject_id="11", study_id="100", path="y")], "Duplicate study_id"),
])
def test_mimic_rejects_invalid_record_schema(tmp_path, records, match):
    record_list, labels = mimic_inputs(tmp_path, records=records)
    with pytest.raises(prepare.PreparationError, match=match):
        prepare.prepare_mimic(tmp_path, record_list, labels, tmp_path / "prepared")


def test_missing_optional_dependency_is_actionable(monkeypatch):
    def unavailable(name):
        raise ImportError(name)
    monkeypatch.setattr(prepare.importlib, "import_module", unavailable)
    with pytest.raises(prepare.PreparationError, match="pip install h5py"):
        prepare._optional_module("h5py")
