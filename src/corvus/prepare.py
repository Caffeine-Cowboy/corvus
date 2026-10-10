"""Convert locally held, explicitly annotated source data to canonical manifests.

These adapters neither download restricted data nor infer clinical labels. The
CHIME adapter supports Catalog 1 FRB HDF5, not CHIME/Pulsar filterbank files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

import numpy as np


MANIFEST_FIELDS = (
    "record_id", "path", "group_id", "parent_id", "domain", "label", "split",
    "sample_rate_hz",
)
TIME_UNIT_SECONDS = {"ms": 1e-3, "us": 1e-6, "s": 1.0}


class PreparationError(ValueError):
    """Input is ambiguous, invalid, or unsuitable for a reproducible conversion."""


def _optional_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise PreparationError(
            f"{name} is required for this adapter; install it with 'pip install {name}'."
        ) from exc


def _csv_rows(path: Path, required: tuple[str, ...]) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = set(required) - set(fields)
        if missing:
            raise PreparationError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        if len(set(fields)) != len(fields):
            raise PreparationError(f"{path}: duplicate CSV column names")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise PreparationError(f"{path}:{line}: malformed CSV row")
            cleaned = {key: value.strip() for key, value in row.items()}
            for key in required:
                if not cleaned[key]:
                    raise PreparationError(f"{path}:{line}: empty {key}")
            yield cleaned


def _label(value: str, identifier: str) -> int:
    if value not in {"0", "1"}:
        raise PreparationError(f"{identifier}: label must be exactly 0 or 1; got {value!r}")
    return int(value)


def _index(value: str, identifier: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise PreparationError(f"{identifier}: interval indices must be integers") from exc
    if str(number) != value or number < 0:
        raise PreparationError(f"{identifier}: interval indices must be nonnegative integers")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _staged_output(output: Path):
    """Publish a complete conversion atomically; never replace an existing output."""
    if output.exists():
        raise PreparationError(f"Output already exists: {output}; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "signals").mkdir()
        yield staging
        if output.exists():
            raise PreparationError(f"Output was created concurrently: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _signal(value, identifier: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1 or value.size < 2:
        raise PreparationError(f"{identifier}: waveform must be 1D with at least two samples")
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise PreparationError(f"{identifier}: waveform must be real numeric data")
    if not np.isfinite(value).all():
        raise PreparationError(f"{identifier}: waveform contains nonfinite samples")
    with np.errstate(over="ignore", invalid="ignore"):
        result = value.astype(np.float32)
    if not np.isfinite(result).all():
        raise PreparationError(f"{identifier}: waveform cannot be represented in float32")
    return result


def _save_signal(staging: Path, record_id: str, signal: np.ndarray) -> str:
    # The ID may contain slashes or other source-specific punctuation; never use
    # it directly as a filename.
    name = hashlib.sha256(record_id.encode()).hexdigest() + ".npy"
    relative = Path("signals") / name
    np.save(staging / relative, signal, allow_pickle=False)
    return relative.as_posix()


def _write_outputs(staging: Path, rows: list[dict], audit: dict) -> None:
    with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit["accepted_records"] = len(rows)
    audit["class_counts"] = dict(sorted(Counter(str(row["label"]) for row in rows).items()))
    audit["split_counts"] = dict(sorted(Counter(row["split"] for row in rows).items()))
    audit["group_count"] = len({row["group_id"] for row in rows})
    with (staging / "preparation-audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _cadence(plot_time: np.ndarray, time_unit: str, identifier: str) -> float:
    if plot_time.ndim != 1 or len(plot_time) < 2:
        raise PreparationError(f"{identifier}: frb/plot_time must be 1D with at least two samples")
    if np.iscomplexobj(plot_time) or not np.issubdtype(plot_time.dtype, np.number):
        raise PreparationError(f"{identifier}: frb/plot_time must be real numeric data")
    times = np.asarray(plot_time, dtype=np.float64)
    if not np.isfinite(times).all():
        raise PreparationError(f"{identifier}: time axis contains nonfinite values")
    differences = np.diff(times)
    delta = float(np.median(differences))
    # Account for stored floating-point rounding without accepting genuinely
    # irregular sampling. Unit conversion happens only after this check.
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(times)))) * 8
    if delta <= 0 or not np.all(differences > 0) or not np.allclose(
        differences, delta, rtol=1e-3, atol=tolerance
    ):
        raise PreparationError(f"{identifier}: time axis must be increasing with uniform cadence")
    rate = 1.0 / (delta * TIME_UNIT_SECONDS[time_unit])
    if not math.isfinite(rate) or rate <= 0:
        raise PreparationError(f"{identifier}: invalid derived sampling rate")
    return rate


def prepare_chime(annotations: str | Path, output: str | Path, time_unit: str) -> Path:
    """Convert explicitly reviewed Catalog 1 FRB sample intervals to a manifest.

    Relative source HDF5 paths resolve against the annotations CSV directory.
    Both class labels must be explicitly present. All intervals from an input
    observation, and all observations of a physical source, stay in one split.
    """
    if time_unit not in TIME_UNIT_SECONDS:
        raise PreparationError("time_unit must be one of: ms, us, s")
    annotations, output = Path(annotations).resolve(), Path(output).resolve()
    required = ("record_id", "path", "group_id", "parent_id", "label", "split", "start", "stop")
    rows = list(_csv_rows(annotations, required))
    if not rows:
        raise PreparationError("CHIME annotations must contain explicit positive and negative windows")
    seen_ids: set[str] = set()
    groups: dict[str, str] = {}
    parents: dict[str, tuple[str, str, str]] = {}
    paths: dict[str, tuple[str, str, str]] = {}
    by_path: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        identifier = row["record_id"]
        if identifier in seen_ids:
            raise PreparationError(f"Duplicate record_id: {identifier}")
        seen_ids.add(identifier)
        row["label"] = _label(row["label"], identifier)
        if row["split"] not in {"train", "val", "test"}:
            raise PreparationError(f"{identifier}: split must be train, val, or test")
        start, stop = _index(row["start"], identifier), _index(row["stop"], identifier)
        if stop - start < 2:
            raise PreparationError(f"{identifier}: [start, stop) must contain at least two samples")
        row["start"], row["stop"] = start, stop
        source = Path(row["path"]).expanduser()
        source = (source if source.is_absolute() else annotations.parent / source).resolve()
        row["path"] = str(source)
        group, parent, split = row["group_id"], row["parent_id"], row["split"]
        if group in groups and groups[group] != split:
            raise PreparationError(f"Source group {group!r} crosses splits")
        groups[group] = split
        parent_value = (str(source), group, split)
        if parent in parents and parents[parent] != parent_value:
            raise PreparationError(f"Observation parent_id {parent!r} has inconsistent path/group/split")
        parents[parent] = parent_value
        path_value = (parent, group, split)
        if str(source) in paths and paths[str(source)] != path_value:
            raise PreparationError(f"Source file {source} has inconsistent parent/group/split")
        paths[str(source)] = path_value
        by_path[str(source)].append(row)
    if {row["label"] for row in rows} != {0, 1}:
        raise PreparationError("CHIME annotations must include reviewed label=0 and label=1 windows")
    for source_rows in by_path.values():
        ordered = sorted(source_rows, key=lambda row: row["start"])
        # Overlap can create pseudo-replication or contradictory labels. Demand
        # disjoint exported windows even within the same split.
        for previous, current in zip(ordered, ordered[1:]):
            if current["start"] < previous["stop"]:
                raise PreparationError(f"Overlapping windows: {previous['record_id']} and {current['record_id']}")
    h5py = _optional_module("h5py")
    prepared: list[dict] = []
    audit = {
        "adapter": "chime-frb-catalog1", "time_unit": time_unit,
        "annotations": str(annotations), "annotations_sha256": _sha256(annotations),
        "input_records": len(rows), "input_observations": len(by_path),
        "excluded_records": 0, "exclusion_counts": {},
        "signal_dataset": "frb/ts", "time_dataset": "frb/plot_time",
        "label_policy": "Explicit externally reviewed positive and negative sample intervals",
    }
    with _staged_output(output) as staging:
        for source, source_rows in by_path.items():
            with h5py.File(source, "r") as handle:
                if "frb/ts" not in handle or "frb/plot_time" not in handle:
                    raise PreparationError(f"{source}: requires frb/ts and frb/plot_time datasets")
                signal = handle["frb/ts"]
                times = np.asarray(handle["frb/plot_time"])
                if signal.ndim != 1 or signal.shape != times.shape:
                    raise PreparationError(f"{source}: ts and plot_time must be equally sized 1D arrays")
                rate = _cadence(times, time_unit, source)
                for row in source_rows:
                    if row["stop"] > len(signal):
                        raise PreparationError(f"{row['record_id']}: interval exceeds waveform length {len(signal)}")
                    window = _signal(signal[row["start"]:row["stop"]], row["record_id"])
                    prepared.append({
                        "record_id": row["record_id"],
                        "path": _save_signal(staging, row["record_id"], window),
                        "group_id": row["group_id"], "parent_id": row["parent_id"],
                        "domain": "astro", "label": row["label"], "split": row["split"],
                        "sample_rate_hz": rate,
                    })
        _write_outputs(staging, prepared, audit)
    return output / "manifest.csv"


def prepare_mimic(
    root: str | Path, record_list: str | Path, labels: str | Path,
    output: str | Path, lead: str = "II",
) -> Path:
    """Join externally supplied study labels to local WFDB diagnostic ECGs.

    Unlabelled records are excluded. Invalid waveform records are excluded with
    explicit reasons in the audit; schema and label ambiguities fail the run.
    Every accepted target record belongs to the untouched test split.
    """
    root, record_list = Path(root).resolve(), Path(record_list).resolve()
    labels, output = Path(labels).resolve(), Path(output).resolve()
    if not lead.strip():
        raise PreparationError("lead cannot be empty")
    label_rows: dict[str, dict] = {}
    for row in _csv_rows(labels, ("study_id", "label")):
        study = row["study_id"]
        if study in label_rows:
            raise PreparationError(f"Duplicate study_id in labels: {study}")
        row["label"] = _label(row["label"], study)
        label_rows[study] = row
    if not label_rows:
        raise PreparationError("Labels CSV contains no explicitly labelled studies")
    selected: list[dict] = []
    seen_studies: set[str] = set()
    seen_paths: set[str] = set()
    exclusions: Counter = Counter()
    for row in _csv_rows(record_list, ("subject_id", "study_id", "path")):
        study = row["study_id"]
        if study in seen_studies:
            raise PreparationError(f"Duplicate study_id in record_list: {study}")
        seen_studies.add(study)
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PreparationError(f"{study}: record path must be relative to --root without '..'")
        if relative.suffix in {".hea", ".dat"}:
            relative = relative.with_suffix("")
        local = (root / relative).resolve()
        if not local.is_relative_to(root):
            raise PreparationError(f"{study}: record path escapes --root")
        if str(local) in seen_paths:
            raise PreparationError(f"Duplicate waveform path in record_list: {local}")
        seen_paths.add(str(local))
        if study not in label_rows:
            exclusions["no_explicit_label"] += 1
            continue
        row["path"] = str(local)
        row["label"] = label_rows[study]["label"]
        selected.append(row)
    unknown = set(label_rows) - seen_studies
    if unknown:
        raise PreparationError(f"Label study_id missing from record_list: {', '.join(sorted(unknown)[:10])}")
    wfdb = _optional_module("wfdb")
    prepared: list[dict] = []
    failed_records: list[dict] = []
    with _staged_output(output) as staging:
        for row in selected:
            study = row["study_id"]
            try:
                header = wfdb.rdheader(row["path"])
                names = list(header.sig_name or [])
                if names.count(lead) != 1:
                    exclusions["missing_or_ambiguous_lead"] += 1
                    failed_records.append({"study_id": study, "reason": "missing_or_ambiguous_lead"})
                    continue
                record = wfdb.rdrecord(row["path"], channels=[names.index(lead)], physical=True)
                rate = float(record.fs)
                if not math.isfinite(rate) or rate <= 0:
                    raise PreparationError("invalid sampling rate")
                physical = np.asarray(record.p_signal)
                if physical.ndim != 2 or physical.shape[1] != 1:
                    raise PreparationError("requested physical lead was not returned")
                signal = _signal(physical[:, 0], study)
            except (OSError, ValueError, TypeError, IndexError, KeyError) as exc:
                exclusions["invalid_or_unreadable_waveform"] += 1
                failed_records.append({"study_id": study, "reason": "invalid_or_unreadable_waveform", "detail": str(exc)})
                continue
            identifier = f"mimic-{study}"
            prepared.append({
                "record_id": identifier, "path": _save_signal(staging, identifier, signal),
                "group_id": f"patient-{row['subject_id']}", "parent_id": f"study-{study}",
                "domain": "ecg", "label": row["label"], "split": "test",
                "sample_rate_hz": rate,
            })
        audit = {
            "adapter": "mimic-iv-ecg", "lead": lead, "root": str(root),
            "record_list": str(record_list), "record_list_sha256": _sha256(record_list),
            "labels": str(labels), "labels_sha256": _sha256(labels),
            "input_records": len(seen_studies), "labelled_records": len(selected),
            "excluded_records": sum(exclusions.values()),
            "exclusion_counts": dict(sorted(exclusions.items())),
            "waveform_exclusions": failed_records,
            "provided_label_provenance_counts": dict(Counter(
                row.get("label_provenance", "unspecified") or "unspecified"
                for row in label_rows.values()
            )),
            "label_policy": "Explicit external study labels only; unlabelled studies excluded; adjudication is not inferred",
        }
        _write_outputs(staging, prepared, audit)
    return output / "manifest.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="adapter", required=True)
    chime = subcommands.add_parser("chime", help="Reviewed CHIME/FRB Catalog 1 HDF5 intervals")
    chime.add_argument("--annotations", type=Path, required=True)
    chime.add_argument("--output", type=Path, required=True)
    chime.add_argument("--time-unit", choices=tuple(TIME_UNIT_SECONDS), required=True)
    mimic = subcommands.add_parser("mimic", help="Locally held MIMIC-IV-ECG WFDB records")
    mimic.add_argument("--root", type=Path, required=True)
    mimic.add_argument("--record-list", type=Path, required=True)
    mimic.add_argument("--labels", type=Path, required=True)
    mimic.add_argument("--output", type=Path, required=True)
    mimic.add_argument("--lead", default="II")
    args = parser.parse_args(argv)
    try:
        if args.adapter == "chime":
            manifest = prepare_chime(args.annotations, args.output, args.time_unit)
        else:
            manifest = prepare_mimic(args.root, args.record_list, args.labels, args.output, args.lead)
    except (PreparationError, OSError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(json.dumps({"manifest": str(manifest), "audit": str(manifest.parent / "preparation-audit.json")}))


if __name__ == "__main__":
    main()
