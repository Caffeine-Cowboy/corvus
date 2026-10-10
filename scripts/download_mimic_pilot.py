#!/usr/bin/env python3
"""Acquire a bounded, deterministic MIMIC-IV-ECG PVC/normal pilot (stdlib only).

The labels are machine-report weak labels, not cardiologist annotations. Selection
uses metadata order and takes at most one ECG per patient across both classes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
from datetime import datetime, timezone
import urllib.request

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"
MIB = 1024 * 1024
REPORT_FIELDS = [f"report_{i}" for i in range(18)]
PVC = re.compile(
    r"\bpvc(?:s|\(s\))?\b|\bpremature ventricular (?:complex(?:es)?|contractions?|beats?)\b"
    r"|\bventricular premature (?:complex(?:es)?|contractions?|beats?)\b"
    r"|\bventricular ectop(?:y|ic beats?)\b", re.I,
)
UNCERTAIN = re.compile(
    r"\b(?:no|not|without|absent|possible|possibly|probable|probably|history|previous"
    r"|prior|resolved|disappeared|consider|suspect(?:ed)?|questionable|versus|vs)\b"
    r"|\bcannot\b|\bmay\b|\bmight\b|\bcould\b|\bor\b|\?", re.I,
)
NORMAL_LINES = {"sinus rhythm", "normal sinus rhythm", "normal ecg", "summary normal ecg"}


def classify(row: dict[str, str]) -> tuple[int | None, str]:
    """Use affirmative PVC lines or an entirely normal machine report."""
    lines = [row.get(k, "").strip() for k in REPORT_FIELDS if row.get(k, "").strip()]
    mentions = [line for line in lines if PVC.search(line)]
    if mentions and all(not UNCERTAIN.search(line) for line in mentions):
        return 1, " || ".join(mentions)
    normalized = [re.sub(r"[^a-z0-9 ]", "", s.lower()).strip() for s in lines]
    if (any(s in {"sinus rhythm", "normal sinus rhythm"} for s in normalized)
            and any(s in {"normal ecg", "summary normal ecg"} for s in normalized)
            and all(s in NORMAL_LINES for s in normalized)):
        return 0, " || ".join(lines)
    return None, ""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str, max_bytes: int, byte_range: tuple[int, int] | None = None) -> tuple[bytes, dict]:
    headers = {"User-Agent": "Project-Corvus-research-pilot/1.0"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if byte_range and response.status != 206:
                    raise RuntimeError(f"Server did not honor bounded byte request: {url}")
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise RuntimeError(f"Response exceeded byte budget: {url}")
                return data, {
                    "url": url, "bytes": len(data), "sha256": sha256(data),
                    "status": response.status, "content_range": response.headers.get("Content-Range"),
                    "etag": response.headers.get("ETag"),
                }
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def csv_prefix(data: bytes) -> csv.DictReader:
    # PhysioNet's table stores one record per line. Discard an incomplete final
    # line of the requested prefix. CSV quoting within a complete line is kept.
    complete = data[:data.rfind(b"\n") + 1].decode("utf-8")
    return csv.DictReader(io.StringIO(complete))


def select(data: bytes, per_class: int) -> tuple[list[dict], int]:
    selected, subjects, counts = [], set(), {0: 0, 1: 0}
    scanned = 0
    for scanned, row in enumerate(csv_prefix(data), 1):
        label, evidence = classify(row)
        if label is None or row["subject_id"] in subjects or counts[label] >= per_class:
            continue
        row.update(label=label, matched_report=evidence)
        selected.append(row)
        subjects.add(row["subject_id"])
        counts[label] += 1
        if all(n == per_class for n in counts.values()):
            break
    return selected, scanned


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fetch_prefix(root: Path, name: str, blocks: int) -> tuple[bytes, dict]:
    path = root / "metadata" / f"{name}.prefix-{blocks}MiB"
    sidecar = path.with_suffix(path.suffix + ".json")
    if path.exists() and sidecar.exists():
        data, provenance = path.read_bytes(), json.loads(sidecar.read_text())
        if sha256(data) == provenance["sha256"]:
            return data, provenance
    data, provenance = get(BASE + name, blocks * MIB, (0, blocks * MIB - 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    provenance["local_path"] = str(path.relative_to(root))
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    return data, provenance


def fetch_file(root: Path, relative: str, max_bytes: int = 4 * MIB) -> dict:
    path = root / relative
    sidecar = root / "metadata" / "downloads" / (relative.replace("/", "_") + ".json")
    if path.exists() and sidecar.exists():
        provenance = json.loads(sidecar.read_text())
        if sha256(path.read_bytes()) == provenance["sha256"]:
            return provenance
    data, provenance = get(BASE + relative, max_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    provenance["local_path"] = relative
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/mimic"))
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--metadata-max-mib", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.per_class < 1 or args.metadata_max_mib < 1 or not 1 <= args.workers <= 8:
        parser.error("positive counts required; workers must be between 1 and 8")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_files = []
    for blocks in range(1, args.metadata_max_mib + 1):
        data, provenance = fetch_prefix(root, "machine_measurements.csv", blocks)
        metadata_files.append(provenance)
        selected, scanned = select(data, args.per_class)
        if len(selected) == 2 * args.per_class:
            break
    else:
        raise RuntimeError("Insufficient qualifying patients within metadata byte budget")
    wanted = {r["study_id"] for r in selected}
    for blocks in range(1, args.metadata_max_mib + 1):
        data, provenance = fetch_prefix(root, "record_list.csv", blocks)
        metadata_files.append(provenance)
        reader = csv_prefix(data)
        records = [r for r in reader if r["study_id"] in wanted]
        if len(records) == len(selected):
            record_fields = reader.fieldnames
            break
    else:
        raise RuntimeError("Record paths missing within metadata byte budget")
    by_study = {r["study_id"]: r for r in records}
    for row in selected:
        record = by_study[row["study_id"]]
        path = PurePosixPath(record["path"])
        if (record["subject_id"] != row["subject_id"] or path.is_absolute()
                or ".." in path.parts or path.name != row["study_id"]):
            raise ValueError("Unexpected remote record path or subject mapping")
    records = [by_study[row["study_id"]] for row in selected]
    write_csv(root / "record_list.csv", records, record_fields)
    write_csv(root / "labels.csv", selected, ["study_id", "label"])
    write_csv(root / "label_evidence.csv", [dict(
        study_id=r["study_id"], subject_id=r["subject_id"], label=r["label"],
        matched_report=r["matched_report"],
        full_report=" || ".join(r.get(k, "") for k in REPORT_FIELDS if r.get(k, "")),
        label_source="machine_measurements.csv report_0..report_17",
    ) for r in selected], ["study_id", "subject_id", "label", "matched_report", "full_report", "label_source"])
    write_csv(root / "machine_measurements_subset.csv", selected,
              [k for k in selected[0] if k not in {"label", "matched_report"}])
    filenames = [r["path"] + extension for r in records for extension in (".hea", ".dat")]
    filenames += ["LICENSE.txt", "machine_measurements_data_dictionary.csv"]
    print(f"Selected {len(selected)} ECGs / unique subjects after scanning {scanned} metadata rows.", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        downloads = list(executor.map(lambda name: fetch_file(root, name), filenames))
    for record in records:
        stem = root / record["path"]
        header = stem.with_suffix(".hea").read_text().splitlines()[0].split()
        if (int(header[1]), float(header[2]), int(header[3])) != (12, 500.0, 5000):
            raise ValueError(f"Unexpected signal geometry: {record['study_id']}")
        if stem.with_suffix(".dat").stat().st_size != 12 * 5000 * 2:
            raise ValueError(f"Truncated or unexpected WFDB format-16 data: {record['study_id']}")
    summary = {
        "dataset": "MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset",
        "version": "1.0", "source": "https://physionet.org/content/mimic-iv-ecg/1.0/",
        "doi": "https://doi.org/10.13026/4nqg-sb35", "license": "ODbL 1.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "affirmative machine-reported PVC/ventricular ectopy vs strict normal sinus rhythm ECG",
        "label_type": "weak machine-generated report label; not clinician-validated ground truth",
        "selection": "first eligible record per selected subject in metadata order until each quota fills",
        "selection_is_population_representative": False,
        "records": len(records), "subjects": len({r['subject_id'] for r in selected}),
        "per_class": args.per_class, "metadata_rows_scanned": scanned,
        "waveform_file_count": len(records) * 2,
        "signal_duration_seconds": 10, "sampling_rate_hz": 500, "leads": 12,
        "metadata_prefixes": metadata_files, "downloads": downloads,
        "download_bytes": sum(r["bytes"] for r in metadata_files + downloads),
        "sha256_note": "Locally calculated content hashes, not verified against the 176 MB publisher checksum index.",
        "study_ids": [r["study_id"] for r in selected],
    }
    (root / "acquisition.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("records", "subjects", "per_class", "download_bytes")}))
    print(f"Ready: {root}")


if __name__ == "__main__":
    main()
