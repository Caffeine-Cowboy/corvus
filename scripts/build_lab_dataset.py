#!/usr/bin/env python3
"""Prepare a separate supervised cohort; exclude every original pilot patient."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from corvus.data import read_manifest, sha256, transform
from corvus.prepare import prepare_mimic


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/mimic-lab"))
    p.add_argument("--exclude", type=Path, default=Path("data/raw/mimic/record_list.csv"))
    p.add_argument("--output", type=Path, default=Path("data/prepared/ecg-lab"))
    args = p.parse_args()
    excluded = {r["subject_id"] for r in read(args.exclude)}
    all_records = read(args.raw / "record_list.csv")
    records = [r for r in all_records if r["subject_id"] not in excluded]
    wanted = {r["study_id"] for r in records}
    labels = [r for r in read(args.raw / "labels.csv") if r["study_id"] in wanted]
    if len(records) != len({r["subject_id"] for r in records}):
        raise ValueError("This cohort requires one ECG per patient")
    selection = args.output.parent / (args.output.name + "-selection")
    selection.mkdir(parents=True, exist_ok=False)
    for row in labels:
        row["label_provenance"] = "machine report PVC / strict normal; weak labels"
    write(selection / "record_list.csv", records)
    write(selection / "labels.csv", labels)
    prepare_mimic(args.raw, selection / "record_list.csv", selection / "labels.csv", args.output, lead="II")
    manifest = args.output / "manifest.csv"
    rows = read(manifest)
    accepted, quality_exclusions = [], []
    for row in rows:
        try:
            transform(np.load(args.output / row["path"], allow_pickle=False))
        except ValueError as exc:
            quality_exclusions.append({"record_id": row["record_id"], "reason": str(exc)})
        else:
            accepted.append(row)
    rows = accepted
    rng = np.random.default_rng(173)
    for label in ("0", "1"):
        class_rows = sorted([r for r in rows if r["label"] == label], key=lambda r: r["record_id"])
        if len(class_rows) < 50:
            raise ValueError("Need at least 50 patients per class")
        order = rng.permutation(len(class_rows))
        train_end, val_end = int(0.6 * len(order)), int(0.8 * len(order))
        for position, index in enumerate(order):
            class_rows[index]["split"] = "train" if position < train_end else "val" if position < val_end else "test"
    write(manifest, rows)
    read_manifest(manifest, "ecg")
    audit = {"endpoint": "Machine-reported PVC evidence versus strict normal ECG",
             "raw_records": len(all_records), "excluded_original_pilot_patients": len(all_records) - len(records),
             "exclusion_manifest_sha256": sha256(args.exclude), "patients": len(rows), "split_seed": 173,
             "model_input_quality_exclusions": quality_exclusions,
             "split_policy": "60/20/20 stratified patient split; one record per patient",
             "manifest_sha256": sha256(manifest),
             "splits": {s: {"patients": sum(r["split"] == s for r in rows),
                              "positive": sum(r["split"] == s and r["label"] == "1" for r in rows)}
                        for s in ("train", "val", "test")},
             "limitations": "Convenience sample; machine weak labels; positive cases may have comorbid abnormalities."}
    (args.output / "cohort.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
