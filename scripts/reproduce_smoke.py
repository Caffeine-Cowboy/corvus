#!/usr/bin/env python3
"""Exercise supervised and strict-transfer paths using artificial fixtures only."""
import argparse
import csv
from pathlib import Path

from corvus.data import FIELDS
from corvus.experiment import evaluate, train
from corvus.repeatability import run
from corvus.synthetic import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True, help="Fresh private/ignored output directory")
    args = p.parse_args()
    if args.work.exists():
        p.error("work directory must be new")
    root = generate(args.work / "artificial-data", seed=20261010)
    astro, ecg = root / "astro/manifest.csv", root / "ecg/manifest.csv"
    train(astro, args.work / "transfer-training", epochs=2, seed=17)
    evaluate(args.work / "transfer-training/checkpoint.pt", ecg,
             args.work / "transfer-evaluation", bootstrap=32)
    with ecg.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = sorted({r["group_id"] for r in rows})
    assignment = {g: "train" if i < 14 else "val" if i < 19 else "test"
                  for i, g in enumerate(groups)}
    for row in rows:
        row["split"] = assignment[row["group_id"]]
    with ecg.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = run(ecg, args.work / "supervised-training", args.work / "synthetic-aggregate.json",
                 epochs=2, seeds=(17, 42), bootstrap=32)
    print("Synthetic smoke passed; no MIMIC/CHIME records, empirical biomedical validity or transfer conclusion.")
    print(f"Artificial test fixtures: {report['protocol']['split_counts']['test']['records']} records")


if __name__ == "__main__":
    main()
