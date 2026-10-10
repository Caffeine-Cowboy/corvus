"""Deliberately artificial fixtures; these cannot establish scientific transfer."""
import csv
from pathlib import Path

import numpy as np

from .data import FIELDS
from .experiment import empty_directory, write_json


def generate(output, seed=42):
    root = empty_directory(output)
    rng = np.random.default_rng(seed)
    for domain, count in [("astro", 36), ("ecg", 24)]:
        folder = root / domain
        folder.mkdir()
        rows = []
        for i in range(count):
            split = ("train" if i < 24 else "val" if i < 30 else "test") if domain == "astro" else "test"
            for label in (0, 1):
                n = 1500 if domain == "astro" else 5000
                t = np.linspace(0, 1, n, endpoint=False)
                x = rng.normal(0, 0.3, n)
                if domain == "astro" and label:
                    center = rng.uniform(0.25, 0.75)
                    x += rng.uniform(2, 4) * np.exp(-0.5 * ((t - center) / 0.012) ** 2)
                elif domain == "ecg":
                    centers = np.arange(0.05, 1, 0.09)
                    if label:
                        centers[5] -= 0.03
                    for j, center in enumerate(centers):
                        width = 0.01 if label and j == 5 else 0.003
                        x += 2 * np.exp(-0.5 * ((t - center) / width) ** 2)
                    x += 0.15 * np.sin(2 * np.pi * 2 * t)
                name = f"synthetic_{domain}_{i:03d}_{label}"
                np.save(folder / f"{name}.npy", x.astype(np.float32))
                rows.append([name, f"{name}.npy", f"synthetic_{domain}_group_{i}",
                             f"synthetic_{domain}_parent_{i}", domain, label, split,
                             100000 if domain == "astro" else 500])
        with (folder / "manifest.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(rows)
    write_json(root / "provenance.json", {"synthetic": True, "seed": seed,
                                          "warning": "Software smoke test only. Not CHIME or MIMIC data."})
    return root
