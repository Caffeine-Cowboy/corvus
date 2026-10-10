#!/usr/bin/env python3
"""Export a standalone research figure from already-frozen pilot results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, default=Path("runs/real_seed42/ecg"))
    p.add_argument("--shuffled-evaluation", type=Path)
    p.add_argument("--output", type=Path, default=Path("reports/pilot"))
    args = p.parse_args()
    report = json.loads((args.evaluation / "metrics.json").read_text())
    rows = list(csv.DictReader((args.evaluation / "predictions.csv").open()))
    y = np.array([int(r["label"]) for r in rows])
    synthetic = any(r["record_id"].startswith("synthetic_") for r in rows)
    names = [("astro_cnn", "Astronomy-trained CNN", "astro_score", "#156b8a"),
             ("random_cnn", "Random CNN", "random_score", "#777777"),
             ("crest_factor", "Crest factor", "crest_score", "#bd7832")]
    series = [(label, np.array([float(r[col]) for r in rows]), report["models"][key], color)
              for key, label, col, color in names]
    if args.shuffled_evaluation:
        shuffled_report = json.loads((args.shuffled_evaluation / "metrics.json").read_text())
        shuffled = list(csv.DictReader((args.shuffled_evaluation / "predictions.csv").open()))
        if [(r["record_id"], r["label"]) for r in rows] != [(r["record_id"], r["label"]) for r in shuffled]:
            raise ValueError("Cohorts must match for plotting")
        series.append(("Shuffled source labels", np.array([float(r["astro_score"]) for r in shuffled]),
                       shuffled_report["models"]["astro_cnn"], "#927197"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for i, (label, values, metrics, color) in enumerate(series):
        fpr, tpr, _ = roc_curve(y, values)
        precision, recall, _ = precision_recall_curve(y, values)
        axs[0].plot(fpr, tpr, color=color, label=label, linewidth=1.8)
        axs[1].step(recall, precision, where="post", color=color, linewidth=1.8)
        low, high = metrics["bootstrap"]["intervals_95_percent"]["auroc"]
        auc = metrics["auroc"]
        axs[2].plot([low, high], [i, i], color=color, linewidth=2)
        axs[2].scatter([auc], [i], color=color, s=35, zorder=3)
        axs[2].text(1.03, i, f"{auc:.3f} [{low:.3f}, {high:.3f}]", va="center", fontsize=9)
    axs[0].plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", linewidth=1)
    axs[1].axhline(y.mean(), color="#aaaaaa", linestyle="--", linewidth=1)
    axs[0].set(title="ECG receiver operating characteristic", xlabel="False-positive rate (fraction)", ylabel="True-positive rate (fraction)")
    axs[1].set(title="ECG precision–recall", xlabel="Recall (fraction)", ylabel="Precision (fraction)")
    for ax in axs[:2]:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axs[2].axvline(0.5, color="#aaaaaa", linestyle="--", linewidth=1)
    axs[2].set(title="AUROC and patient-bootstrap 95% intervals", xlabel="AUROC (fraction)", ylabel="Frozen score")
    axs[2].set_yticks(range(len(series)), [s[0] for s in series], fontsize=9)
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(len(series) - 0.5, -0.5)
    axs[0].legend(loc="lower right", fontsize=8, frameon=False)
    title = (f"Synthetic software test | {len(y)} artificial records — no clinical evidence" if synthetic
             else f"CHIME/FRB → MIMIC-IV-ECG | {len(y)}-patient PVC weak-label pilot")
    fig.suptitle(title, x=0.06, ha="left", fontsize=16)
    fig.text(0.06, 0.88, "Source-only training and thresholds · No ECG adaptation · Convenience cohort, not clinical validation", fontsize=10)
    caption = ("Source: generated synthetic fixtures; these are not observed CHIME or MIMIC recordings.\n"
               "Software verification only. Synthetic performance cannot demonstrate real cross-domain transfer." if synthetic else
               "Sources: CHIME/FRB Catalog 1 and MIMIC-IV-ECG v1.0. Selected acquisition cohort; no longitudinal analysis.\n"
               f"Machine-report PVC versus strict normal labels; {y.mean():.0%} selected prevalence. Intervals condition on one training seed and this cohort.")
    fig.text(0.06, 0.03, caption, fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.06, right=0.87, bottom=0.22, top=0.77, wspace=0.65)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output) + ".png", dpi=180)
    fig.savefig(str(args.output) + ".svg")
    print(str(args.output) + ".png")


if __name__ == "__main__":
    main()
