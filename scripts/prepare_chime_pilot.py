#!/usr/bin/env python3
"""Propose source-only CHIME windows, plot them, then explicitly finalize review.

With no arguments, creates data/review/chime (no training manifest). Inspect its
source_qa.png, then pass --finalize to publish data/prepared/chime. Raw files are
opened read-only. No target ECG data or model scores are loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import h5py
import numpy as np

EVENTS = ["FRB20180725A", "FRB20180729A", "FRB20180810B", "FRB20180904A",
          "FRB20180909A", "FRB20180910A", "FRB20180911A", "FRB20180916A"]
FIELDS = ["record_id", "path", "group_id", "parent_id", "domain", "label", "split", "sample_rate_hz"]
WIDTH, GUARD = 32, 8
TUTORIAL = "https://chime-frb-open-data.github.io/waterfall/"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_value(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_catalog(path: Path) -> dict:
    data = path.read_bytes()
    text = data[:data.rfind(b"\n") + 1].decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    result = {}
    for event in EVENTS:
        matches = [r for r in rows if r.get("tns_name") == event]
        if not matches:
            raise ValueError(f"Catalog prefix does not contain {event}")
        groups = {r.get("repeater_name", "").strip() or event for r in matches}
        if len(groups) != 1:
            raise ValueError(f"Inconsistent repeater associations for {event}")
        result[event] = {"group_id": groups.pop(), "catalog_rows": [
            {k: r.get(k, "") for k in ["tns_name", "previous_name", "repeater_name", "event_id",
                                       "sub_num", "ra", "dec", "dm_fitb", "catalog1_flag"]}
            for r in matches]}
    return result


def read_trace(path: Path, expected_event: str) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as handle:
        frb = handle["frb"]
        attrs = {key: json_value(value) for key, value in frb.attrs.items()}
        if attrs.get("tns_name") != expected_event:
            raise ValueError(f"Unexpected event name inside {path}")
        waterfall = np.asarray(frb["calibrated_wfall"], dtype=np.float64)
        frequencies = np.asarray(frb["plot_freq"], dtype=np.float64)
        plot_time = np.asarray(frb["plot_time"], dtype=np.float64)
        if waterfall.ndim != 2 or waterfall.shape[0] != len(frequencies):
            raise ValueError(f"Frequency axis mismatch: {path}")
        if np.isinf(waterfall).any() or not np.isfinite(waterfall).any(axis=0).all():
            raise ValueError(f"Invalid or completely masked time bins: {path}")
        signal = np.nanmean(waterfall, axis=0)
        diffs = np.diff(plot_time)
        dt_ms = float(np.median(diffs))
        cadence_multiple = int(round(dt_ms / 0.98304))
        if (not len(diffs) or not np.isfinite(diffs).all() or dt_ms <= 0
                or not np.allclose(diffs, dt_ms, rtol=2e-4, atol=1e-7)
                or cadence_multiple < 1
                or not np.isclose(dt_ms, cadence_multiple * 0.98304, rtol=2e-4)):
            raise ValueError(f"Unexpected CHIME time cadence: {path}")
        metadata = {
            "raw_filename": path.name, "raw_sha256": digest(path), "raw_bytes": path.stat().st_size,
            "hdf5_attributes": attrs, "measured_dataset": "frb/calibrated_wfall",
            "frequency_collapse": "numpy.nanmean(calibrated_wfall, axis=0), float64, existing NaN masks",
            "waterfall_shape": list(waterfall.shape), "native_samples": len(signal),
            "finite_channel_fraction": float(np.isfinite(waterfall).mean()),
            "frequency_channels": len(frequencies),
            "frequency_range_mhz": [float(np.nanmin(frequencies)), float(np.nanmax(frequencies))],
            "plot_time_samples": len(plot_time), "raw_ts_samples": len(frb["ts"]),
            "plot_time_first_values": plot_time[:5].tolist(),
            "sample_interval_ms": dt_ms, "sample_rate_hz": 1000.0 / dt_ms,
            "cadence_multiple_of_0_98304_ms": cadence_multiple,
            "timebase_interpretation": (
                "Use median(diff(plot_time)) as milliseconds for the full calibrated waterfall, "
                "following the official calibrated-plot tutorial's shared dt and ms axis. "
                "The tutorial metadata bullet says microseconds inconsistently; ms is the locked "
                "operational interpretation and matches integer multiples of CHIME 0.98304 ms cadence. "
                "Do not stretch the shorter plot_time extent onto the longer calibrated array. "
                "Native sample indices define annotations; downstream normalization is dimensionless."),
            "timebase_reference": TUTORIAL,
        }
    return signal, metadata


def propose_windows(signal: np.ndarray) -> dict:
    peak, n = int(np.argmax(signal)), len(signal)
    centered = int(np.clip(peak - WIDTH // 2, 0, n - WIDTH))
    candidates = []
    for positive_start in range(max(0, peak - WIDTH + 1), min(peak, n - WIDTH) + 1):
        positive_end = positive_start + WIDTH
        for background_start in (0, n - WIDTH):
            background_end = background_start + WIDTH
            gap = max(positive_start - background_end, background_start - positive_end)
            if gap >= GUARD:
                candidates.append((abs(positive_start - centered),
                                   -abs((background_start + (WIDTH - 1) / 2) - peak),
                                   positive_start, background_start, gap))
    if not candidates:
        raise ValueError(f"No nonoverlapping {WIDTH}-sample pair with guard {GUARD}; n={n}, peak={peak}")
    shift, _, positive_start, background_start, gap = min(candidates)
    if shift > WIDTH // 4:
        raise ValueError(f"Feasible burst window would shift too far from the peak center ({shift} samples)")
    return {"peak_sample": peak, "centered_start_proposal": centered,
            "center_shift_samples": positive_start - centered,
            "positive": [positive_start, positive_start + WIDTH],
            "background": [background_start, background_start + WIDTH],
            "guard_samples": gap,
            "selection_rule": "Peak-centered 32 samples; smallest shift needed for feasible background. "
                              "Choose the farthest feasible endpoint background of 32 samples; no noise-score optimization."}


def load_all(root: Path, catalog: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    missing = [str(root / f"{event}_waterfall.h5") for event in EVENTS
               if not (root / f"{event}_waterfall.h5").is_file()]
    if missing:
        raise FileNotFoundError("Source downloads are incomplete; missing:\n" + "\n".join(missing))
    grouping = load_catalog(catalog)
    records, signals, group_splits = [], {}, {}
    for index, event in enumerate(EVENTS):
        signal, metadata = read_trace(root / f"{event}_waterfall.h5", event)
        split = "train" if index < 4 else "val" if index < 6 else "test"
        group_id = grouping[event]["group_id"]
        if group_id in group_splits and group_splits[group_id] != split:
            raise ValueError(f"Physical source crosses planned splits: {group_id}; revise grouping before training")
        group_splits[group_id] = split
        records.append({"event": event, "split": split, **grouping[event],
                        "windows": propose_windows(signal), "metadata": metadata})
        signals[event] = signal
    return records, signals


def plot_qa(records: list[dict], signals: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    fig.suptitle("CHIME/FRB source-only window review — measured frequency-mean calibrated intensity", fontsize=15)
    for ax, record in zip(axes.flat, records):
        signal, windows = signals[record["event"]], record["windows"]
        ps, pe = windows["positive"]
        bs, be = windows["background"]
        ax.axvspan(max(-0.5, ps - GUARD - 0.5), min(len(signal) - 0.5, pe + GUARD - 0.5),
                   color="#f2c94c", alpha=0.14)
        ax.axvspan(ps - 0.5, pe - 0.5, color="#e58430", alpha=0.27)
        ax.axvspan(bs - 0.5, be - 0.5, color="#148a85", alpha=0.22)
        ax.plot(np.arange(len(signal)), signal, color="#26384d", linewidth=1)
        ax.scatter([windows["peak_sample"]], [signal[windows["peak_sample"]]], color="#bb4b17", s=19)
        ax.set_title(f"{record['event']} · {record['split']} · group {record['group_id']}\n"
                     f"burst [{ps},{pe}) / background [{bs},{be}) · gap {windows['guard_samples']} samples · "
                     f"dt {record['metadata']['sample_interval_ms']:.5f} ms", fontsize=10)
        ax.set_xlabel("Native calibrated time-bin index")
        ax.set_ylabel("Frequency-mean flux (Jy)")
        ax.set_xlim(-0.5, len(signal) - 0.5)
        ax.grid(alpha=0.15)
    fig.legend(handles=[Patch(color="#e58430", alpha=0.4, label="Burst window: 32 samples"),
                        Patch(color="#148a85", alpha=0.4, label="Background proxy: 32 samples"),
                        Patch(color="#f2c94c", alpha=0.3, label="Minimum 8-sample exclusion guard")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stage_for(output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))


def propose(root: Path, catalog: Path, review: Path) -> None:
    records, signals = load_all(root, catalog)
    staging = stage_for(review)
    try:
        plot_qa(records, signals, staging / "source_qa.png")
        proposal = {
            "status": "UNREVIEWED_PROPOSALS_ONLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "CHIME/FRB Catalog 1 calibrated measured waterfalls, not CHIME/Pulsar",
            "raw_directory": str(root), "catalog_path": str(catalog), "catalog_sha256": digest(catalog),
            "width_samples": WIDTH, "minimum_guard_samples": GUARD, "indices": "zero-based, end-exclusive",
            "source_qa_sha256": digest(staging / "source_qa.png"),
            "review_instructions": "Inspect all full traces and proposed intervals in source_qa.png. "
                                   "Check burst selection, off-pulse contamination, grouping and masking. "
                                   "Only after source-only review, explicitly call this script with --finalize.",
            "limitations": ["Off-burst windows are weak background proxies, not independent verified noise/RFI labels.",
                            "Catalog 2 repeater associations group these Catalog 1 files; unknown repetition can remain.",
                            "Eight observations are a convenience pilot, not a representative source population.",
                            "Absolute time units are operationally interpreted as ms; see per-file timebase notes."],
            "observations": records,
        }
        write_json(staging / "proposals.json", proposal)
        if review.exists():
            raise FileExistsError(f"Review directory was created concurrently: {review}")
        staging.rename(review)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"UNREVIEWED proposals: {review / 'proposals.json'}")
    print(f"Review all source traces: {review / 'source_qa.png'}")
    print("No training manifest has been created. After source-only review, rerun with --finalize.")


def finalize(root: Path, catalog: Path, review: Path, output: Path) -> None:
    proposal_path = review / "proposals.json"
    proposal = json.loads(proposal_path.read_text())
    records, signals = load_all(root, catalog)
    if (proposal.get("status") != "UNREVIEWED_PROPOSALS_ONLY"
            or proposal.get("width_samples") != WIDTH or proposal.get("minimum_guard_samples") != GUARD
            or proposal.get("catalog_sha256") != digest(catalog)
            or proposal.get("source_qa_sha256") != digest(review / "source_qa.png")
            or proposal.get("observations") != records):
        raise ValueError("Proposals, QA figure, raw content, or source grouping changed; create and review fresh proposals")
    staging = stage_for(output)
    try:
        (staging / "signals").mkdir()
        rows, annotations = [], []
        for record in records:
            for kind, label in (("positive", 1), ("background", 0)):
                start, end = record["windows"][kind]
                record_id = f"{record['event']}_{kind}"
                relative = f"signals/{record_id}.npy"
                window = signals[record["event"]][start:end].astype(np.float32)
                if len(window) != WIDTH or not np.isfinite(window).all() or np.std(window) < 1e-8:
                    raise ValueError(f"Invalid source window: {record_id}")
                np.save(staging / relative, window, allow_pickle=False)
                rows.append([record_id, relative, record["group_id"], record["event"], "astro", label,
                             record["split"], record["metadata"]["sample_rate_hz"]])
                annotations.append({"record_id": record_id, "start_sample": start, "end_sample": end,
                                    "label": label, "label_meaning": "burst" if label else "off-burst background proxy",
                                    "group_id": record["group_id"], "split": record["split"],
                                    "raw_filename": record["metadata"]["raw_filename"],
                                    "raw_sha256": record["metadata"]["raw_sha256"],
                                    "signal_sha256": digest(staging / relative)})
        with (staging / "manifest.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        write_json(staging / "annotations.json", annotations)
        write_json(staging / "provenance.json", {
            **proposal, "status": "FINALIZED_AFTER_EXPLICIT_SOURCE_QA_REVIEW",
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "review_confirmation": "Explicit --finalize invocation after inspection of source QA figure",
            "reviewed_proposal_sha256": digest(proposal_path),
            "manifest_sha256": digest(staging / "manifest.csv"),
            "observations_count": len(records), "windows_count": len(rows),
            "split_groups": {split: len({r['group_id'] for r in records if r['split'] == split})
                             for split in ("train", "val", "test")},
        })
        shutil.copyfile(review / "source_qa.png", staging / "source_qa.png")
        if output.exists():
            raise FileExistsError(f"Output directory was created concurrently: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Final source manifest: {output / 'manifest.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/chime"))
    parser.add_argument("--catalog", type=Path, default=Path("data/raw/chime/chimefrbcat2.csv.prefix-64KiB"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/review/chime"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/chime"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--review-only", action="store_true", help="Default: write proposals and QA figure only")
    modes.add_argument("--finalize", action="store_true", help="Confirm source-only visual review and create training manifest")
    args = parser.parse_args()
    try:
        if args.finalize:
            finalize(args.root.resolve(), args.catalog.resolve(), args.review_dir.resolve(), args.output.resolve())
        else:
            propose(args.root.resolve(), args.catalog.resolve(), args.review_dir.resolve())
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
