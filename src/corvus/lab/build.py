"""Freeze and benchmark a self-contained, versioned serving bundle."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time

import numpy as np
import torch

from corvus.data import read_manifest, sha256, transform, waveform_inventory
from corvus.experiment import empty_directory, write_json
from corvus.metrics import binary_metrics, cluster_bootstrap
from corvus.model import TransientCNN
from .runtime import Predictor
from .stress import SCENARIOS, corrupt, release_gate, stable_seed


def latency(predictor, x, repeats=200):
    for _ in range(20):
        predictor.score(x[:1])
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        predictor.score(x[:1])
        durations.append((time.perf_counter() - start) * 1000)
    batched = np.repeat(x[:1], 32, axis=0)
    for _ in range(5):
        predictor.score(batched)
    start = time.perf_counter()
    for _ in range(30):
        predictor.score(batched)
    elapsed = time.perf_counter() - start
    return {"p50_ms": float(np.median(durations)), "p95_ms": float(np.quantile(durations, 0.95)),
            "p99_ms": float(np.quantile(durations, 0.99)), "batch_32_records_per_second": 960 / elapsed,
            "warmups": 20, "repeats": repeats, "scope": "Warm CPU model inference only; excludes preprocessing and HTTP"}


def build_bundle(manifest, training, astro_checkpoint, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite serving bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".lab-build-", dir=output.parent))
    try:
        result = _build_bundle(manifest, training, astro_checkpoint, staging)
        staging.rename(output)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_bundle(manifest, training, astro_checkpoint, output):
    torch.set_num_threads(2)
    records = read_manifest(manifest, "ecg")
    test = [r for r in records if r.split == "test"]
    training, output = Path(training), empty_directory(output)
    trained = json.loads((training / "training.json").read_text())
    if trained["manifest_sha256"] != sha256(manifest) or trained["test_used_for_selection"]:
        raise ValueError("Training provenance does not match this held-out cohort")
    test_inventory = waveform_inventory(test)
    used = {r["signal_sha256"] for r in trained["training_waveforms"] + trained["validation_waveforms"]}
    if used.intersection(r["signal_sha256"] for r in test_inventory):
        raise ValueError("Training/validation waveform overlaps test cohort")
    raw = np.stack([np.load(r.path, allow_pickle=False) for r in test])
    rates = np.array([r.sample_rate_hz for r in test])
    if raw.shape[1] != 5000 or not np.all(rates == 500):
        raise ValueError("This serving release supports the acquired 10-second, 500 Hz cohort")
    labels = np.array([r.label for r in test], dtype=np.int64)
    groups = [r.group_id for r in test]
    np.savez_compressed(output / "examples.npz", signals=raw, labels=labels, sample_rates=rates,
                        record_ids=np.array([r.record_id for r in test]))
    shutil.copyfile(training / "weights.pt", output / "ecg.pt")
    shutil.copyfile(training / "training.json", output / "training.json")
    astro = torch.load(astro_checkpoint, map_location="cpu", weights_only=True)
    torch.save(astro["state_dict"], output / "astro.pt")
    model = TransientCNN()
    model.load_state_dict(torch.load(output / "ecg.pt", weights_only=True))
    model.eval()
    print("Exporting the frozen supervised model to ONNX", flush=True)
    torch.onnx.export(model, torch.zeros(2, 1, 1024), output / "ecg.onnx",
                      input_names=["signal"], output_names=["logits"],
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
                      opset_version=18, dynamo=True, external_data=False)
    specs = [
        {"id": "ecg-cnn", "title": "ECG CNN", "runtime": "pytorch", "artifact": "ecg.pt",
         "threshold": trained["threshold"], "training_domain": "ECG", "description": "Supervised ECG baseline"},
        {"id": "ecg-onnx", "title": "ECG CNN · ONNX", "runtime": "onnx", "artifact": "ecg.onnx",
         "threshold": trained["threshold"], "training_domain": "ECG", "description": "Same weights, optimized serving runtime"},
        {"id": "astro-cnn", "title": "Astronomy CNN", "runtime": "pytorch", "artifact": "astro.pt",
         "threshold": astro["threshold"], "training_domain": "CHIME/FRB", "description": "Original zero-shot transfer candidate"},
    ]
    predictors = {s["id"]: Predictor(output / s["artifact"], s["runtime"]) for s in specs}
    clean_x = np.stack([transform(x) for x in raw])
    clean_scores = {key: p.batch(clean_x) for key, p in predictors.items()}
    parity = float(np.max(np.abs(clean_scores["ecg-cnn"] - clean_scores["ecg-onnx"])))
    if parity > 1e-4:
        raise ValueError(f"ONNX score parity failed: {parity}")
    measurements = {key: latency(p, clean_x) for key, p in predictors.items()}
    prep_times = []
    for i in range(200):
        start = time.perf_counter()
        transform(raw[i % len(raw)])
        prep_times.append((time.perf_counter() - start) * 1000)
    results = {s["id"]: {"clean": binary_metrics(labels, clean_scores[s["id"]], s["threshold"]),
                           "bootstrap": cluster_bootstrap(labels, clean_scores[s["id"]], groups, s["threshold"], 500, 42),
                           "latency": measurements[s["id"]], "scenarios": []} for s in specs}
    for scenario in SCENARIOS:
        x = np.stack([transform(corrupt(wave, rate, scenario["kind"], scenario["severity"], stable_seed(r.record_id))[0])
                      for r, wave, rate in zip(test, raw, rates)])
        for spec in specs:
            values = predictors[spec["id"]].batch(x)
            result = {**scenario, **binary_metrics(labels, values, spec["threshold"])}
            results[spec["id"]]["scenarios"].append(result)
        print(f"Evaluated {scenario['id']} on {len(test)} held-out patients", flush=True)
    for spec in specs:
        result = results[spec["id"]]
        noise = next(s for s in result["scenarios"] if s["id"] == "noise-25")
        error = parity if spec["runtime"] == "onnx" else None
        spec["gate"] = release_gate(result["clean"], noise, result["latency"]["p95_ms"], error)
        spec["sha256"] = sha256(output / spec["artifact"])
        spec["version"] = spec["id"] + ":" + spec["sha256"][:10]
        spec["artifact_bytes"] = (output / spec["artifact"]).stat().st_size
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "MIMIC-IV-ECG v1.0",
              "endpoint": "Machine-reported PVC versus strictly normal report",
              "cohort": json.loads((Path(manifest).parent / "cohort.json").read_text()),
              "test_patients": len(test), "test_positive": int(labels.sum()),
              "test_inventory": test_inventory, "models": results, "onnx_max_score_error": parity,
              "preprocessing_p95_ms": float(np.quantile(prep_times, 0.95)),
              "environment": {"platform": platform.platform(), "processor": platform.machine(),
                              "python": platform.python_version(), "torch": str(torch.__version__), "cpu_threads": 2},
              "notes": ["Weak machine labels; balanced convenience cohort; research demonstration.",
                        "Scores are not calibrated clinical probabilities.",
                        "All scenarios share patients; corruptions simulate nuisance changes.",
                        "Passing these engineering gates is not clinical approval."]}
    write_json(output / "report.json", report)
    registry = {"schema_version": 1, "created_utc": report["created_utc"], "models": specs,
                "report_sha256": sha256(output / "report.json"), "examples_sha256": sha256(output / "examples.npz")}
    write_json(output / "registry.json", registry)
    np.savez_compressed(output / "test_predictions.npz", labels=labels, **clean_scores)
    print(json.dumps({s["id"]: {"auc": results[s["id"]]["clean"]["auroc"], "gate": s["gate"]["passed"]} for s in specs}, indent=2))
    return registry
