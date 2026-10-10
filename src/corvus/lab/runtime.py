"""Immutable model runtimes; all serving artifacts are hash checked."""
import json
from pathlib import Path
import threading

import numpy as np
import torch

from corvus.data import sha256, transform
from corvus.model import TransientCNN
from .stress import corrupt, stable_seed


class Predictor:
    def __init__(self, path, runtime="pytorch"):
        self.runtime = runtime
        if runtime == "pytorch":
            self.model = TransientCNN()
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.model.eval().requires_grad_(False)
        elif runtime == "onnx":
            import onnxruntime as ort
            settings = ort.SessionOptions()
            settings.intra_op_num_threads = 2
            settings.inter_op_num_threads = 1
            self.model = ort.InferenceSession(str(path), sess_options=settings, providers=["CPUExecutionProvider"])
        else:
            raise ValueError("Unknown inference runtime")

    @torch.inference_mode()
    def score(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.runtime == "pytorch":
            return torch.sigmoid(self.model(torch.from_numpy(x))).numpy()
        logits = self.model.run(None, {"signal": x})[0]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))

    def batch(self, x, size=32):
        return np.concatenate([self.score(x[i:i + size]) for i in range(0, len(x), size)])


class LabService:
    def __init__(self, directory):
        torch.set_num_threads(2)
        self.directory = Path(directory).resolve()
        self.registry = json.loads((self.directory / "registry.json").read_text())
        if self.registry.get("schema_version") != 1:
            raise ValueError("Unsupported registry schema")
        for name, key in [("report.json", "report_sha256"), ("examples.npz", "examples_sha256")]:
            if sha256(self.directory / name) != self.registry[key]:
                raise ValueError(f"Artifact integrity failure: {name}")
        self.report = json.loads((self.directory / "report.json").read_text())
        with np.load(self.directory / "examples.npz", allow_pickle=False) as bundle:
            self.signals = bundle["signals"]
            self.labels = bundle["labels"]
            self.sample_rates = bundle["sample_rates"]
            self.original_ids = bundle["record_ids"]
        self.records = [{"id": f"ECG-{i + 1:04d}", "label": int(label),
                         "label_text": "PVC reported" if label else "Reported normal",
                         "samples": len(self.signals[i]), "sample_rate_hz": float(self.sample_rates[i])}
                        for i, label in enumerate(self.labels)]
        self.record_index = {r["id"]: i for i, r in enumerate(self.records)}
        self.specs, self.predictors = {}, {}
        self.lock = threading.Lock()
        self.metrics = {"requests": 0, "errors": 0, "inferences": 0, "inference_seconds": 0.0}
        for spec in self.registry["models"]:
            path = (self.directory / spec["artifact"]).resolve()
            if not path.is_relative_to(self.directory) or sha256(path) != spec["sha256"]:
                raise ValueError("Model artifact integrity failure")
            self.specs[spec["id"]] = spec
            self.predictors[spec["id"]] = Predictor(path, spec["runtime"])
            self.predictors[spec["id"]].score(transform(self.signals[0])[None])

    def replay(self, record_id, model_id, kind, severity, seed):
        import time
        if record_id not in self.record_index or model_id not in self.specs:
            raise KeyError("Unknown record or model")
        index = self.record_index[record_id]
        raw, fs = self.signals[index], float(self.sample_rates[index])
        modified, corruption = corrupt(raw, fs, kind, severity, stable_seed(str(self.original_ids[index]), seed))
        start = time.perf_counter()
        prepared = transform(modified)[None]
        preparation_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        score = float(self.predictors[model_id].score(prepared)[0])
        inference_ms = (time.perf_counter() - start) * 1000
        clean_x = transform(raw)[None]
        clean_start = time.perf_counter()
        clean_score = float(self.predictors[model_id].score(clean_x)[0])
        clean_ms = (time.perf_counter() - clean_start) * 1000
        with self.lock:
            self.metrics["inferences"] += 2
            self.metrics["inference_seconds"] += (inference_ms + clean_ms) / 1000
        spec = self.specs[model_id]
        return {"record": self.records[index], "model_id": model_id, "model_version": spec["version"],
                "score": score, "clean_score": clean_score, "threshold": spec["threshold"],
                "prediction": int(score > spec["threshold"]), "clean_prediction": int(clean_score > spec["threshold"]),
                "agrees_with_weak_label": int(score > spec["threshold"]) == int(self.labels[index]),
                "corruption": corruption, "inference_ms": inference_ms, "preprocessing_ms": preparation_ms,
                "clean_signal": raw.tolist(), "signal": modified.tolist(),
                "release_passed": spec["gate"]["passed"], "sandbox": True}
