import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from corvus.data import sha256
from corvus.lab.api import MAX_BODY_BYTES, create_app
from corvus.lab.runtime import LabService
from corvus.lab.stress import KINDS, corrupt, release_gate, stable_seed
from corvus.lab.train import validation_threshold
from corvus.model import TransientCNN


@pytest.mark.parametrize("kind", KINDS)
def test_corruption_preserves_shape_and_is_repeatable(kind):
    x = np.sin(np.arange(5000) / 17).astype(np.float32)
    a, _ = corrupt(x, 500, kind, 0.7, 42)
    b, _ = corrupt(x, 500, kind, 0.7, 42)
    assert a.shape == x.shape and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    clean, _ = corrupt(x, 500, kind, 0.0, 42)
    np.testing.assert_array_equal(clean, x)
    if kind != "clean":
        assert not np.array_equal(a, x)


def test_release_gate_fails_closed_and_uses_all_criteria():
    clean = {"auroc": 0.9, "balanced_accuracy": 0.8}
    assert release_gate(clean, {"auroc": 0.85}, 1.0, 1e-6)["passed"]
    assert not release_gate(clean, {"auroc": 0.6}, 1.0)["passed"]
    assert not release_gate(clean, {"auroc": 0.85}, float("nan"))["passed"]
    assert not release_gate({"auroc": None}, {}, 1.0)["passed"]
    assert not release_gate(clean, {"auroc": 0.85}, 1.0, 0.001)["passed"]
    assert stable_seed("example", 42) == stable_seed("example", 42)
    assert stable_seed("example", 42) != stable_seed("different", 42)


def test_validation_threshold_uses_specificity_tie_break():
    assert validation_threshold([0, 1], np.array([0.5, 0.5])) == 1.0
    threshold = validation_threshold([0, 0, 1, 1], np.array([0.1, 0.2, 0.7, 0.9]))
    assert 0.2 <= threshold < 0.7


@pytest.fixture
def serving_bundle(tmp_path):
    torch.manual_seed(42)
    weights = tmp_path / "ecg.pt"
    torch.save(TransientCNN().state_dict(), weights)
    t = np.arange(5000) / 500
    signals = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]).astype(np.float32)
    np.savez_compressed(tmp_path / "examples.npz", signals=signals, labels=[0, 1], sample_rates=[500, 500], record_ids=["r0", "r1"])
    (tmp_path / "report.json").write_text(json.dumps({"models": {}, "test_patients": 2}))
    specs = []
    for name, passed in [("ecg-onnx", True), ("astro-cnn", False)]:
        specs.append({"id": name, "artifact": "ecg.pt", "runtime": "pytorch", "threshold": .5,
                      "version": name + ":test", "sha256": sha256(weights), "gate": {"passed": passed, "checks": []}})
    (tmp_path / "registry.json").write_text(json.dumps({"schema_version": 1, "models": specs,
        "report_sha256": sha256(tmp_path / "report.json"), "examples_sha256": sha256(tmp_path / "examples.npz")}))
    return tmp_path


def test_api_real_inference_and_promotion_block(serving_bundle):
    with TestClient(create_app(serving_bundle)) as client:
        assert client.get("/health").json()["models_loaded"] == 2
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        assert len(client.get("/api/v1/records").json()) == 2
        body = {"record_id": "ECG-0001", "model_id": "astro-cnn", "corruption": "noise", "severity": .5}
        replay = client.post("/api/v1/replay", json=body)
        assert replay.status_code == 200
        data = replay.json()
        assert data["sandbox"] and not data["release_passed"]
        assert data["signal"] != data["clean_signal"]
        again = client.post("/api/v1/replay", json=body).json()
        assert again["signal"] == data["signal"] and again["score"] == data["score"]
        signal = data["clean_signal"]
        blocked = client.post("/api/v1/predict", json={"signal": signal, "sample_rate_hz": 500, "model_id": "astro-cnn"})
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["code"] == "model_not_promoted"
        prediction = client.post("/api/v1/predict", json={"signal": signal, "sample_rate_hz": 500})
        assert prediction.status_code == 200
        assert 0 <= prediction.json()["score"] <= 1
        assert "corvus_inferences_total" in client.get("/metrics").text


def test_api_rejects_invalid_payloads(serving_bundle):
    with TestClient(create_app(serving_bundle)) as client:
        assert client.post("/api/v1/replay", json={"record_id": "unknown"}).status_code == 404
        assert client.post("/api/v1/replay", json={"record_id": "ECG-0001", "severity": 2}).status_code == 422
        assert client.post("/api/v1/predict", json={"signal": [1.] * 5000, "sample_rate_hz": 500}).status_code == 422
        assert client.post("/api/v1/predict", json={"signal": [1.] * 2000, "sample_rate_hz": 500}).status_code == 422
        assert client.post("/api/v1/predict", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413
        payload = '{"signal":[' + ','.join(["NaN"] + ["1"] * 4999) + '],"sample_rate_hz":500}'
        assert client.post("/api/v1/predict", content=payload, headers={"Content-Type": "application/json"}).status_code == 422


def test_tampered_model_is_rejected(serving_bundle):
    with (serving_bundle / "ecg.pt").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        LabService(serving_bundle)
