"""Deterministic nuisance corruptions in physical time, before preprocessing."""
import hashlib
import math

import numpy as np
from scipy.signal import resample_poly

KINDS = ("clean", "noise", "drift", "dropout", "sampling", "gain")


def stable_seed(record_id, seed=42):
    return int.from_bytes(hashlib.sha256(f"{seed}:{record_id}".encode()).digest()[:4], "little")


def corrupt(signal, sample_rate_hz, kind="clean", severity=0.0, seed=42):
    if kind not in KINDS or not math.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("Unknown corruption or severity outside [0,1]")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Invalid sample rate")
    x = np.asarray(signal, dtype=np.float64).copy()
    if x.ndim != 1 or len(x) < 16 or not np.isfinite(x).all():
        raise ValueError("Expected a finite 1D waveform")
    info = {"kind": kind, "severity": float(severity), "seed": int(seed)}
    if kind == "clean" or severity == 0:
        return x.astype(np.float32), {**info, "description": "Original recording"}
    rng = np.random.default_rng(seed)
    rms = max(float(np.std(x)), 1e-8)
    if kind == "noise":
        snr = 30 - 25 * severity
        x += rng.normal(0, rms / (10 ** (snr / 20)), size=len(x))
        info.update(snr_db=snr, description=f"Gaussian noise · {snr:.1f} dB nominal SNR")
    elif kind == "drift":
        phase = rng.uniform(0, 2 * np.pi)
        x += 2 * severity * rms * np.sin(2 * np.pi * 0.33 * np.arange(len(x)) / sample_rate_hz + phase)
        info.update(frequency_hz=0.33, description=f"Baseline drift · {2 * severity:.1f} × signal SD")
    elif kind == "dropout":
        count = max(1, round(len(x) * 0.3 * severity))
        start = int(rng.integers(0, len(x) - count + 1))
        x[start:start + count] = np.median(x)
        info.update(start_sample=start, missing_samples=count,
                    description=f"Missing segment · {count / sample_rate_hz:.2f} s, median filled")
    elif kind == "sampling":
        count = max(16, round(len(x) * (1 - 0.9 * severity)))
        gcd = math.gcd(count, len(x))
        reduced = resample_poly(x, count // gcd, len(x) // gcd)
        x = resample_poly(reduced, len(x) // gcd, count // gcd)[:len(x)]
        info.update(effective_sample_rate_hz=sample_rate_hz * count / len(x),
                    description=f"Reduced capture rate · {sample_rate_hz * count / len(x):.0f} Hz")
    else:
        factor = 1 + 4 * severity
        x *= factor
        info.update(gain=factor, description=f"Amplitude gain · {factor:.1f}×")
    return x.astype(np.float32), info


SCENARIOS = [{"id": "clean", "kind": "clean", "severity": 0.0}]
SCENARIOS += [{"id": f"{kind}-{int(level * 100)}", "kind": kind, "severity": level}
              for kind in KINDS[1:] for level in (0.25, 0.5, 0.75, 1.0)]


def release_gate(clean, mild_noise, p95_ms, parity_error=None):
    """Fixed illustrative engineering criteria; not a clinical release standard."""
    auc = clean.get("auroc")
    noisy_auc = mild_noise.get("auroc")
    drop = auc - noisy_auc if auc is not None and noisy_auc is not None else None
    rules = [
        ("clean_auc", "Clean AUROC", auc, ">=", 0.75),
        ("balanced_accuracy", "Balanced accuracy", clean.get("balanced_accuracy"), ">=", 0.65),
        ("noise_regression", "AUROC loss at 23.75 dB", drop, "<=", 0.10),
        ("latency", "Warm inference p95 (ms)", p95_ms, "<=", 25.0),
    ]
    if parity_error is not None:
        rules.append(("export_parity", "Maximum score error after export", parity_error, "<=", 1e-4))
    checks = []
    for key, name, value, op, limit in rules:
        valid = value is not None and math.isfinite(float(value))
        passed = valid and (value >= limit if op == ">=" else value <= limit)
        checks.append({"id": key, "name": name, "value": value, "operator": op,
                       "limit": limit, "passed": bool(passed)})
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "scope": "Demonstration release gate on this weak-label cohort; not approval for clinical use."}
