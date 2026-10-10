#!/usr/bin/env python3
"""Fail if the public source tree contains prohibited data or obvious secrets.

This is an additional mechanical check, not a guarantee of anonymization. Inspect
the exact staged files before release. Private work is ignored and excluded here.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "runs", "artifacts", "work", "build", "dist"}
PROHIBITED = {".npy", ".npz", ".pt", ".onnx", ".dat", ".hea", ".h5", ".hdf5", ".parquet", ".csv", ".pem", ".key", ".zip"}
PATTERNS = [re.compile(r"(?:mimic-|patient-|study-)[0-9]{7,}"),
            re.compile(r"\bp[0-9]{8}\b|\bs[0-9]{8}\b"),
            re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
            re.compile(r"sk-[A-Za-z0-9_-]{40,}"),
            re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
INDIVIDUAL_EVIDENCE_KEYS = ('"record_id":', '"group_id":', '"study_id":', '"subject_id":',
                            '"test_inventory":', '"target_waveforms":', '"training_waveforms":',
                            '"validation_waveforms":', '"signal_sha256":')


def audit(root=ROOT):
    findings = []
    for p in sorted(root.rglob("*")):
        relative = p.relative_to(root)
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if p.is_symlink():
            findings.append(f"{relative}: symlink prohibited")
        elif p.is_file():
            if p.suffix.lower() in PROHIBITED or p.name.startswith(".env"):
                findings.append(f"{relative}: data/credential extension prohibited")
            try:
                value = p.read_text()
            except UnicodeError:
                findings.append(f"{relative}: unreviewed binary")
                continue
            if any(pattern.search(value) for pattern in PATTERNS):
                findings.append(f"{relative}: record-identifier/secret pattern")
            if relative.parts[0] == "evidence" and any(k in value for k in INDIVIDUAL_EVIDENCE_KEYS):
                findings.append(f"{relative}: individual-level provenance in public evidence")
            if "/" + "Users/" in value or "/" + "home/" in value:
                findings.append(f"{relative}: local filesystem path")
    return findings


if __name__ == "__main__":
    findings = audit()
    for finding in findings:
        print(finding)
    print(f"Release source audit: {len(findings)} finding(s)")
    sys.exit(bool(findings))
