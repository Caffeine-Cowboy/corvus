import importlib.util
from pathlib import Path


def load_audit():
    source = Path(__file__).parents[1] / "scripts/audit_release.py"
    spec = importlib.util.spec_from_file_location("release_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_detects_sensitive_data_and_allows_aggregate(tmp_path):
    audit = load_audit().audit
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/aggregate.json").write_text('{"auroc": 0.95, "n": 193}')
    assert not audit(tmp_path)
    (tmp_path / "evidence/raw.json").write_text('{"subject_id": "fake-identifier"}')
    (tmp_path / "signal.npy").write_bytes(b"private fixture")
    assert len(audit(tmp_path)) >= 2
    (tmp_path / "data").mkdir()
    (tmp_path / "data/private.npy").write_bytes(b"ignored private fixture")
    assert not any("private.npy" in x for x in audit(tmp_path))
