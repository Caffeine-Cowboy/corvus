"""Scientific cohort and source-window rules must not silently broaden."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("lines, expected", [
    (["SINUS RHYTHM", "NORMAL ECG"], 0),
    (["Normal sinus rhythm", "Summary: Normal ECG"], 0),
    (["SINUS RHYTHM"], None),
    (["NORMAL ECG"], None),
    (["SINUS RHYTHM", "NORMAL ECG", "ST ELEVATION"], None),
    (["PREMATURE VENTRICULAR COMPLEXES"], 1),
    (["Ventricular ectopy"], 1),
    (["PVC(s)"], 1),
    (["NO PREMATURE VENTRICULAR COMPLEXES"], None),
    (["POSSIBLE PVCs"], None),
    (["PVCs or aberrant ventricular conduction"], None),
    (["History of PVCs"], None),
    (["PVCs", "Prior PVCs have disappeared"], None),
    (["ATRIAL FIBRILLATION"], None),
    (["ATRIAL FIBRILLATION", "PREMATURE VENTRICULAR CONTRACTION"], 1),
])
def test_machine_report_weak_label_policy(lines, expected):
    module = script("download_mimic_pilot")
    row = {f"report_{i}": text for i, text in enumerate(lines)}
    assert module.classify(row)[0] == expected


def test_short_source_window_guard_preserves_known_peak():
    module = script("prepare_chime_pilot")
    signal = np.zeros(96)
    signal[42] = 1
    proposal = module.propose_windows(signal)
    assert proposal["positive"] == [24, 56]
    assert proposal["background"] == [64, 96]
    assert proposal["guard_samples"] == 8
    assert proposal["center_shift_samples"] == -2
    with pytest.raises(ValueError):
        module.propose_windows(np.arange(40))
