# Corvus: reproducible ECG training and evaluation evidence

A source-only research engineering package for patient-disjoint ECG modeling,
validation-only selection, negative-transfer controls, deterministic stress tests,
and local inference gates. It publishes executable methods and aggregate evidence.
No patient recordings, study/subject identifiers, individual predictions, trained
clinical models, confidential university assets, or patient-derived demo examples
are included.

## Measured evidence

The supervised baseline was rerun on October 10, 2026 with three fixed seeds
(17, 42, 173), 30 epochs each, and the existing 574/192/193 train/validation/test
patient split. Mean held-out AUROC was **0.9593**, training-seed sample SD
**0.0064**, range **0.9538–0.9663**. Mean balanced accuracy was **0.9034**.
The patient-bootstrap 95% interval for the mean AUROC of these frozen models was
**0.9329–0.9820** (2,000 paired resamples).

This is **repeatability on a previously observed holdout**. The original test
results were already known when the rerun protocol was frozen. The cohort is a
balanced convenience sample of ten-second lead-II recordings with machine-report
weak labels: affirmative PVC evidence versus a strict normal report. These
measurements do not establish clinical accuracy or external generalization.
See [exact methods and limitations](docs/research-note.md) and
[machine-readable rerun evidence](evidence/repeatability-2026-10-10.json).

The original astronomy-to-ECG transfer pilot is retained as negative evidence:
trained source CNN AUROC **0.4473**, versus random-network **0.7646** and
crest-factor **0.4297** on 64 patient recordings. No successful zero-shot transfer
is claimed. In the separate historical 193-patient lab, the astronomy model
achieved AUROC **0.3661** and failed its predefined serving gate. Scores were
never inverted using target labels. Historical values are saved evidence,
explicitly distinguished from the new reruns.

## Reproduce the software without biomedical data

Use Python 3.12. The exact environment snapshot used for the local results is
`requirements-lab-lock.txt`; it pins dependencies but is not a hash-locked,
universal platform lock. The default CPU path uses two threads and deterministic
PyTorch operations. Numerical equality across hardware/toolchains is not assumed.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lab-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/audit_release.py
python scripts/reproduce_smoke.py --work work/synthetic-smoke
```

The smoke command creates artificial waveform fixtures and exercises source-only
training/transfer scoring and supervised repeated training. It requires a fresh
output directory. Synthetic performance is a software check, never evidence
of biomedical validity. Every API test uses artificial waveforms.

On Linux CPU-only systems, install the CPU PyTorch wheel before the environment
snapshot to avoid acquiring the GPU runtime:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

## Reproduce the empirical cohort locally

Data are acquired directly from the publisher under its own terms. The
source-only release keeps data and outputs in ignored directories. The precise
acquisition, label and split rules are in [the reproduction guide](docs/reproduce.md).

For an already prepared local ECG manifest:

```sh
corvus-repeatability --manifest data/prepared/ecg-lab/manifest.csv \
  --work work/repeatability-rerun --aggregate work/repeatability-aggregate.json
```

The runner freezes code and manifest hashes before training, fits every declared
seed before opening test scores, selects checkpoints/thresholds on validation,
checks that code/data did not change, and exports aggregate metrics only. Private
training provenance and weights stay in the ignored work directory. It refuses
existing run/report paths. A later rerun remains a rerun on the same observed
cohort; changes require an independently reserved holdout for fresh evidence.

## Contents and verification

- `src/corvus`: preparation, leakage checks, transforms, model, training,
  evaluation, bootstrap uncertainty, stress tests, and optional local API.
- `scripts`: bounded source acquisition, preparation, synthetic smoke and
  release audit. Download commands require explicit invocation.
- `tests`: statistical input contracts, target-label independence, group/content
  leakage rejection, deterministic corruptions, API gates, model integrity and
  source-release hygiene.
- `evidence`: aggregate historical reports, frozen rerun protocol and new rerun.
- `docs`: methods, reproduction, provenance, model/data limitations and roadmap.

Local verification passed **70 tests** plus synthetic smoke. GitHub Actions is
configured for the same source-only checks; remote status must be read from the
repository. Original ONNX/latency results are historical saved measurements;
this release does not imply a new timing benchmark or container verification.

The optional `corvus-lab` API expects a locally built private bundle. That builder
creates patient-level examples from local data and must stay in ignored
`artifacts/`; **do not upload or publicly serve such a bundle**. This distribution
provides source/API tests, not a preloaded clinical demo. Its registry is trusted
local configuration, not a signed clinical approval mechanism.

## Scope and next research

Corvus is an ECG engineering study, not a multimodal foundation model or a
post-training result. [The research roadmap](docs/next-experiments.md) specifies
how to evaluate external generalization, artifact failures, calibration, and
training interventions with independent holdouts and adequate compute.

Code is MIT licensed. MIMIC-IV-ECG remains under its publisher's ODbL 1.0 terms;
no dataset is redistributed. See [data/source attribution](docs/provenance.md).
