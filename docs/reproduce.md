# Reproduction guide

## Public software reproduction

The README installation, test and smoke commands require Python3.12 and no
biomedical records. The smoke generates all fixtures anew from RNG seed20261010,
uses two-epoch training as a software check, and produces ignored local outputs.
No score target is asserted: smoke performance is not a clinical endpoint.

`requirements-lab-lock.txt` is the exact observed macOS environment snapshot.
`requirements-lock.txt` is the original core/data snapshot. All listed versions
are pinned; they are not cryptographically hash-locked dependency resolutions.
GPU runtimes are unnecessary. CPU Linux wheels and the GitHub runner can differ
from the local arm64 environment; check CI before claiming remote reproducibility.

## Reacquire the original supervised cohort

The original record identifiers are intentionally withheld from this release.
The deterministic acquisition rule reconstructs its exclusion cohort from the
public publisher source rather than shipping identifying manifests. Only
invoke download commands for data you are permitted to use under source terms.

```sh
python scripts/download_mimic_pilot.py --output-dir data/raw/original-pilot \
  --per-class 32 --workers 4 --metadata-max-mib 8
python scripts/download_mimic_pilot.py --output-dir data/raw/ecg-lab \
  --per-class 512 --workers 8 --metadata-max-mib 16
python scripts/build_lab_dataset.py --raw data/raw/ecg-lab \
  --exclude data/raw/original-pilot/record_list.csv --output data/prepared/ecg-lab
corvus-repeatability --manifest data/prepared/ecg-lab/manifest.csv \
  --work work/repeatability-reacquired --aggregate work/repeatability-aggregate.json
```

The downloads are bounded and store local source receipts. The script uses
metadata order and one record per patient; metadata changes can change the cohort.
The observed manifest SHA256 is
`80660edb33a3f96a980dc7f5c737be465a35b9b347e31535834df453a725c10a`.
Preparation excludes invalid/nonfinite waveforms and outputs its local audit.
Check the aggregate counts and hash before labeling the experiment an exact
cohort reproduction. The October10 rerun reused the locally held prepared cohort;
it did not download the corpus again. Reacquisition from today's remote source
is implemented but was not executed again during this release.

Keep `data/`, `runs/`, `work/` and `artifacts/` private. They contain raw records,
machine reports, identifiers, predictions, checkpoints or individual provenance.
The repeatability command publishes only the requested aggregate file. Review
the exact aggregate report before copying it into public `evidence/`.

## Rerun invariants

The frozen protocol records seeds, selection rules, hyperparameters, architecture,
class/group counts, manifest SHA256, source SHA256 and environment before fitting.
All declared models train first; the runner then checks code/data/protocol hashes
before scoring test. It refuses existing output paths. Checkpoints/thresholds
are selected on validation only; test scores never select a seed or alter rules.

A fresh directory does not create a fresh statistical holdout. This cohort was
already observed historically. Any hypothesis shaped by its results requires a
new independent test partition, ideally externally acquired and held by a
separate evaluator. Do not optimize against the reported test or stress metrics
and then describe a successor's score on it as new generalization evidence.

## Optional local lab

The original `corvus-lab train/build/serve` modules are retained as source and
exercised with synthetic test fixtures. `build` deliberately constructs a
**private** serving bundle including examples from its local test records and
needs an independently source-trained astronomy checkpoint. Nothing in that
private bundle is included in this repository. The bundled static UI reflects
the original ECG lab experiment; it is not a public patient-data deployment.

Gates are demonstrative engineering thresholds, not regulatory or clinical
approval. Hashes verify consistency relative to a trusted local registry; they
do not prevent a person who controls that registry from changing approval.
No container or public API deployment was verified in the new release.
