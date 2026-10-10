# Research note: conditional repeatability of a weak-label ECG baseline

## Question and study status

Does a fixed small supervised ECG CNN give similar results under independent
training seeds, while preserving validation-only selection and the original
patient-disjoint split? This is an engineering repeatability study, not a claim
of methodological novelty, clinical validation, or doctoral equivalence.

The original one-seed holdout had already been observed. On October 10, 2026,
the rerun protocol was written before the three new training runs. No architecture,
hyperparameter, gate, score direction or threshold rule was revised after
examining the new test results. It would be incorrect to call the test partition
a newly untouched holdout or this retrospective rerun a prospective preregistration.

## Cohort and endpoint

The original bounded acquisition chose the first qualifying record for each
patient in metadata order, until 512 records per class were selected. The label
parser uses affirmative PVC/ventricular ectopy machine-report language for
positives. It rejects negated, historical, uncertain and ambiguous PVC mentions.
Negatives require a sinus/normal-sinus-rhythm line and an explicit normal-ECG line,
with all report lines belonging to the strict normal vocabulary. Labels concern
reports and are not cardiologist-adjudicated diagnoses or beat localization.

All 64 original transfer-pilot patients were excluded. One nonfinite waveform
was excluded during preparation, leaving 959 unique patients, one record each.
Within each class, patients are sorted by record identifier, permuted with split
seed 173, and assigned using floor(0.6 n)/floor(0.8 n) boundaries. Final counts are
574 train (289 positive), 192 validation (97 positive), and 193 test (97 positive).
Records cannot cross splits through patient or parent identifiers; identical
numerical waveforms stored under different file names or float dtypes are checked
for cross-split overlap. No record identifiers are published.

Sampling is neither representative of clinical prevalence nor random from the
full population. Strict normal negatives and potentially comorbid positives can
make class separation easier than a clinically realistic differential diagnosis.
Performance may depend on report correlates, device/site artifacts or coexisting
abnormalities. This small balanced sample does not support demographic subgroup,
hospital-shift, ambulatory or prospective deployment claims.

## Model and fitting

Input is a complete ten-second 500 Hz lead-II signal. The per-window transform
subtracts the median, polyphase resamples to 1,024 points with anti-aliasing,
subtracts the new median, divides by 1.4826 times median absolute deviation
(standard deviation fallback), and clips to ±8. No population moments are fitted.
The dimensionless transform is not a claim that astronomy and ECG share a
physical sample rate. Amplitude normalization also discards potentially useful
clinical amplitude information.

The CNN has three stride-2 convolution blocks: 1→16/kernel15, 16→32/kernel9,
32→64/kernel5, each with GroupNorm(4 groups) and GELU. Concatenated temporal mean
and max pooling gives 128 features followed by a single-logit linear head.
GroupNorm requires no test-time population-statistic adaptation.

For each seed 17, 42 and 173: train 30 epochs using batch32, binary cross entropy
with logits, AdamW learning rate0.001/weight decay0.01, and gradient norm clipping5.
Choose the checkpoint with lowest validation log loss (earliest epoch on exact
ties). Choose the threshold maximizing validation balanced accuracy, preferring
specificity and then a higher threshold on ties. Decisions use strict
`score > threshold`. All seeds and selected thresholds are frozen before test
scores are computed. There is no seed selection, test-label fitting, score
inversion, ensemble optimization or architecture search.

## New results and uncertainty

Seed17: best epoch5, AUROC0.953823. Seed42: best epoch8, AUROC0.957904.
Seed173: best epoch8, AUROC0.966280. The seed42 checkpoint selection and threshold
match the recorded historical run; the full rerun is independently executed.

Across three runs: mean AUROC0.959336, sample SD0.006351; mean average precision
0.966829, SD0.004586; mean balanced accuracy0.903422, SD0.005968; mean sensitivity
0.876289, SD0.010309; mean specificity0.930556, SD0.006014. These are descriptive
training-seed summaries, not a reliable estimate of the complete optimization
variance distribution. The seed SD is not divided by √3 and presented as a
precise inferential confidence interval.

We used 2,000 unstratified patient-cluster bootstrap samples with RNG seed20261010.
All models are evaluated on the same resampled patients, preserving pairing.
The percentile 95% interval for mean AUROC across the three frozen models is
0.932866–0.982007. Seed-specific fixed-threshold metric intervals are also
recorded. Bootstrap samples lacking either class are skipped and counted.
The interval describes patient sampling uncertainty conditional on these fixed
models and this selected cohort. It excludes new-population shift, label error,
the full training-seed distribution and uncertainty from study design choices.

The rerun took 27.12 seconds in the recorded local CPU environment with two
threads. This is elapsed training plus evaluation for a small cohort/model,
not an inference SLA or a general training-efficiency claim.

## Historical evidence, retained without promotion

`historical-lab-aggregate.json` preserves the original one-seed lab: supervised
CNN AUROC0.957904, maximum ONNX score difference4.17e-7, and the saved robustness
and warm-runtime measurements. It is not an independently rerun ONNX/serving
benchmark. Its engineered corruption suite shares the same test patients across
conditions and cannot be counted as 21 independent test cohorts. Median/MAD
normalization explains much of gain invariance; synthetic corruptions do not
establish real artifact, device or hospital robustness.

`historical-transfer-pilot-aggregate.json` preserves the separate strict
astronomy-to-ECG pilot. Source-trained CNN AUROC0.447266 versus random-network
0.764648 does not demonstrate useful transfer. Neither score direction nor
operating point was retuned on ECG outcomes. Controls are kept despite
unfavorable results. The later supervised baseline is a separate experiment
and cannot rescue the original zero-shot hypothesis.

## Engineering verification and research limits

Tests verify silent invalid-label/broadcasting rejection, group/content leakage,
test-label independence of supervised fitting, immutability of completed reruns,
deterministic stress generation, release-gate failure, synthetic API inference,
and model-integrity rejection. Reproduction without data exercises both transfer
and supervised paths using deliberately artificial fixtures. CI requires no
patient data. Code/protocol/data hashes bind the local rerun; hashes of raw
individual records remain private.

This is a complete, limited research engineering artifact. External holdouts,
adjudicated labels, properly powered comparisons and independently implemented
baselines remain necessary for stronger scientific claims. No language-model
post-training, medical decision support, clinical calibration, multimodal
health reasoning or novel general representation is established here.
