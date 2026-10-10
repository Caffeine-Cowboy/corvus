# Next experiments: stronger evidence with independent holdouts

## External validity and label quality

Before changing the CNN, reserve an unseen cohort across a separate acquisition
window or dataset. Predeclare unit of analysis, positive endpoint, exclusion
rules, metadata stratification, main metric and minimum effect of interest.
Keep all recordings of a patient in one partition. Power uncertainty estimates
using expected class counts, rather than treating thousands of correlated windows
as independent participants. Evaluate an independently implemented simple
feature/logistic baseline and the fixed CNN with paired patient bootstraps.

Obtain authorized adjudication or a dataset with reviewed beat labels. Measure
agreement with machine-report labels and analyze ambiguous/comorbid cases. Never
use strict-normal-negative separation alone as evidence of real clinical
discrimination. Keep adjudication metadata out of the public repository.

## Training interventions

Freeze a train/validation-only factorial comparison: no augmentation versus
noise/lead-dropout augmentation; supervised learning versus masked-window or
contrastive pretraining; full fine tuning versus frozen encoder linear probes.
Use equal optimization/compute budgets and at least five declared seeds after
an inexpensive development pilot. Prespecify one main test comparison and report
all seeds/negative results. Statistical selection happens on validation;
correlated corruption conditions are exploratory, not independent confirmations.

Temperature scaling and operating thresholds must fit a distinct calibration
partition or cross-fitting scheme. Report Brier score, log loss and reliability
with intervals under prevalence shifts. The present sigmoid scores are not
validated clinical probabilities, and this repository does not claim completed
calibration or self-supervised pretraining.

## Multimodal reasoning and language-model post-training

Corvus can serve as a waveform engineering component. A separate benchmark must
define multimodal alignment, missingness, contradictory sensor evidence,
abstention, and intervention-based consistency. Synthetic physiology offers a
public development path, but simulation fidelity and label shortcuts must be
audited; success in simulation needs external empirical validation.

For language-model post-training, start with a small explicitly permitted model
and training/validation cases separate by latent scenario family. Compare a
frozen baseline with supervised fine tuning and preference optimization using
identical held-out evaluation and reported compute. Keep answer keys and test
families unavailable to training. These are future experiments, not results
already established by ECG CNN repeatability. The companion PhysioBench package
should own the actual completed training/post-training experiments and their
scope claims.

## Publication bar

Publish the frozen protocol, exact environment, executable evaluation, aggregate
results with uncertainty, baselines/ablations, negative findings and data/model
limitations. Strong engineering execution supports a portfolio; scientific
novelty and doctoral-level contribution require defensible original questions,
adequate experiments and independent critical review.
