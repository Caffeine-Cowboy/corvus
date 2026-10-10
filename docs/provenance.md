# Provenance, attribution and release boundary

## Source provenance

The author confirmed that Project Corvus is independent work and application
details are shareable. This release copies selected Corvus code, tests and
acquisition/preparation scripts from the author's local project reference.
The synced reference remained read-only. It does not copy confidential university
research, protected datasets or any university application assets.

Release additions: an immutable multi-seed repeatability runner; synthetic
reproduction; statistical/schema hardening against invalid labels, broadcasting,
misaligned cluster groups and malformed rows; tests for test-outcome independence;
and public-artifact checks, documentation, license and source-only CI.

Historical aggregate files retain source-report SHA256s and fixed metrics but
strip record inventories, individual identifiers, local paths and raw examples.
The October10 aggregate is a newly executed three-seed repeatability study.
The frozen source hash applies to the exact `src/corvus/*.py` source collection
used in that rerun, with relative paths and bytes hashed in lexicographic order.

## Dataset attribution

The empirical results use MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched
Subset v1.0 (Gow et al., 2023), published by PhysioNet:
[dataset and publisher terms](https://physionet.org/content/mimic-iv-ecg/1.0/),
[DOI](https://doi.org/10.13026/4nqg-sb35). The publisher describes ten-second,
12-lead, 500 Hz diagnostic ECGs; this model uses leadII only. Dataset access is
open subject to the Open Data Commons Open Database License1.0. Associated
clinical note modules have separate access conditions and are not used here.

The original source-domain negative-transfer study used CHIME/FRB Catalog1
astronomical time series. Data adapters and acquisition code remain for method
inspection; no astronomical source data are bundled. Synthetic fixtures are
explicitly invented signals and cannot establish transfer between these domains.

Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C.,
Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Chaudhari, A., Herbst, E.,
Moukheiber, D., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECG:
Diagnostic Electrocardiogram Matched Subset (version1.0). PhysioNet.
doi:10.13026/4nqg-sb35.

Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C.,
Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for
biomedical research. Nature Health. doi:10.1038/s44360-026-00096-z.

## Distribution policy

Only original source/documents and aggregate evidence are released. No raw or
prepared waveform, subject/study identifier, individual outcome/prediction,
patient-derived demo, trained clinical checkpoint, metadata/report subset,
secret or local absolute path is distributed. Ignored paths prevent routine
tracking; `scripts/audit_release.py` adds mechanical checks. These checks cannot
replace inspection of the exact staged publication files.

MIT covers original software and documentation. It does not relicense datasets
or private derived data. Reacquired data and artifacts keep publisher terms and
attribution obligations. Publication of the source package does not authorize
redistributing a local patient-data serving bundle.
