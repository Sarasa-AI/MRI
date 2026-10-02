# Experiment Report — RSNA-DATA-002 (G1b Metadata Audit)

**Status:** COMPLETE  
**Date:** 2026-10-02  
**Execution plane:** Kaggle Notebook (CPU) attached to competition  
**Kernel:** https://www.kaggle.com/code/alikarimiansarasa/rsna-data-002-metadata-audit-g1b  
**Artifacts:** `outputs/RSNA-DATA-002/` (CSV/JSON summaries only; no DICOMs)

---

## Hypothesis

Kaggle-mounted competition CSVs yield authoritative study/series/label
statistics and a leakage-safe `StudyInstanceUID` GroupKFold map without any
DICOM decode or local competition-data download.

**Pass bar:** artifacts written; mount path resolved; schemas verified; zero
study overlap across folds; all gold studies assigned exactly once; Report
absent from test; baseline architecture unchanged.

---

## Method (constraints honored)

- Read **only**: `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`,
  `sample_submission.csv`
- Mount discovery: shallow candidate paths under `/kaggle/input` (no recursive
  DICOM tree walk)
- No DICOM open/decode; no training; no baseline architecture change
- Deterministic 5-fold `GroupKFold` on `StudyInstanceUID` (seed=42)

---

## VERIFIED

### Exact mount path

`/kaggle/input/competitions/rsna-knee-abnormality-detection`

(Not `/kaggle/input/rsna-knee-abnormality-detection` — that path does **not**
exist on this account’s runtime.)

### Dataset schema

| File | Rows | Columns |
|---|---:|---|
| `train.csv` | 4407 | `StudyInstanceUID`, `Report`, 12 labels (`float64`) |
| `train_series.csv` | 24371 | `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `test.csv` | 3 (example) | `StudyInstanceUID` only |
| `test_series.csv` | 15 (example) | same 5 cols as train_series |
| `sample_submission.csv` | 3 | `StudyInstanceUID` + 12 labels |

### Counts

| Metric | Value |
|---|---:|
| Train studies | **4407** |
| Train series | **24371** |
| Gold-labelled studies (all 12 non-null) | **58** |
| Partial-label rows | **0** (all-or-nothing) |
| Reports non-null (train) | **4407 / 4407** |
| Report in test | **No** |
| Series / study | min 3, median 5, mean 5.53, max 14 |

### Series descriptors (CSV)

| Anatomical_Plane | n_series | n_studies covered |
|---|---:|---:|
| Sagittal | 9864 | 4407 (all) |
| Coronal | 8609 | 4407 (all) |
| Axial | 5898 | 4407 (all) |

`Fluid_Sensitive` / `Fat_Suppression` appear coupled in train CSV (only
`0/0` and `1/1` combinations observed).

### Label counts (gold n=58)

| Label | Pos | Neg | Prev |
|---|---:|---:|---:|
| ACL | 24 | 34 | 0.414 |
| MCL | 9 | 49 | 0.155 |
| Medial Meniscus | 26 | 32 | 0.448 |
| Lateral Meniscus | 23 | 35 | 0.397 |
| Medial OA | 15 | 43 | 0.259 |
| Lateral OA | 11 | 47 | 0.190 |
| PF OA | 21 | 37 | 0.362 |
| Effusion | 35 | 23 | 0.603 |
| Synovitis | 27 | 31 | 0.466 |
| Baker's | 12 | 46 | 0.207 |
| Contusion | 19 | 39 | 0.328 |
| Fracture | 18 | 40 | 0.310 |

### Field usability (CSV metadata)

| Field | Present / usable |
|---|---|
| StudyInstanceUID | **Yes** (primary grouping key) |
| SeriesInstanceUID | **Yes** |
| SOPInstanceUID | **No** in CSVs (filename-only in DICOM tree; not audited) |
| Report | **Yes in train only**; absent from test |
| Sequence/series descriptors | **Yes**: `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression` |
| Orientation-related CSV metadata | **No** |
| Patient identifier | **No** (`PatientID` absent) |
| Laterality | **No** in CSVs |

### Fold feasibility (seed=42, n_splits=5)

- Zero `StudyInstanceUID` overlap between folds: **PASS**
- All 58 gold studies assigned exactly once: **PASS**
- Gold fold sizes: 13 / 11 / 11 / 13 / 10
- Zero-positive fold×label cells: **2**
  - fold 1 × `Lateral OA`
  - fold 4 × `Medial OA`
- Zero-negative fold×label cells: **0**

Fold map: `outputs/RSNA-DATA-002/study_folds.csv`  
Per-fold prevalence: `outputs/RSNA-DATA-002/fold_label_prevalence.csv`

---

## UNKNOWN (explicitly not measured)

- DICOM header information (allowlisted tags beyond CSV)
- DICOM orientation / ImageOrientationPatient
- Pixel dimensions / spacing
- Slice counts per series
- Actual series quality / corrupt files
- Usable image availability after decode
- True patient-level leakage (no PatientID in public CSVs)

---

## Train / test schema differences

- Shared study key: `StudyInstanceUID` only
- Train-only: `Report` + 12 labels
- Test example has no labels / no Report (scoring-time test ≈1300 studies replaces example)
- Series schemas identical train↔test

---

## GATE DECISION

### **PARTIALLY_READY**

**Why not BLOCKED:** mount, schemas, Report-at-train-only, StudyInstanceUID
fold leakage checks, and gold coverage all pass. Baseline can run.

**Why not READY_FOR_BASELINE:** with n_gold=58 and plain GroupKFold, two rare
OA labels have a fold with zero positives (AUC undefined / unstable for that
fold×label). This is a **CV measurement caveat**, not a schema blocker.

**Baseline action:** do **not** change RSNA-BASELINE-001 architecture from this
audit. Optional later (separate gate): StratifiedGroupKFold / fewer folds for
rare labels — only after documenting as its own experiment.

---

## Artifacts produced

| Artifact | Path |
|---|---|
| runtime_paths.json | `outputs/RSNA-DATA-002/runtime_paths.json` |
| study_summary.json | `outputs/RSNA-DATA-002/study_summary.json` |
| series_summary.json | `outputs/RSNA-DATA-002/series_summary.json` |
| label_coverage.csv | `outputs/RSNA-DATA-002/label_coverage.csv` |
| study_folds.csv | `outputs/RSNA-DATA-002/study_folds.csv` |
| fold_label_prevalence.csv | `outputs/RSNA-DATA-002/fold_label_prevalence.csv` |
| GATE_DECISION.json | `outputs/RSNA-DATA-002/GATE_DECISION.json` |

Source template: `notebooks/kaggle/01_metadata_audit.py`  
Push package: `notebooks/kaggle/kernels/rsna-data-002-metadata-audit/`

---

## Next

1. Proceed to **RSNA-BASELINE-001** Kaggle GPU run using this mount path +
   `study_folds.csv` (expect per-fold macro-AUC noise on rare labels).
2. Keep gold-only (D-010). Do not invent PatientID/Laterality from CSV.
3. DICOM header / slice / quality audit remains a **separate** gated probe if needed.
