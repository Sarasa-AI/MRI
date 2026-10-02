# Experiment Report — RSNA-BASELINE-001-PILOT

**Status:** COMPLETE  
**Date:** 2026-10-02  
**Execution plane:** Kaggle Notebook GPU  
**Kernel:** https://www.kaggle.com/code/alikarimiansarasa/rsna-baseline-001-pilot  
**Artifacts:** `outputs/RSNA-BASELINE-001-PILOT/` (metrics/JSON/CSV/checkpoint only; no DICOMs)

---

## Hypothesis

The existing EfficientNet-B0 multi-plane mid-slice pipeline can load real competition
DICOMs on Kaggle GPU, decode selected series, maintain StudyInstanceUID leakage safety,
train one gold-only fold for ≤5 epochs with finite loss, and emit a valid submission
without critical orientation, ordering, leakage, or silent-zero failures.

**Pass bar:** GPU confirmed; DICOM decode reliable; non-degenerate mid-plane tensors;
fold-0 leakage green; training completes; submission validates; no critical
orientation/order blocker → proceed toward full 5-fold only if OOF path is trustworthy.

---

## A — Executive Summary

### **READY_WITH_CAVEATS**

Automated notebook gate emitted `READY_FOR_FULL_BASELINE` based on artifact presence.
Human review **overrides** to `READY_WITH_CAVEATS` because:

- Real DICOM decode / ordering / GPU / leakage / training / submission **PASS**
- Pilot **OOF predictions are entirely NaN** (0/58 studies scored)
- Val macro-AUC became **NaN after epoch 0** (epoch 0 was 0.408)

The imaging I/O + GPU train path is validated. The OOF/metric aggregation path is **not**
yet trustworthy enough to mint an official baseline floor without a follow-up fix/review.

Official `configs/experiment/rsna_baseline_001.yaml` was **not** mutated.

---

## B — Environment

| Item | Value |
|---|---|
| Kaggle competition mount | `/kaggle/input/competitions/rsna-knee-abnormality-detection` |
| Code dataset mount | `/kaggle/input/datasets/alikarimiansarasa/rsna-knee-control` |
| GPU | Tesla T4 (CUDA available; device `cuda:0`) |
| CUDA | 12.8 (`torch` 2.10.0+cu128) |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12.13 |
| Config | `rsna_baseline_001.yaml` via Dataset |
| Pilot overrides | `experiment_id=RSNA-BASELINE-001-PILOT`, `pilot_folds=1`, `epochs=5` |
| Seed | 42 |
| Git commit recorded in Dataset | `2092ef19…+pilot` (kernel env git_commit null) |

---

## C — DICOM Findings

Gold mid-slice audit (58 studies × 3 planes = **174** attempts):

| Metric | Value |
|---|---:|
| Success rate | **1.000** |
| Degenerate (zero/near-zero) rate | **0.000** |
| Failure counts | `{}` |
| Mid-slice audit wall time | ~70 s |

Header-rich sample (n=30 selected series; **sample stats only**):

| | rows | cols | aspect | slices |
|---|---:|---:|---:|---:|
| min | 256 | 256 | 1.00 | 18 |
| p25 | 336 | 336 | 1.00 | 24.5 |
| median | 512 | 512 | 1.00 | 30 |
| p75 | 640 | 640 | 1.00 | 33.25 |
| p95 | 739 | 739 | 1.04 | 45.5 |
| max | 800 | 800 | 1.07 | 320 |

### Ordering (Risk A)

- Method in baseline: **`InstanceNumber` sort** (`load_mid_slice`)
- Audit: InstanceNumber vs IPP projection on slice normal
- Comparable series: **30 / 30**
- Disagreement rate: **0.0**
- Mean \|Spearman\|: **1.0**
- Direction may be reversed (Spearman −1) while remaining monotonic — **acceptable** for mid-slice

**Verdict:** No material InstanceNumber vs IPP disagreement on the sample. Not a blocker.

### Orientation (Risk B)

- `ImageOrientationPatient` present on sampled series
- Example IOP shows consistent within-series orientation; row×col normal well-defined
- No automated evidence of unsafe transposition for the current mid-slice RGB stacking
- Full anatomic L/R consistency vs compartment labels remains a residual uncertainty (CSV laterality absent)

**Verdict:** No critical orientation blocker from sampled headers. Residual uncertainty → caveat, not blocker.

### Silent failures (Risk C)

Independent audit found **0** decode failures and **0** degenerate tensors on gold selected series.
Baseline `StudyMultiPlaneDataset` still silently `continue`s on failure — risk remains latent but
was **not** triggered on this gold set.

---

## D — Pipeline Findings

| Stage | Result |
|---|---|
| CSV → path discovery | PASS (competitions/ mount; 4407 train_series entries; sample `.dcm` found) |
| Series selection | PASS (existing Fluid_Sensitive → Fat_Suppression policy) |
| DICOM decode mid-slice | PASS (174/174) |
| Preprocess 224² | PASS (cached tensors finite, non-degenerate) |
| Model EfficientNet-B0 | PASS (pretrained; checkpoint saved) |
| Val metric / OOF write | **FAIL / caveat** (OOF all-NaN; val AUC NaN after ep0) |
| Test submission | PASS (validator OK) |

---

## E — GPU Findings

| Item | Value |
|---|---|
| GPU used? | **Yes** (`cuda`) |
| Name | Tesla T4 |
| Peak CUDA alloc | ~420 MB (train) |
| Peak RSS | ~2.2 GB |
| Train wall | ~42 s (5 epochs, fold 0) |
| Approx epoch | ~8.4 s |
| Approx samples/sec | ~5.5 |
| DICOM audit wall | ~70 s (dominated first-pass decode) |

---

## F — Training Findings

| Item | Value |
|---|---|
| Splitter | StratifiedGroupKFold on ACL (existing `imaging_loop`) |
| Fold | 0 only (`pilot_folds=1`) |
| n_train / n_val | 46 / 12 |
| Epochs | 5 (override; official YAML still 25) |
| Train loss (ep0→ep4) | 0.694 → 0.688 → 0.673 → 0.665 → 0.674 (finite) |
| Val macro AUC ep0 | **0.408** |
| Val macro AUC ep1–4 | **NaN** |
| Best checkpoint | epoch 0, stored auc=0.408, **no NaN weights** |
| Pilot OOF macro AUC | **NaN / undefined** (no scored OOF rows) |
| Per-label OOF AUC | all undefined |
| Leakage | PASS (`train ∩ val = ∅`) |

Pilot AUC is **not** an official floor.

### Split vs RSNA-DATA-002

Baseline uses StratifiedGroupKFold(ACL); DATA-002 used plain GroupKFold.
DATA-002 `study_folds.csv` was **not** found on the Dataset mount during this run
(`data002_fold_map_found=false`) — comparison skipped. Difference remains a documented caveat.

---

## G — Data Quality

| Item | Value |
|---|---:|
| Missing planes (gold selected) | 0 |
| Decode failures | 0 |
| Degenerate tensors | 0 |
| Unusual dims | rows/cols 256–800 (sample); one series with 320 slices (sample max) |

---

## H — Submission Validation

| Check | Result |
|---|---|
| `rsna_knee.submission.validate` | **PASS** |
| Rows | 3 (example test) |
| Probabilities in [0,1] | Yes |
| Duplicates / missing | None |

Not submitted to LB (gate does not require submission).

---

## I — Problems / Risks

| Severity | Issue |
|---|---|
| **HIGH** | OOF CSV all-NaN despite training + valid test submission |
| **HIGH** | Val macro-AUC NaN for epochs 1–4 after a finite epoch-0 value |
| MEDIUM | Baseline still silent on decode failure (not triggered here) |
| MEDIUM | InstanceNumber-only ordering (audited OK on sample; keep monitoring) |
| MEDIUM | Case-1 local fix: `_safe_auc` masks NaN `y_score` for `pilot_folds` |
| LOW | Pilot AUC not a floor; DATA-002 fold map missing on mount |

---

## FINAL GATE DECISION

### **READY_WITH_CAVEATS**

**Meaning:** Real DICOM + GPU training + submission path works. Before burning a full
5-fold official baseline, diagnose the OOF NaN / post-epoch-0 val AUC NaN so the
eventual OOF macro-AUC is trustworthy.

**Do not** start LABELS-002 / INPUT-001 / full 5-fold until that review.

---

## Checklist

```text
[x] Kaggle mount verified
[x] Competition DICOMs found
[x] CUDA/GPU verified
[x] Gold-only fold verified
[x] StudyInstanceUID leakage check passed
[x] DICOM decode audit completed
[x] InstanceNumber vs IPP ordering audited
[x] ImageOrientationPatient audited
[x] Non-degenerate tensors verified
[x] Training completed
[x] Finite loss verified
[~] Finite gradients verified (inferred via finite loss + non-NaN checkpoint weights)
[x] Checkpoint created
[~] OOF predictions created (file exists, values all NaN)
[x] Submission created
[x] Submission validator passed
[x] Gate decision written
```

```text
GATE_DECISION=READY_WITH_CAVEATS
```

## Next action

**STOP — await review.**
