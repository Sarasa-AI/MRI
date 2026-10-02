# Experiment Report — RSNA-BASELINE-001

**Status:** PIPELINE READY — metrics require Kaggle GPU execution  
**Date drafted:** 2026-10-02  
**Execution plane:** Kaggle only (Mac must not download competition DICOMs)

---

## 1. What is the true baseline?

| Field | Value |
|---|---|
| Experiment ID | `RSNA-BASELINE-001` |
| Model | EfficientNet-B0 (torchvision), ImageNet init, Dropout 0.2 → 12-logit head |
| Representation | Multi-plane mid-slice 2.5D: Sagittal + Coronal + Axial → `(3, 224, 224)` |
| Series policy | Prefer `Fluid_Sensitive=1`, then `Fat_Suppression=1`, one series per plane |
| Labels | Gold rows only (`all 12 targets non-null`) — ~58 studies |
| CV | 5-fold `GroupKFold` by `StudyInstanceUID`, seed 42 |
| Loss / Opt | BCEWithLogits + AdamW 1e-4 / wd 1e-4 / cosine / AMP(CUDA) / 25 epochs / ES patience 8 |
| Submission | Mean of fold-checkpoint sigmoid probabilities |

### Why this design (not more complex)

- **2.5D multi-plane mid-slice** is compact on Kaggle GPU, maps cleanly onto ImageNet RGB weights, and gives each plane a dedicated channel (ACL-friendly sagittal, meniscus/OA-friendly coronal, effusion-friendly axial).
- **EfficientNet-B0 @ 224** is strong enough to be a real floor, small enough for 5-fold on a sparse gold set without burning the 9h budget.
- **Gold-only** is intentional: user forbade pseudo-labeling / LLM labels for this run. The number this produces is the *honest imaging floor under expert labels*, not a LB-chasing score.

### Exact configuration
See `configs/experiment/rsna_baseline_001.yaml` (snapshotted to
`config_snapshot.yaml` at run time).

---

## 2. Metrics (fill after Kaggle run)

Copy from `/kaggle/working/RSNA-BASELINE-001/metrics_summary.json`:

| Metric | Value |
|---|---:|
| OOF macro ROC-AUC | **PENDING_KAGGLE_RUN** |
| Train runtime (s) | **PENDING** |
| Infer runtime (s) | **PENDING** |
| Peak RSS (MB) | **PENDING** |
| CUDA peak alloc (MB) | **PENDING** |

### Per-label OOF AUC

| Label | AUC |
|---|---:|
| ACL | PENDING |
| MCL | PENDING |
| Medial Meniscus | PENDING |
| Lateral Meniscus | PENDING |
| Medial OA | PENDING |
| Lateral OA | PENDING |
| PF OA | PENDING |
| Effusion | PENDING |
| Synovitis | PENDING |
| Baker's | PENDING |
| Contusion | PENDING |
| Fracture | PENDING |

### Fold-level macro AUC

| Fold | n_val | macro AUC | best epoch |
|---:|---:|---:|---:|
| 0–4 | ~11–12 | PENDING | PENDING |

---

## 3. Analysis questions (answer from metrics_summary.json)

### Which labels are strongest / weakest?
- Read `strongest_weakest` in metrics summary after the run.
- Expectation under gold-only + tiny N: high variance; fluid-related labels
  (Effusion / Synovitis / Baker's) often easier than subtle ligament grades.

### Which folds are unstable?
- Use `fold_stability.std` and `range`. Flag folds with macro AUC
  `|fold - mean| > 0.05` as unstable (expected with ~11 val studies).

### Signs of leakage?
Checklist (enforced in code):
- StudyInstanceUID train∩val == ∅ per fold
- `Report` never used at inference (`assert_vision_only_inference_inputs`)
- PatientID unavailable → residual same-patient multi-study leakage
  **cannot be ruled out** from public CSVs (D-008 limitation)

### Are predictions overly correlated across labels?
- Inspect `oof_label_correlation.csv` / `max_abs_offdiag_pred_corr`.
- If off-diagonal |ρ| ≳ 0.95, the head is collapsing to a shared abnormality prior
  (architecture/head bottleneck), not learning finding-specific signals.

### Likely bottleneck (pre-run call)

**labels** — with only ~58 gold studies, architecture / aug / aggregation
improvements are secondary. The next high-value move after this floor is
**training-time-only** report→pseudo-label supervision (G3), evaluated on the
same gold OOF protocol so the imaging stack stays honest.

Secondary risks once labels expand: series selection (fluid-sensitive coverage),
then input geometry (mid-slice vs multi-slice 2.5D).

---

## 4. How to run on Kaggle

1. Upload this repo (at least `src/`, `configs/`) as a private Kaggle Dataset,
   e.g. `rsna-knee-control`.
2. Create Notebook attached to competition `rsna-knee-abnormality-detection`
   + the code Dataset. GPU accelerator. Internet ON for first weight download.
3. Run `notebooks/kaggle/02_train_baseline.py` (or the `.ipynb` twin).
4. Download **only** small artifacts back to Mac:
   - `metrics_summary.json`, `oof_predictions.csv`, `experiment_card.json`,
     `fold metrics`, `config_snapshot.yaml`, `submission.csv`
   - Do **not** download `train_series/` DICOMs.
5. Paste numbers into this report; flip decision checkbox.

### Local plumbing check (Mac, synthetic only)

```bash
cd /Users/sarasa/Documents/MRI/MRI
pip install -e ".[torch]"
python scripts/run_baseline_smoke_local.py --epochs 1 --folds 2
pytest -q tests/test_baseline_imaging.py -k "not slow"
```

---

## 5. Next three highest-value experiments

1. **RSNA-LABELS-001 (G3 entry)** — Offline report→pseudo-labels (regex or
   frozen LLM), train the *same* EfficientNet-B0 2.5D stack on all 4,407
   studies, evaluate OOF **only on the 58 gold studies** held out by the same
   fold map. Pass if gold OOF macro AUC ↑ > 0.005 vs RSNA-BASELINE-001.
   Inference remains vision-only.

2. **RSNA-INPUT-001 (G4)** — Same labels/model; change only input geometry to
   K=3 evenly spaced slices × 3 planes (9-ch → 1×1 stem) or 288² resolution.
   Pass if gold OOF macro AUC ↑ > 0.005.

3. **RSNA-SERIES-001 (G4b)** — Ablate series policy: fluid-sensitive-only vs
   all-series mean vs fat-sat preference off. Single change. Pass if ↑ > 0.003
   with no fold instability regression.

Do **not** ensemble or HP-search until (1) lands a stable label set.
