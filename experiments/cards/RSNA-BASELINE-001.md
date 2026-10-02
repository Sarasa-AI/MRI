# Experiment Card — RSNA-BASELINE-001

## Required fields
- **experiment_id**: RSNA-BASELINE-001
- **configuration**: configs/experiment/rsna_baseline_001.yaml
- **seed**: 42
- **model**: EfficientNetB0MultiPlane25D / torchvision efficientnet_b0 (ImageNet)
- **input_geometry**: multi-plane mid-slice 2.5D, 3×224×224 (Sag/Cor/Ax)
- **data_selection**: gold_labels_only (~58 studies with all 12 targets non-null)
- **labels**: official_12_targets
- **folds**: GroupKFold n_splits=5 by StudyInstanceUID (seed=42); ACL stratify when possible
- **augmentation**: H-flip w/ medial↔lateral swap (D-007); ±10° rotate; brightness/contrast ±0.15; scale 0.90–1.00
- **optimizer**: AdamW lr=1e-4 weight_decay=1e-4
- **scheduler**: CosineAnnealingLR
- **metric**: macro_roc_auc (OOF) + per-label AUC — **PENDING_KAGGLE_RUN**
- **runtime**: Kaggle GPU; AMP on CUDA — **PENDING_KAGGLE_RUN**
- **oof_predictions**: `/kaggle/working/RSNA-BASELINE-001/oof_predictions.csv` — **PENDING_KAGGLE_RUN**
- **conclusion**: **PENDING_KAGGLE_RUN** — local synthetic smoke validates plumbing only

## Hypothesis
A single EfficientNet-B0 multi-plane mid-slice 2.5D model trained on the 58
gold-labeled studies with StudyInstanceUID 5-fold CV establishes a trustworthy
OOF macro-AUC floor that later experiments must beat on the same protocol.

## Pass / fail criterion
- Zero StudyInstanceUID overlap across folds (programmatic assertion)
- Artifacts written: OOF, fold metrics, per-label AUC, submission.csv, experiment card
- Wall-clock fits Kaggle GPU ≤9h
- Report at inference == false

## Decision
- [ ] proceed (after Kaggle metrics land)
- [ ] iterate
- [ ] abandon

## Notes
- Explicitly excludes pseudo-labels, LLM labels, ensembles, HP search.
- PatientID unavailable in official CSVs — study-level grouping is best available.
- True competitive ceiling is label-limited until G3 weak supervision.
