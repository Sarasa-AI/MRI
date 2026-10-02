# Experiment Card — RSNA-LABELS-001

## Required fields
- **experiment_id**: RSNA-LABELS-001
- **configuration**: scripts/run_label_audit.py + rsna_knee.labels.{report_parser,strategies,audit}
- **seed**: 42
- **model**: n/a (label-quality gate; no imaging train in this card)
- **input_geometry**: n/a
- **data_selection**: full train.csv (4407); eval on gold 58 only
- **labels**: A gold / B hard report / C soft / D conf-weighted / E gold+silver
- **folds**: StudyInstanceUID GroupKFold seed=42 (imaging path ready; not executed)
- **augmentation**: n/a (label audit)
- **optimizer**: n/a
- **scheduler**: n/a
- **metric**: report→gold macro ROC-AUC = 0.6213 soft / 0.6130 hard; bootstrap 1000
- **runtime**: Mac CPU ~40s for full audit (no DICOMs)
- **oof_predictions**: outputs/RSNA-LABELS-001-audit/report_soft_predictions.csv
- **conclusion**: REJECT B–E for regex silver; KEEP A; next RSNA-LABELS-002

## Hypothesis
Offline report→pseudo-labels can improve gold imaging OOF without inference
report use — but only if report→gold agreement is strong first.

## Pass / fail criterion
- Label gate: report→gold macro AUC ≥0.70 (prefer >0.80) before imaging OOF
- Imaging gate (deferred): gold OOF ↑ >0.005 vs A, broad lift, CI excludes 0
- Report never required at inference

## Decision
- [x] A_gold KEEP
- [x] B–E REJECT (this extractor)
- [ ] imaging OOF (blocked on label quality)

## Notes
- Inference remains vision-only.
- Do not declare success from n=58 alone.
- Synovitis/OA/Effusion/Contusion are the failure modes of the current lexicon.
