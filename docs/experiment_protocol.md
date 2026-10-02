# Experiment Protocol

Every experiment must produce an `ExperimentCard` with:

1. experiment_id
2. configuration
3. seed
4. model
5. input_geometry
6. data_selection
7. labels
8. folds
9. augmentation
10. optimizer
11. scheduler
12. metric
13. runtime
14. oof_predictions
15. conclusion

## Rules

- One primary change per experiment.
- Primary metric: **macro ROC-AUC** over the 12 official targets.
- Always report **per-label AUC** alongside macro.
- Always store OOF predictions keyed by `StudyInstanceUID`.
- Local Mac runs are metadata/fixtures only unless an approved external machine is used.
- Do not start GNN / CLIP / MAE work until baseline + CV gates pass.

## Templates

- Markdown: `experiments/templates/experiment_card.md`
- JSON schema: `rsna_knee.utils.experiment.ExperimentCard`
- Registry: `experiments/registry.yaml`
- Results table: `experiments/results/results_table.csv` (generated)
