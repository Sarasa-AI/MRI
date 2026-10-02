# Local data policy
#
# This directory must NEVER contain the RSNA competition DICOM corpus.
# `fixtures/` holds synthetic metadata CSVs for unit tests only.
#
# Controlled exception (G3 label science, 2026-10-02): `raw/train.csv` may
# exist locally for report/label audits. It is gitignored via `data/raw/`.
# Never download `train_series/` or other DICOM trees to the Mac.
#
# On Kaggle, competition data is mounted under `/kaggle/input/...` and
# resolved by `rsna_knee.runtime.adapter.resolve_data_paths()`.
#
# Override with env var `RSNA_DATA_ROOT` when needed (Kaggle or approved
# external machines only — not for bulk Mac DICOM downloads).
