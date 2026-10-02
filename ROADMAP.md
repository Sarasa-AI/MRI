# Roadmap

## Immediate (this week) — RSNA P0

1. Package this repo / `src` as a private Kaggle Dataset for notebook import.
2. (Optional same session) Run `01_metadata_audit.py` for authoritative gold counts.
3. Run `notebooks/kaggle/02_train_baseline.ipynb` on **Kaggle GPU** → fill
   `experiments/reports/RSNA-BASELINE-001.md` with OOF metrics.
4. Next science: `RSNA-LABELS-001` (report pseudo-labels, training-time only).

## Next 7 Days (supersedes 2026-08-20 list for RSNA)

Historical 7-day list (2026-08-20) retained for audit trail:

1. Security: rotate/revoke the exposed Anthropic key and replace plaintext
   credentials; verify approval mode. *(still open)*
2. RSNA: identify the official competition and run a metadata-only probe…
   → **DONE / PARTIAL** via `RSNA-G1-KAGGLE-DATA-ACCESS.md` (2026-10-02).
3. RSNA: create the first executable imaging baseline only after the probe
   succeeds; log it as `RSNA-BASELINE-001`. → **PIPELINE READY** (2026-10-02);
   Kaggle execution pending.
4. Phoenix / Revenue items from 2026-08-20 remain valid at reduced priority
   while competition deadline pressure is active.

## 30 / 60 / 90 Days

Unchanged in spirit from 2026-08-20, with these RSNA corrections:

- **30 days:** study-level CV audit, strong 2.5D baseline on Kaggle, first
  valid submission, experiment registry with OOF lineage. No local DICOM lake.
- **60 days:** multi-plane / weak-supervision ablations only if G2 stable.
- **90 days:** calibrated ensemble + private-LB-aware submit strategy within
  notebooks-only constraints.

Late bets (GNN, CLIP report-image contrastive, from-scratch MAE) stay deferred
until baseline + CV gates pass and compute budget remains.

## Gate board

See [docs/gates.md](docs/gates.md).
