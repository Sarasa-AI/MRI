# Gate Roadmap (G1 → G8)

Aligned with CLAUDE.md and G1 discovery (`RSNA-G1-KAGGLE-DATA-ACCESS.md`).

| Gate | Hypothesis | Minimal experiment | Pass / fail | Decision |
|---|---|---|---|---|
| G1 Data access | Competition identity + Mac/Kaggle split can be established without local DICOMs | API/docs probe + architecture doc | Competition slug, schemas, notebook-only constraints verified | **PASS** (closed by G1b RSNA-DATA-002, 2026-10-02) |
| G1b Metadata audit | Official CSVs yield study/series/label statistics + fold map on Kaggle | `notebooks/kaggle/01_metadata_audit.py` | Emits study_folds + label_coverage; confirms no Report in test | **PARTIALLY_READY** (2026-10-02) — proceed to G2 with rare-label fold caveat |
| G1c Leakage CV | StudyInstanceUID GroupKFold has zero overlap | Unit tests + fold artifact | Assertions green on fixtures and Kaggle fold map | **local tests in place**; blocks modeling |
| G2 Baseline | A single 2.5D CNN is a beatable reference | `RSNA-BASELINE-001` EfficientNet-B0 multi-plane mid-slice on Kaggle gold | CV macro AUC logged with OOF; leakage audit passes | **PIPELINE READY** — execute on Kaggle; decide proceed/iterate after metrics |
| G3 Weak supervision | Report-derived pseudo-labels improve CV without inference report dependence | Offline parse → joint loss / distillation | CV AUC ↑ >0.005 on held-out study split; inference path report-free | **LABEL GATE:** regex silver REJECT (macro 0.621 vs gold); next RSNA-LABELS-002 |

| G4 Plane/series selection | Using fluid-sensitive / multi-plane inputs improves macro AUC | Controlled input ablation | CV AUC ↑ >0.005 vs G2 | proceed / iterate / abandon |
| G5 Augmentation | Compartment-safe flips/intensity aug help | Single aug change | CV AUC ↑ >0.003; no medial/lateral inversion bugs | proceed / iterate / abandon |
| G6 Calibration / threshold | Probabilities are well-ranked for macro AUC | Temperature / stacking on OOF | OOF macro AUC ↑ without leakage | proceed / iterate / abandon |
| G7 Ensemble | 2–3 diverse seeds/folds beat single model | Fixed recipe ensemble | Public LB consistent with OOF; submit ≤5/day budget | proceed / iterate / abandon |
| G8 Submit strategy | Offline notebook reproducibly emits valid submission | `03_submit.py` + validator | Validator OK; notebook ≤9h internet OFF | proceed |

Late-stage bets (GNN, CLIP report-image, from-scratch MAE) are out of scope until G2+G1c pass and compute budget remains.
