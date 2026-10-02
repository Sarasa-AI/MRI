# Experiment Queue

Only experiments with an explicit hypothesis are listed here.

| ID | Hypothesis | Baseline | Change | Primary metric | Cost | Decision gate |
|---|---|---|---|---|---:|---|
| RSNA-DATA-001 | A metadata-only API probe can map competition identifiers without bulk pagination | No verified RSNA pipeline | Probe metadata, pagination, manifests, and size estimates | Complete mapping coverage + API calls + estimated bytes | Low | **PARTIAL PASS** (2026-10-02) — see `RSNA-G1-KAGGLE-DATA-ACCESS.md`; close via Kaggle CSV audit |
| EXP-000-smoke | Control-plane package can exercise splits/metrics/OOF/submission validation on synthetic fixtures without DICOMs | n/a | Implement skeleton + ConstantPrior smoke | Unit tests green; valid fixture submission | Low | **DEFINED** — run `pytest` + `scripts/run_smoke_local.py` locally |
| RSNA-DATA-002 | Kaggle-mounted CSVs yield authoritative study/series/label stats + StudyInstanceUID fold map without DICOM decode | RSNA-DATA-001 | `notebooks/kaggle/01_metadata_audit.py` | Artifacts written; no Report in test; fold leakage assertions | Low | **PARTIALLY_READY** (2026-10-02) — mount `/kaggle/input/competitions/...`; 4407/24371/58 gold; 2 rare-label zero-pos folds; see `experiments/reports/RSNA-DATA-002.md` |
| RSNA-BASELINE-001 | A reproducible EfficientNet-B0 multi-plane mid-slice 2.5D model on gold labels establishes a beatable OOF macro-AUC floor | No executable imaging baseline | Fixed preprocessing/model/split on Kaggle GPU; gold-only; no pseudo-labels | OOF macro ROC-AUC + per-label + fold metrics + submission | Medium | **PIPELINE READY** — DEBUG-002 cleared NaN blocker; full 5-fold awaits explicit human approval |
| RSNA-BASELINE-001-PILOT | Existing baseline can process real Kaggle DICOMs on GPU for 1 fold / ≤5 epochs without orientation/order/leakage/silent-zero blockers | RSNA-BASELINE-001 (frozen) | Real-DICOM audit + 1-fold pilot; no architecture change | Gate decision + DICOM/GPU/OOF/submission artifacts | Low–Med | **READY_WITH_CAVEATS** (2026-10-02) — DICOM/GPU/submit PASS; OOF all-NaN + val AUC NaN after ep0; see `experiments/reports/RSNA-BASELINE-001-PILOT.md` |
| RSNA-BASELINE-001-DEBUG-001 | Under frozen pilot settings, NaN first appears at one measurable stage; test-vs-OOF explained by that stage | RSNA-BASELINE-001-PILOT | Read-only probes; no architecture/HP change | FIRST_FAILURE + root-cause class + epoch health table | Low | **BLOCKED_PENDING_FIX** (2026-10-02) — cache reload returns [0,255]; AMP logits NaN from epoch 1; see `experiments/reports/RSNA-BASELINE-001-DEBUG-001.md` |
| RSNA-BASELINE-001-DEBUG-002 | Correcting cached uint8→float `/255` restores finite validation/OOF under frozen pilot settings | RSNA-BASELINE-001-DEBUG-001 | Single `load_cached` dtype-preserve fix; no AMP/model/YAML change | Epoch0–4 input `[0,1]` + finite logits/OOF + submission | Low | **READY_FOR_FULL_BASELINE** (2026-10-02) — inputs `[0,1]` all epochs; logits/OOF finite; see `experiments/reports/RSNA-BASELINE-001-DEBUG-002.md` |
| RSNA-LABELS-001 | Training-time report→pseudo-labels lift gold OOF macro AUC >0.005 vs RSNA-BASELINE-001 without inference report dependence | RSNA-BASELINE-001 | Same backbone/geometry; expand train labels offline from reports | Gold-only OOF macro AUC | Medium | **LABEL GATE FAIL for regex extractor** (2026-10-02) — report→gold soft macro AUC=0.621 [0.574,0.666]; B–E REJECT for this source; A KEEP; see `experiments/reports/RSNA-LABELS-001.md` |
| RSNA-LABELS-002 | Offline LLM (train-only) report labels reach report→gold macro AUC >0.80 | RSNA-LABELS-001 regex | Replace extractor; no imaging yet | Report→gold macro AUC + per-label + bootstrap | Low–Med | Proceed to imaging OOF only if pass |
| RSNA-INPUT-001 | Multi-slice or 288² geometry improves gold OOF macro AUC >0.005 vs baseline | RSNA-BASELINE-001 (+ labels if ready) | Single input-geometry change | Gold OOF macro AUC | Medium | Abandon if no lift or runtime blows budget |

| RSNA-SERIES-001 | Fluid-sensitive / fat-sat series policy is a material lever (>0.003 AUC) | Prior best | Single series-selection ablation | Gold OOF macro AUC | Low–Med | Keep winning policy |
| PHX-CTRL-001 | The corrected full-split metric and checkpoint path can be replicated independently | BASELINE-001-R artifacts | No scientific hyperparameter change | `val_qwk_full`, accuracy, class metrics, best/last invariants | High | Accept as valid measurement even if QWK does not improve |
| PHX-IMB-001 | A single imbalance-aware intervention reduces grade-0 collapse without damaging ordinal calibration | PHX-CTRL-001 | One sampling or loss change | QWK, macro F1, grade-0 recall, Brier/ECE | Medium | Reject if aggregate gain hides clinical-critical regression |

## Next experiment to run

**STOP after RSNA-BASELINE-001-DEBUG-002.** Cache fix validated; gate is
`READY_FOR_FULL_BASELINE`. Do **not** auto-launch full 5-fold
`RSNA-BASELINE-001` or any tuning — await explicit human approval.
Pull back only metrics / OOF / cards — never DICOMs.
