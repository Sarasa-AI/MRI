# Experiment Queue

Only experiments with an explicit hypothesis are listed here.

| ID | Hypothesis | Baseline | Change | Primary metric | Cost | Decision gate |
|---|---|---|---|---|---:|---|
| RSNA-DATA-001 | A metadata-only API probe can map competition identifiers without bulk pagination | No verified RSNA pipeline | Probe metadata, pagination, manifests, and size estimates | Complete mapping coverage + API calls + estimated bytes | Low | Proceed to selective acquisition only if resumable and within storage budget |
| RSNA-BASELINE-001 | A reproducible 2D baseline establishes a valid local reference after data integrity passes | No executable baseline | Single fixed preprocessing/model/split | Competition metric plus per-class/error report | Medium | Keep only if split and leakage audits pass |
| PHX-CTRL-001 | The corrected full-split metric and checkpoint path can be replicated independently | BASELINE-001-R artifacts | No scientific hyperparameter change | `val_qwk_full`, accuracy, class metrics, best/last invariants | High | Accept as valid measurement even if QWK does not improve |
| PHX-IMB-001 | A single imbalance-aware intervention reduces grade-0 collapse without damaging ordinal calibration | PHX-CTRL-001 | One sampling or loss change | QWK, macro F1, grade-0 recall, Brier/ECE | Medium | Reject if aggregate gain hides clinical-critical regression |

