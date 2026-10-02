# Project State

Last verified: 2026-10-02

## Executive State (current)

- **RSNA Knee 2026 is P0.** Competition slug
  `rsna-knee-abnormality-detection` (**VERIFIED**). Final deadline
  **2026-10-22**; entry/merger **2026-10-15**.
- This repo (`MRI/MRI`) is now the **Mac control plane + RSNA engineering
  skeleton**. Competition DICOMs stay on Kaggle. No local dataset download.
- G1 data-access probe: **PARTIAL PASS** — see
  `RSNA-G1-KAGGLE-DATA-ACCESS.md`.
- G1b metadata audit (**RSNA-DATA-002**): **PARTIALLY_READY** (2026-10-02) —
  mount `/kaggle/input/competitions/rsna-knee-abnormality-detection`; 4407
  studies / 24371 series / 58 gold; Report train-only; no PatientID/Laterality
  in CSVs; StudyInstanceUID folds leakage-safe with 2 rare-label zero-pos
  fold cells. See `experiments/reports/RSNA-DATA-002.md`.
- Engineering skeleton (package, configs, metrics, StudyInstanceUID CV,
  submission validator, Kaggle templates, unit tests): **implemented
  2026-10-02**.
- **RSNA-BASELINE-001 imaging pipeline: READY** (2026-10-02) — EfficientNet-B0
  multi-plane mid-slice 2.5D, gold-only, Kaggle-executable. Real OOF metrics
  **PENDING_KAGGLE_RUN**. Local synthetic smoke available.
- **RSNA-LABELS-001 label audit: COMPLETE** (2026-10-02) — regex report→gold
  soft macro AUC **0.621** [0.574, 0.666]. Strategies B–E **REJECT** for this
  extractor; A **KEEP**. Next: **RSNA-LABELS-002** (stronger offline labels).
- Biggest current bottleneck: label quality for silver supervision; imaging
  baseline GPU run still needed for the floor number.

---

## Historical snapshot (2026-08-20) — retained

> Last verified: 2026-08-20
>
> - Portfolio objective: maximize outcome per unit of time across RSNA, Phoenix,
>   agent capability, and near-term revenue.
> - Control-plane repo: `MRI/MRI`, Git `main`, initial commit only, clean at audit
>   start.
> - Primary execution repos: `SARASA_kaggle` and `phonix`; they are separate Git
>   worktrees and are not currently wired into this repo.
> - Biggest bottleneck: the RSNA repo has no verified knee dataset manifest,
>   DICOM acquisition index, or executable imaging baseline. The central repo is
>   also only a control plane, not a model implementation.

**Reconciliation notes (2026-10-02):**

1. `SARASA_kaggle` was inspected and is a **heart-disease tabular** project —
   not RSNA knee. Do not treat it as the RSNA implementation home.
2. RSNA skeleton is co-located here per **D-006**.
3. Bulk DICOM acquisition to Mac remains **forbidden** (D-003 amendment); G1
   replaced “selective local DICOM download” with Kaggle-mounted compute.
4. Security finding on plaintext Anthropic credentials in Claude settings
   remains open from the 2026-08-20 audit (not modified in this change).

## Portfolio Dashboard (updated 2026-10-02)

| Project | Status | Priority | Current phase | Verified latest result | Blocker | Next owner/action |
|---|---|---:|---|---|---|---|
| RSNA knee | ACTIVE | P0 | G2 baseline ready; awaiting Kaggle GPU run | Pipeline + synthetic smoke; OOF pending | Run `02_train_baseline` on Kaggle; pull metrics only | Human + agent: upload Dataset, Save Version |
| Revenue engine | ACTIVE | P1 | Offer validation | Unchanged from 2026-08-20 | ICP / outreach | Human + Business Agent |
| Agent ecosystem | ACTIVE | P2 | Capability inventory | Unchanged | Security boundary | Defer unless blocking RSNA |
| Phoenix | ACTIVE | P2 | Training-core reliability | Unchanged | External datasets / clinical metrics | After RSNA G2 or parallel only if spare capacity |

## Verified Environment (2026-10-02 delta)

- Competition: notebooks-only, internet OFF at submit, GPU/CPU ≤9h, metric =
  macro ROC-AUC over 12 labels (**VERIFIED**).
- Local account GPU quota was 30h at G1 probe time (**VERIFIED** then; re-check
  before training).
- Package: `src/rsna_knee`, Python ≥3.11, pytest suite for metrics/splits/
  submission/runtime.

## STOP / CONTINUE / START (2026-10-02)

### Stop

- Stop any plan to download full `train_series/` to the Mac.
- Stop modeling work that requires Report at inference.
- Stop treating `SARASA_kaggle` as the RSNA knee codebase.

### Continue

- Continue Mac control / Kaggle compute architecture from G1.
- Continue experiment-card discipline (D-004/D-006 fields).

### Start

- Start Kaggle GPU run of `RSNA-BASELINE-001` (`notebooks/kaggle/02_train_baseline.ipynb`).
- Start packing this repo as a private Kaggle Dataset for notebook import.
- After metrics land: start RSNA-LABELS-001 (report→pseudo-labels, training-time only).

## Missing Information That Changes Decisions

1. ~~Exact live Notebook mount path~~ — **CLOSED** by RSNA-DATA-002:
   `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
2. ~~Authoritative gold-label count / prevalence~~ — **CLOSED**: 58 gold;
   see `outputs/RSNA-DATA-002/label_coverage.csv`.
3. Whether DICOM headers expose PatientID / Laterality usable for CV/aug
   (CSV: absent; headers still UNKNOWN).
4. Train DICOM total bytes (community ~570GB inferred only).
5. GPU SKU/RAM on Kaggle image for this account.
