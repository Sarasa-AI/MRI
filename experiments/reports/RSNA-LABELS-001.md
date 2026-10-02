# Experiment Report — RSNA-LABELS-001 (Label Audit)

**Status:** COMPLETE (label-quality gate) — imaging OOF deferred  
**Date:** 2026-10-02  
**Execution plane:** Mac control plane on `train.csv` metadata only (no DICOMs)

---

## Hypothesis

Report-derived training labels (strategies B–E) raise gold-held-out imaging OOF
macro AUC by >0.005 vs gold-only (A) without report dependence at inference.

**Pass bar (label gate, this run):** report→gold macro AUC ≥0.70 (preferably >0.80)
with bootstrap CI above chance, before spending GPU on imaging OOF.

**Pass bar (imaging gate, deferred):** gold OOF macro AUC ↑ >0.005 vs
RSNA-BASELINE-001 / strategy A, broad per-label lift, bootstrap CI excludes 0.

---

## Corpus facts (VERIFIED from train.csv)

| Fact | Value |
|---|---:|
| Train studies | 4407 |
| Gold (all 12 non-null) | 58 |
| Partial label rows | 0 (all-or-nothing) |
| Report-only silver pool | 4349 |
| Duplicate report groups | 55 (208 studies) |
| Test reports used | No |

### Gold prevalence (n=58)

| Label | Pos | Neg | Prev |
|---|---:|---:|---:|
| ACL | 24 | 34 | 0.414 |
| MCL | 9 | 49 | 0.155 |
| Medial Meniscus | 26 | 32 | 0.448 |
| Lateral Meniscus | 23 | 35 | 0.397 |
| Medial OA | 15 | 43 | 0.259 |
| Lateral OA | 11 | 47 | 0.190 |
| PF OA | 21 | 37 | 0.362 |
| Effusion | 35 | 23 | 0.603 |
| Synovitis | 27 | 31 | 0.466 |
| Baker's | 12 | 46 | 0.207 |
| Contusion | 19 | 39 | 0.328 |
| Fracture | 18 | 40 | 0.310 |

---

## Report extractor (this experiment)

Rule-based multilingual lexicon with sentence-scoped negation, severity, and
uncertainty. Emits soft ∈[0,1], confidence, and state ∈ {P,N,U,M}.
**Reproducible** (deterministic; no LLM). Artifacts under
`outputs/RSNA-LABELS-001-audit/`.

---

## Label-quality OOF (report labels as predictors vs gold)

| Strategy / predictor | Macro AUC | 95% bootstrap CI | Reproducible |
|---|---:|---|---|
| Chance | 0.500 | [0.500, 0.500] | yes |
| B Hard report | 0.613 | [0.568, 0.657] | yes |
| C/D Soft report | 0.621 | [0.574, 0.666] | yes |
| E High-conf blend | ~0.613 | (see summary JSON) | yes |

### Per-label soft AUC vs gold

| Label | AUC | Note |
|---|---:|---|
| Fracture | 0.805 | usable |
| ACL | 0.792 | usable |
| MCL | 0.768 | usable (wide CI; rare) |
| Baker's | 0.729 | usable |
| Medial Meniscus | 0.659 | marginal |
| Lateral Meniscus | 0.620 | marginal |
| PF OA | 0.618 | marginal |
| Effusion | 0.547 | weak |
| Contusion | 0.535 | weak |
| Lateral OA | 0.515 | chance |
| Medial OA | 0.454 | **harmful** |
| Synovitis | 0.414 | **harmful** (90% silence) |

Lift is **not broad** — driven by ligaments / Baker's / fracture. OA + Synovitis
are at or below chance. 218 gold contradictions: 109 silent misses, 58 FP, 51 FN.

**Hanley–McNeil caution:** with n=58 and rare positives (e.g. MCL n_pos=9),
per-label CI widths are ~0.2–0.4. Do not declare success from gold alone.

---

## Strategy decisions

| Strategy | Decision | Imaging OOF | Rationale |
|---|---|---|---|
| A. Gold only | **KEEP** | pending baseline metrics in-repo | Evaluation anchor / honest floor |
| B. Report hard | **REJECT** | not run | Macro 0.61; injects noise especially on silent findings |
| C. Report soft | **REJECT** *(this extractor)* | not run | Macro 0.62; Synovitis/OA below chance |
| D. Conf-weighted | **REJECT** *(this extractor)* | not run | Same soft scores; weighting cannot fix wrong ranking |
| E. Gold + high-conf silver | **REJECT** *(this extractor)* | not run | High-conf subset still inherits lexicon failure modes |

**Important:** REJECT applies to silver labels from **this regex extractor**.
It does **not** falsify G3 / report supervision as a class. External public
LLM-derived labels report ~0.89 vs gold — that is a different label source and
must be audited separately (train-time only; never at inference).

Inference path remains vision-only. No test reports used.

---

## Imaging OOF (A–E)

**Status:** NOT RUN locally (D-003: no competition DICOMs on Mac).

Plumbing ready:
- `rsna_knee.training.strategy_oof.run_strategy_imaging_oof`
- `notebooks/kaggle/05_label_strategies_oof.py`

Do not run imaging OOF on the current regex silver — expected value is low
given label agreement ≪ 0.70.

---

## Next highest-information experiment

**RSNA-LABELS-002:** Offline train-only LLM (or validated public LLM label set)
→ soft probabilities + confidence; score report→gold macro AUC on the 58.

- **Pass:** macro AUC >0.80 with bootstrap CI >0.75; no label with AUC <0.55
  (or mask those labels).
- **Then:** imaging OOF D/E vs A on Kaggle (`05_label_strategies_oof.py`).
- **Fail:** if LLM still <0.70, pivot to RSNA-INPUT-001 on gold-only.

---

## Artifacts

- `outputs/RSNA-LABELS-001-audit/label_audit_summary.json`
- `outputs/RSNA-LABELS-001-audit/LABEL_AUDIT_REPORT.md`
- `outputs/RSNA-LABELS-001-audit/report_extractions.csv`
- `outputs/RSNA-LABELS-001-audit/gold_report_contradictions.csv`
- `outputs/RSNA-LABELS-001-audit/supervision_*_{labels,weights}.csv`
