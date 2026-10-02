# Experiment Report — RSNA-BASELINE-001-DEBUG-002

**Status:** COMPLETE  
**Date:** 2026-10-02  
**Execution plane:** Kaggle Notebook GPU  
**Kernel:** https://www.kaggle.com/code/alikarimiansarasa/rsna-baseline-001-debug-002  
**Artifacts:** `outputs/RSNA-BASELINE-001-DEBUG-002/` (JSON/CSV only; no DICOMs)

---

## Hypothesis

Correcting the single cache reload bug (`load_cached` prematurely casting uint8 →
float32, which skipped `/255`) restores stable unaugmented validation inputs in
`[0, 1]` across epochs 0–4 under frozen pilot settings, and thereby restores
finite AMP logits / probabilities / OOF.

**Pass bar:** Unaugmented validation inputs `[0, 1]` every epoch; no `[0, 255]`
transition; finite logits/probs/OOF; production metric not NaN from corrupt
predictions; checkpoint reload healthy; submission validates →
`READY_FOR_FULL_BASELINE`. Otherwise stop with `BLOCKED_PENDING_FIX` (and
`NEW_ROOT_CAUSE_DISCOVERED` if inputs fixed but something else fails). No second
production patch.

---

## Exact production change

**Only file:** `src/rsna_knee/data/cache.py`

```python
# BEFORE
return data["image"].astype(np.float32)

# AFTER
return np.copy(data["image"])  # preserve uint8 so load_cached_float /255 runs
```

`save_cached`, AMP, model, loss, optimizer, YAML, split, labels, DICOM, OOF,
metric, checkpoint, and augmentation were **not** changed.

Control-plane dataset `alikarimiansarasa/rsna-knee-control` was re-versioned
with **only** this `cache.py` delta relative to the DEBUG-001 control plane.

---

## Local regression

- Added `tests/test_cache_roundtrip.py` (miss / hit / repeat / no-[0,255]).
- Full suite: **41 passed**.

---

## Kaggle validation (frozen pilot)

| Item | Value |
|---|---|
| GPU | Tesla T4 (CUDA) |
| AMP | enabled |
| Fold | 0 only (`pilot_folds=1`) |
| Epochs | 5 |
| Seed | 42 |
| Model | EfficientNet-B0 multi-plane mid-slice |
| Image size | 224 |
| Labels | gold-only |
| Cache dir | fresh under experiment id (epoch0 miss → epoch1+ hit) |
| Official YAML | unchanged (`epochs: 25`, `pilot_folds: null`) |

DICOM mid-slice audit: **174/174** success.

Cache round-trip probe: on-disk `uint8`, `load_cached` returns `uint8`,
`load_cached_float` returns float32 in `[0, 1]` (max abs err ≈ 0.002).

---

## Critical before → after

### DEBUG-001 (broken)

| Epoch | input max | logits finite | macro AUC |
|---|---:|---|---:|
| 0 | 1.0 | True | 0.408 |
| 1 | **255.0** | **False** | NaN |
| 2–4 | 255 | False | NaN |

### DEBUG-002 (fixed)

| Epoch | input min/max | logits finite | probs finite | macro AUC |
|---|---|---|---|---:|
| 0 | 0.0 / **1.0** | True | True | 0.408 |
| 1 | 0.0 / **1.0** | True | True | 0.446 |
| 2 | 0.0 / **1.0** | True | True | 0.521 |
| 3 | 0.0 / **1.0** | True | True | 0.522 |
| 4 | 0.0 / **1.0** | True | True | 0.513 |

No `[0, 255]` transition. GradScaler stayed at 65536 (no overflow cascade).

---

## Other audits

| Check | Result |
|---|---|
| OOF assignment | 12/12 UIDs matched; finite_count invariant 144==144; identical to val |
| OOF finite cells | 144 (fold-0 val only; other gold rows remain NaN by design under `pilot_folds=1`) |
| Best checkpoint | epoch 3, auc≈0.522, state_dict healthy |
| Reload vs fresh | logits/probs max abs diff **0.0** |
| Test submission | validator **PASS**; predictions finite in `[0, 1]` |
| NEW ISSUES | NONE |

---

## Gate

```text
RSNA-BASELINE-001-DEBUG-002

FIX:
cache uint8 normalization corrected

ROOT_CAUSE:
confirmed

CACHE MISS: PASS
CACHE HIT: PASS
INPUT RANGE: PASS
GPU: PASS
DICOM: PASS
TRAINING: PASS
VALIDATION: PASS
OOF: PASS
METRIC: PASS
CHECKPOINT: PASS
SUBMISSION: PASS

NEW ISSUES:
NONE

GATE:
READY_FOR_FULL_BASELINE
```

---

## STOP

Do **not** auto-launch full 5-fold `RSNA-BASELINE-001`.
Do **not** tune. Next action requires explicit human approval.
