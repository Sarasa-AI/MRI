# Experiment Report — RSNA-BASELINE-001-DEBUG-001

**Status:** COMPLETE (diagnosis only — no fix applied)  
**Date:** 2026-10-02  
**Execution plane:** Kaggle Notebook GPU  
**Kernel:** https://www.kaggle.com/code/alikarimiansarasa/rsna-baseline-001-debug-001  
**Artifacts:** `outputs/RSNA-BASELINE-001-DEBUG-001/` (JSON/CSV only; no DICOMs)

---

## Hypothesis

Under frozen pilot settings (fold 0, 5 epochs, seed 42, EfficientNet-B0, AMP on),
NaN first appears at one measurable stage between DICOM tensors and the metric, and
the valid test submission versus all-NaN OOF is explained by that stage.

**Pass bar:** Name `FIRST_FAILURE` + root-cause class from measured arrays; fill the
epoch 0–4 health table; record validation↔OOF finite-count invariant fields.

---

## A. Executive diagnosis

```text
ROOT_CAUSE:
MULTIPLE

PRIMARY_CAUSE:
DATA/LABEL_PROBLEM
  (mid-plane disk cache reload returns float32 in [0,255] instead of [0,1])

SECONDARY_CAUSE:
MODEL_NUMERICAL_INSTABILITY
  (AMP float16 validation forward overflows → all-NaN logits/probabilities)

CONFIDENCE:
HIGH
```

**WHAT IS BROKEN:** Validation (and train, after the first epoch) consume cached
images at ~255× the intended intensity. Under CUDA AMP this yields all-NaN logits
from epoch 1 onward. Epoch-0 AUC 0.408 is real; later NaN macro-AUC is real.

**WHY:** [`save_cached`](src/rsna_knee/data/cache.py) stores `uint8 = round(x*255)`.
[`load_cached`](src/rsna_knee/data/cache.py) does `astype(np.float32)` **before**
[`load_cached_float`](src/rsna_knee/data/cache.py) checks `dtype == uint8`, so the
`/255.0` branch never runs and values stay in `[0, 255]`.

**WHERE IT FIRST BECOMES BROKEN:** First **cache hit** (epoch 1 validation / later
train batches), stage `cached_input_scale` → then `logits` under AMP.

**EVIDENCE (measured):**

| Probe | Epoch 0 | Epoch 1 |
|---|---|---|
| Input max | **1.0** | **255.0** |
| AMP logits finite | True | False (144/144 NaN) |
| Shadow FP32 probs finite | True | True |
| Param NaN tensors | 0 | 0 |
| Buffer NaN tensors | 0 | 0 |
| Macro AUC | 0.408 | NaN |
| Per-label status | all VALID | all PREDICTION_NAN |

On-disk cache dtype confirmed `uint8` with max 255.

---

## B. Epoch transition

```text
epoch 0: cache MISS → decode path returns [0,1] → save uint8 → return [0,1]
         AMP logits finite (min≈-0.54, max≈0.43) → probs finite → macro AUC 0.408
         checkpoint saved (best)

epoch 1: cache HIT → load_cached casts uint8→float32 (0..255) → no /255
         inputs max=255 → AMP fp16 forward → logits all NaN → probs all NaN
         macro AUC undefined (0 valid labels) → checkpoint NOT replaced
         shadow FP32 forward still finite (wrong scale, saturated sigmoid)
```

Class counts do **not** change between epochs. `ONE_CLASS_METRIC_LIMITATION` is
ruled out: epoch 0 had 12/12 VALID labels; epoch 1 statuses are `PREDICTION_NAN`.

---

## C. Tensor health

| Stage | Health |
|---|---|
| Inputs (epoch 0) | finite, `[0,1]` |
| Inputs (epoch 1+) | finite, **`[0,255]`** (wrong) |
| AMP logits (epoch 0) | finite, fp16 |
| AMP logits (epoch 1+) | **all NaN** |
| AMP probabilities | track logits |
| Shadow FP32 probs | finite every epoch |
| `y_true` | finite every epoch; pos/neg stable |

AMP dtypes: inputs `float32`, params `float32`, logits/probs under autocast `float16`.

---

## D. Parameter / gradient / checkpoint health

| Item | Result |
|---|---|
| Parameters | finite all epochs (max abs ≈ 15.57) |
| Buffers (BN) | finite; max abs grows 543 → 710 (no NaN) |
| Gradients | epoch 0 finite; epochs 1/2/4 show some NaN grad tensors under AMP; losses stay finite |
| GradScaler scale | 65536 → 32768 → 16384 → 8192 (overflow pressure) |
| Best checkpoint | epoch 0, auc=0.408, **state_dict healthy** (0 NaN/Inf tensors) |
| Selection rule | `macro == macro and macro > best_auc` correctly rejects NaN |

Post-reload validation still NaNs because the **input cache** is already wrong, not
because the checkpoint is corrupt. Fresh load of the same ckpt matches.

---

## E. Metric health

Production `macro_roc_auc` and independent sklearn agree:

- Epoch 0: production macro = 0.40814594 = sklearn macro; 12 evaluated labels.
- Epoch 1: both undefined; all 12 labels `PREDICTION_NAN`.

Metric implementation is **not** the primary bug. Undefined→NaN contract preserved.

---

## F. OOF health

| Field | Value |
|---|---|
| Init | 58×12 all NaN |
| Assignment | production `oof.loc[idx, cols] = score`; 12/12 UIDs matched |
| Scores assigned | all NaN (from post-reload AMP val) |
| `same_shape` | True |
| `same_uid_order` | True |
| `same_column_order` | True |
| `finite_count_validation` | 0 |
| `finite_count_oof` | 0 |
| `nan_count_validation` | 144 |
| `nan_count_oof` | 144 |
| `max_abs_diff` | 0.0 |
| **finite-count invariant** | **HOLDS** (`0 == 0`) |
| identical | True |

OOF assignment is **not** dropping finite scores. It faithfully writes NaN validation
predictions. This is **not** an `OOF_ASSIGNMENT_BUG`.

---

## G. Test-vs-OOF discrepancy

```text
test submission = finite   (validator PASS)
OOF             = all NaN
```

**Why:**

1. Test uses a **separate** cache directory (`cache_midplane_224_test`).
2. Inference is a **single pass**: cache miss → decode → `[0,1]` returned to the
   model → then uint8 saved. The buggy reload path is never exercised for test.
3. OOF uses the train cache after many hits, then AMP val on `[0,255]` → NaN scores
   → `.loc` writes those NaNs.

Column-wise test write vs row-wise OOF write is **not** the differentiator here;
input cache scale is.

---

## Epoch health table (mandatory)

| Epoch | logits finite | probs finite | y_true finite | pred finite | valid labels | macro AUC |
| ----- | ------------- | ------------ | ------------- | ----------- | ------------ | --------- |
| 0     | True          | True         | True          | True        | 12           | 0.408     |
| 1     | False         | False        | True          | False       | 0            | NaN       |
| 2     | False         | False        | True          | False       | 0            | NaN       |
| 3     | False         | False        | True          | False       | 0            | NaN       |
| 4     | False         | False        | True          | False       | 0            | NaN       |

Source: `outputs/RSNA-BASELINE-001-DEBUG-001/epoch_health.csv`

---

## Proposed minimal fix (NOT APPLIED)

In `src/rsna_knee/data/cache.py`:

1. Keep storing quantized uint8.
2. On load, **always** rescale: `return data["image"].astype(np.float32) / 255.0`
   (or check max>1 / original uint8 before the premature cast).
3. Add a unit test: save `[0,1]` float → reload → `max <= 1 + 1e-5`.
4. Do **not** change EfficientNet, AMP, loss, OOF `.loc`, or baseline YAML in the
   same change set.

Optional follow-up (separate experiment): clear existing bad caches on Kaggle
working dirs before the next pilot/baseline.

---

## Root-cause classification

```text
MULTIPLE
  PRIMARY:   DATA/LABEL_PROBLEM   (cache reload scale)
  SECONDARY: MODEL_NUMERICAL_INSTABILITY  (AMP fp16 NaN logits on bad inputs)
```

Ruled out as primary: `OOF_ASSIGNMENT_BUG`, `METRIC_BUG`, `CHECKPOINT_BUG`,
`ONE_CLASS_METRIC_LIMITATION`.

---

## Gate

```text
GATE: BLOCKED_PENDING_FIX
NEXT_ACTION: STOP — await review
```

Do **not** run full 5-fold `RSNA-BASELINE-001` until the cache rescale fix is
reviewed and a short re-pilot confirms epoch 1+ finite AMP logits + finite OOF.
