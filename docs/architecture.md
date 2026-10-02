# Architecture — Mac Control Plane / Kaggle Data Plane

## Split of responsibilities

| Concern | Mac (this repo) | Kaggle |
|---|---|---|
| Source code, configs, tests | yes | consume via Dataset / pasted notebooks |
| Experiment definitions + cards | yes | write metrics/OOF back as Notebook output |
| Competition DICOMs | **never** | mounted under `/kaggle/input/...` |
| GPU training / inference | no (policy) | yes |
| `submission.csv` | validate schema locally on fixtures | generate for real test set |

## Path resolution

`rsna_knee.runtime.adapter.resolve_data_paths()`:

1. Explicit `root=` argument
2. `RSNA_DATA_ROOT` env var
3. Kaggle discovery via `train.csv` under `/kaggle/input`
4. Local `data/fixtures` for metadata-only mode

No committed hard-coded Mac dataset path.

## Report boundary

- `Report` may be used **offline during training** only (pseudo-labels / distillation).
- Inference / submission path is vision-only. Enforced by
  `assert_vision_only_inference_inputs`.

## Leakage policy

- Group by `StudyInstanceUID` (only ID available in official CSVs).
- Patient-level grouping is **unknown / unavailable** in released metadata.
- Series-level frames must be collapsed before fold assignment.

## Compartment flip policy

Horizontal flips allowed **only** with medial↔lateral label swaps:

- Medial Meniscus ↔ Lateral Meniscus
- Medial OA ↔ Lateral OA

See decision D-007.
