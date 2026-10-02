# RSNA G1 — Kaggle Data Access Discovery

**Date:** 2026-10-02  
**Mode:** READ-ONLY Kaggle API + docs (no competition data downloaded)  
**CLI:** Kaggle 2.2.3 via `/Library/Frameworks/Python.framework/Versions/3.11/bin/kaggle`  
**Auth:** Confirmed (`user_has_entered=true`)  
**Local constraint:** Full RSNA dataset must NEVER be downloaded to the Mac.

Evidence tags used below:
- **VERIFIED** — confirmed via Kaggle API / official competition pages in this session
- **INFERRED** — strong evidence from official Kaggle platform docs or consistent third-party/community sources; not re-confirmed by our API against live notebook mounts
- **UNKNOWN** — not established without prohibited download or a live Kaggle Notebook session

---

## A. Verified facts

| Fact | Status | Source |
|---|---|---|
| Competition slug / identifier | **VERIFIED** `rsna-knee-abnormality-detection` | `kaggle competitions list -s rsna-knee` |
| Title | **VERIFIED** RSNA Knee Abnormality Detection | API competition object |
| Competition ID | **VERIFIED** `154281` | API |
| Host | **VERIFIED** Radiological Society of North America | API |
| Category / reward | **VERIFIED** Research / `$77,000` | API |
| Code competition (notebooks only) | **VERIFIED** `is_kernels_submissions_only=true` | API |
| Account entered | **VERIFIED** `user_has_entered=true` | API |
| Final submission deadline | **VERIFIED** 2026-10-22 23:59 UTC | API |
| Entry / merger deadline | **VERIFIED** 2026-10-15 23:59 UTC | API |
| Max daily submissions | **VERIFIED** 5 | API |
| Max team size | **VERIFIED** 5 | API |
| Metric | **VERIFIED** macro-average ROC AUC over 12 labels | Evaluation page + API `evaluation_metric` |
| Submission must be Notebook | **VERIFIED** | Code Requirements page |
| Internet disabled at submit | **VERIFIED** | Code Requirements page |
| External public data + pretrained models allowed | **VERIFIED** | Code Requirements page |
| Submission filename | **VERIFIED** `submission.csv` | Code Requirements page |
| Runtime caps | **VERIFIED** CPU ≤ 9h, GPU ≤ 9h | Code Requirements page |
| Reports present in train only | **VERIFIED** `train.csv` has `Report`; test has no report at scoring | data-description page |
| Approximate full test size | **VERIFIED** “about 1300 studies”; listed `test_*` is example only (3 studies) | data-description page |
| Local account GPU quota remaining | **VERIFIED** 30.00h GPU / 20.00h TPU (refresh 2026-10-03) | `kaggle quota` |
| No competition DICOMs present in this control-plane repo | **VERIFIED** | local filesystem inspection |

**G1 hypothesis (restated):** competition identity, file inventory, sizes, and a Mac-control / Kaggle-compute architecture can be established via API/docs alone, without local DICOM materialization.

---

## B. API-accessible metadata

### Competition object (VERIFIED)

```
ref:                 https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
id:                  154281
title:               RSNA Knee Abnormality Detection
deadline:            2026-10-22T23:59:00Z
merger_deadline:     2026-10-15T23:59:00Z
new_entrant_deadline:2026-10-15T23:59:00Z
enabled_date:        2026-08-05T15:51:31Z
is_kernels_submissions_only: true
max_daily_submissions: 5
max_team_size: 5
evaluation_metric: Roc Auc Score
team_count: ~4858 (at probe time)
user_has_entered: true
```

### Official pages retrieved via API (VERIFIED)

`rules`, `Description`, `Evaluation`, `Timeline`, `data-description`, `Prizes`, `Code Requirements`, `abstract`, `Efficiency Prize Evaluation`, `Acknowledgements`

### Files listing API (VERIFIED)

```
kaggle competitions files -c rsna-knee-abnormality-detection --page-size 200 [--page-token TOKEN]
```

SDK response type: `ApiListDataFilesResponse` with fields `files[]`, `next_page_token`.  
Each file exposes `name`, `ref`, `total_bytes`, `creation_date`.

**No download commands were used.** Full per-DICOM byte summation of `train_series/` was not completed (pagination is ~10⁵–10⁶ files; bulk enumeration was stopped to respect the no-local-dataset constraint). CSV and example-`test_series` sizes were obtained from early listing pages only.

---

## C. Dataset structure

### Top-level files (VERIFIED via API listing + official data-description)

| Path | Role | Size (bytes) | Status |
|---|---|---:|---|
| `sample_submission.csv` | Valid all-0.5 submission template | 470 | **VERIFIED** |
| `test.csv` | Example test study IDs (3 studies); replaced at scoring | 212 | **VERIFIED** |
| `test_series.csv` | Example test series metadata | 2,213 | **VERIFIED** |
| `train.csv` | Study-level labels + `Report` | 5,690,007 (~5.43 MiB) | **VERIFIED** |
| `train_series.csv` | Series-level plane / fluid / fat-sat flags | 3,458,834 (~3.30 MiB) | **VERIFIED** |
| `test_series/` | Example test DICOMs | 557 files / 599,962,984 bytes (~572 MiB) | **VERIFIED** (complete prefix) |
| `train_series/` | Full training DICOMs | large | size **UNKNOWN** exactly; see estimates below |

No other non-`.dcm` top-level competition files appeared in API listing through the start of `train_series/` (**VERIFIED**).

### Schemas (VERIFIED — official data-description)

**`train.csv`** (one row per study):
- `StudyInstanceUID`
- `Report` (free-text, multiple languages)
- 12 binary labels: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

**`train_series.csv` / `test_series.csv`** (one row per series):
- `StudyInstanceUID`
- `SeriesInstanceUID`
- `Fluid_Sensitive` (0/1)
- `Fat_Suppression` (0/1)
- `Anatomical_Plane` ∈ {`Sagittal`,`Coronal`,`Axial`}

**`test.csv`** (scoring-time):
- `StudyInstanceUID` only — **no `Report`**

### Directory layout (VERIFIED)

```
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
test_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

Official notes (**VERIFIED**):
- Each `.dcm` is one slice
- Series typically 20–45 slices (median 30), long tail to a few hundred
- Mixed transfer syntaxes; DICOM tags stripped to an allowlisted set of 86
- Intensities / orientations / resolutions vary
- Example `test_*` is replaced with the real hidden test set (~1300 studies) at scoring
- Abnormality prevalence is **not** guaranteed equal across train / public LB / private LB

### Example path observed in API listing (VERIFIED)

```
test_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm
train_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm
```

Example test coverage in listed files: **3 studies / 15 series / 557 DICOMs** (**VERIFIED**).

### Report availability (VERIFIED)

- Training: radiology reports in `train.csv.Report`
- Inference / test: reports **not** provided
- Implication (aligned with project constraints): any report use is **training-time only** (pseudo-labels / distillation). Vision-only inference path is mandatory.

### Row counts / label sparsity

| Claim | Status |
|---|---|
| ~4,407 training studies; ~24,371 training series | **INFERRED** (consistent community datasets/blogs derived from competition CSVs; not counted here because CSVs were not downloaded) |
| Only a small subset of train studies have per-condition labels; rest rely on reports | **VERIFIED** (official data-description wording) |
| Exact gold-label count (e.g. 58) | **INFERRED** (community analyses); treat as unverified until CSV is read **on Kaggle** |

### Size estimates for `train_series/`

| Estimate | Status | Notes |
|---|---|---|
| Exact API-summed total bytes | **UNKNOWN** | Full DICOM pagination not completed |
| Community published “original dataset ~570 GB” | **INFERRED** | Stated on public derived JPEG dataset card |
| Order-of-magnitude from official median 30 slices × ~24k series × ~0.7 MB/slice (from partial listing) | **INFERRED** | Roughly hundreds of GB; compatible with ~570 GB claim |
| Mac local download | **FORBIDDEN** by project decision | Disk on Mac is insufficient / policy-blocked for full DICOM tree |

**Decision implication:** treat full imaging as Kaggle-resident only.

---

## D. Kaggle runtime structure

### Competition data mount (INFERRED from official Notebook docs + consistent public RSNA notebooks)

When a Notebook is created from / attached to this competition, data is mounted under `/kaggle/input/`:

| Candidate path | Confidence |
|---|---|
| `/kaggle/input/rsna-knee-abnormality-detection/` | **INFERRED** (standard competition mount; seen in public notebooks) |
| `/kaggle/input/competitions/rsna-knee-abnormality-detection/` | **INFERRED** (alternate layout reported on newer kernels) |

**Operational rule:** resolve with `Path("/kaggle/input").rglob("train.csv")` rather than hardcoding one path. Exact live path on our account is **UNKNOWN** until a Notebook probe runs.

### Standard Kaggle paths (VERIFIED — official Notebooks documentation)

| Path | Purpose |
|---|---|
| `/kaggle/input/...` | Attached competition / dataset / model / notebook-output inputs |
| `/kaggle/working/` | Writable output (persisted on Save Version; up to **20 GB**) |
| `/kaggle/usr/lib/notebooks/<owner>/<slug>/` | Alternate notebook-output mount (**INFERRED** community) |
| `/kaggle/input/models/<owner>/<slug>/<framework>/<variation>/<version>` | Kaggle Models mount (**INFERRED** community + Models UX docs) |
| `/kaggle/input/datasets/<owner>/<slug>/` | Alternate dataset mount (**INFERRED** community) |

### Submission runtime constraints (VERIFIED)

- Notebooks-only submission
- Internet **disabled**
- CPU/GPU ≤ 9 hours
- Output must include `/kaggle/working/submission.csv`
- Freely & publicly available external data allowed, including pretrained models
- Hidden test replacement: code must read `test.csv` / `test_series/` dynamically

### Does Kaggle mount competition data at runtime without local download?

**YES — VERIFIED** by competition being `is_kernels_submissions_only` + official Code Competition docs: initializing a Notebook with the competition dataset attaches data in the Kaggle VM; no Mac-side DICOM copy is required.

---

## E. Training architecture recommendation

### Safest split (recommended)

```
MacBook (control plane)              Kaggle (data + GPU compute plane)
─────────────────────────            ─────────────────────────────────
code, configs, docs                  competition DICOMs (mounted)
CV manifests / fold IDs              training notebooks / scripts
report-pseudo-label logic (code)     report text (train.csv on Kaggle only)
experiment logs (summaries)          checkpoints / OOF preds (as Dataset or Notebook output)
git history                          submission notebook (internet OFF)
```

### Why this is the correct architecture

1. **Dataset scale** is hundreds of GB — incompatible with the Mac no-download rule.
2. **Submission path is Notebook-only** with internet off — weights must already live on Kaggle as Dataset / Model / Notebook output.
3. **Mac still adds leverage:** authoring, review, unit tests of split/leakage logic on metadata, Hydra configs, and pushing kernels via CLI — without ever holding DICOMs.
4. **MPS local training on full data is not viable** under the no-download rule; local work stays metadata/code only unless a tiny synthetic fixture is created later.

### Training can be performed entirely inside Kaggle without any local dataset?

**YES — VERIFIED** (platform capability + competition design + remaining GPU quota).  
Caveat (**INFERRED**): weekly GPU hours (30h observed) may be a throughput bottleneck for large 2.5D sweeps; plan experiments tightly or use processed/cached intermediates stored as Kaggle Datasets.

### Recommended Kaggle workflow (no modeling code yet)

1. **Metadata Notebook (G1 completion):** read `train.csv` / `train_series.csv` only; emit study counts, label sparsity, plane/fat-sat distributions, CV fold JSON/CSV as Notebook output.
2. **Cache Notebook (optional, high leverage):** convert selected slices/series to compact arrays (e.g. JPEG/NPZ/NPY) → publish as **private Kaggle Dataset** to avoid repeated DICOM decode cost.
3. **Train Notebook:** attach competition data (+ cache dataset); write checkpoints to `/kaggle/working`; Save Version.
4. **Promote weights:** either (a) attach prior Notebook output, or (b) create a Dataset/Model version from checkpoints.
5. **Submit Notebook:** internet OFF; attach weights + competition data; write `submission.csv`.

---

## F. Submission architecture

```
[Train NB / offline cluster]
        │ checkpoints
        ▼
[Kaggle Dataset]  or  [Kaggle Model]  or  [Train NB Output ≤20GB]
        │ attach as Input
        ▼
[Submit NB]  internet=OFF  GPU/CPU ≤9h
   inputs: competition data + weights (+ optional wheels via Dependency Manager)
   reads:  /kaggle/input/.../test.csv and test_series/
   writes: /kaggle/working/submission.csv
        │
        ▼
   Submit from Notebook Output
```

### Can submission load pretrained weights with Internet disabled?

**YES — VERIFIED** by Code Requirements (“Freely & publicly available external data is allowed, including pre-trained models”) + official Notebook docs (attach Datasets/Models/Notebook outputs under `/kaggle/input`; Dependency Manager installs wheels before offline commit).

Practical rules:
- Attach **all** required weight sources **before** Save & Run All
- Do not rely on runtime downloads (`torch.hub`, HuggingFace, `kagglehub` network fetch) during submit
- Prefer explicit local paths under `/kaggle/input/...`
- Keep total attached weight pack well under Notebook disk/time limits; 20 GB is the persisted `/kaggle/working` cap for a single NB output

---

## G. What must remain on Kaggle

- All `train_series/` and `test_series/` DICOMs
- Full `train.csv` / `train_series.csv` content used for training (may export **aggregates/manifests** only to Mac)
- Report text (PHI/clinical narrative risk + size; training-only)
- Model checkpoints intended for submission
- Any large cached tensors / JPEG stacks
- Final submission Notebook execution

---

## H. What may remain on the Mac

- This control-plane repo (`MRI/MRI`) and implementation git repos
- Code, configs, unit tests (especially leakage-safe GroupKFold tests on **ID lists**)
- Fold manifests / study UID lists / hash fingerprints (no pixels)
- Experiment result tables (CSV/JSON summaries)
- Kernel metadata / push scripts
- Small synthetic fixtures for pipeline unit tests (optional, later)
- Documentation (this file, roadmap, decisions)

**Must not** land on the Mac: competition DICOM trees, bulk CSV downloads of full train tables if that becomes a slippery slope to local imaging pulls, checkpoints copied off Kaggle unless explicitly approved and size-bounded.

---

## I. What is still unknown

1. **Exact live Notebook mount path** on our account (`/kaggle/input/<slug>` vs `/kaggle/input/competitions/<slug>`).
2. **Exact total `train_series/` byte size and file count** from a complete API summation (**UNKNOWN**; community ~570 GB is **INFERRED** only).
3. **Exact train study/series/label counts and gold-label cardinality** without reading CSVs on Kaggle (**INFERRED** only from community).
4. **PatientID availability** in DICOM headers or CSVs for leakage-safe patient-level splits — official CSVs expose `StudyInstanceUID` only; patient-level grouping may be impossible (**UNKNOWN** until header audit on Kaggle). This is a CV-strategy risk.
5. **Whether left/right knee laterality exists as a field** — official 12 labels are compartment findings within a study, not L/R knee flags. Project rule about laterality-dependent flips still needs confirmation against actual DICOM laterality tags (**UNKNOWN**).
6. **Private test replacement behavior details** (exact file swap timing, whether paths stay identical) — assumed standard code-comp behavior (**INFERRED**).
7. **GPU SKU / RAM in current Kaggle image for this account** during training (**UNKNOWN** until Notebook session inspect).
8. **Whether Kaggle Models vs Dataset is better for multi-fold checkpoints** under 20 GB output limits (**UNKNOWN**; engineering choice).

---

## J. Exact next step

**Do not download data locally.**

**Next action (G1 close-out on Kaggle):** create a private Kaggle Notebook attached to `rsna-knee-abnormality-detection` that:

1. Prints resolved input roots under `/kaggle/input`
2. Loads **only** `train.csv` and `train_series.csv`
3. Emits: study count, series count, label non-null rates per column, plane/fluid/fat-sat histograms, languages heuristic on `Report`, and a study-level GroupKFold fold map
4. Saves those artifacts to `/kaggle/working/` as the first authoritative metadata pack
5. Confirms no report column exists in `test.csv`

**Pass criterion:** Notebook completes with internet optional; produces machine-readable manifests; zero DICOM decode required for this step.

**Then:** update this doc’s UNKNOWN items from that Notebook output, and only after that propose G2 baseline training **on Kaggle**.

---

## G1 verdict

### PARTIALLY READY

**Ready:** competition identity, access, notebooks-only constraints, file schema, report train-only rule, example test structure, CSV sizes, Mac-vs-Kaggle architecture, offline weight-reuse path.

**Not fully ready:** exact train DICOM corpus size/file counts, live mount path confirmation, patient-level split feasibility, authoritative label/report statistics (require Kaggle-side CSV read).

**Blocked items for local modeling:** none unexpected — local full-data modeling remains correctly **out of scope** by policy.

---

## Probe log (commands used — read-only)

```bash
# identity
kaggle competitions list -s rsna-knee --csv

# files (listing only; no download)
kaggle competitions files -c rsna-knee-abnormality-detection --page-size 200

# official pages
kaggle competitions pages -c rsna-knee-abnormality-detection --content --page-name data-description
kaggle competitions pages -c rsna-knee-abnormality-detection --content --page-name "Code Requirements"

# quota
kaggle quota

# Python SDK listing fields (name, total_bytes) — metadata only
# api.competition_list_files(...); api.competitions_list(...); api.competition_list_pages(...)
```

**Explicitly NOT run:** `kaggle competitions download`, `kaggle datasets download`, wget/curl of dataset URLs, DICOM fetches, bulk CSV materialization to disk.
