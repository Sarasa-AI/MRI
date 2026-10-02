# Decisions

## D-001 Portfolio ownership

The MRI repo is the portfolio control plane, not a duplicate implementation
repo. RSNA and Phoenix remain in their existing repositories until a measured
integration need justifies consolidation.

### Amendment (2026-10-02) — see D-006

RSNA knee engineering skeleton is now co-located in this repo because no RSNA
knee implementation existed in `SARASA_kaggle` (verified heart-disease tabular
repo). Phoenix remains separate. Historical D-001 text retained above.

## D-002 Priority allocation

For the next 7 days: RSNA data unblock is P0, revenue validation runs in
parallel as P0/P1, Phoenix receives one controlled experiment, and agent
infrastructure work is limited to security and routing needed by those tasks.

### Amendment (2026-10-02)

RSNA remains P0. Competition deadline is **2026-10-22**. Entry/merger deadline
**2026-10-15**. Focus this repo on Kaggle-executable pipeline + leakage-safe CV.

## D-003 Data safety

Never commit DICOMs, raw medical datasets, checkpoints, secrets, or generated
submissions. Use versioned manifests, hashes, and external storage references.

### Amendment (2026-10-02)

Strengthen: the local Mac must **not** contain the RSNA competition dataset.
No `kaggle competitions download` for this competition on the Mac. Synthetic
fixtures under `data/fixtures/` are allowed for unit tests.

## D-004 Experiment discipline

No experiment is valid without a hypothesis, baseline, single primary change,
authoritative metric, artifact path, interpretation, and next decision.

### Amendment (2026-10-02)

Every experiment must also record the required `ExperimentCard` fields
(experiment ID, configuration, seed, model, input geometry, data selection,
labels, folds, augmentation, optimizer, scheduler, metric, runtime, OOF
predictions, conclusion). See `docs/experiment_protocol.md`.

## D-005 Agent permissions

Default agent mode is least privilege: read-only research by default; writes,
code execution, external communication, credentials, destructive actions, and
submissions require explicit human approval or a narrowly scoped tool policy.

## D-006 RSNA skeleton co-located in control plane (2026-10-02)

**Decision:** Implement the RSNA Knee 2026 engineering package in `MRI/MRI`
under a Mac control / Kaggle data+compute architecture.

**Why:** G1 verified notebooks-only code competition + hundreds-of-GB DICOM
corpus; `SARASA_kaggle` is not an RSNA knee codebase; local full-data training
is policy-blocked.

**Implication:** Modeling code may live here, but execution of imaging
training/inference targets Kaggle runtimes.

## D-007 Horizontal flip / compartment label policy (2026-10-02)

**Decision:** Allow horizontal flips only with medial↔lateral label swaps
(`Medial Meniscus`↔`Lateral Meniscus`, `Medial OA`↔`Lateral OA`). Otherwise
disable H-flips.

**Why:** Official targets include compartment findings; naive flips would
invert labels. This is distinct from left/right knee `PatientLaterality`,
which is not present in official CSVs (**UNKNOWN** in DICOM headers until
Kaggle audit).

## D-008 Grouping key (2026-10-02)

**Decision:** Use `StudyInstanceUID` for GroupKFold / StratifiedGroupKFold.
Assert zero study overlap between folds in code.

**Limitation:** No `PatientID` in released CSVs; true patient-level leakage
prevention is not possible with public metadata alone.

## D-009 Baseline model / geometry (2026-10-02)

**Decision:** First imaging baseline (`RSNA-BASELINE-001`) is EfficientNet-B0
with multi-plane mid-slice 2.5D input at 224² (Sagittal/Coronal/Axial channels),
fluid-sensitive series preference, gold-labels-only, 5-fold StudyInstanceUID
GroupKFold, compartment-safe H-flips (D-007).

**Why:** Strong ImageNet prior, low VRAM, clinical plane coverage without full
3D; gold-only keeps the floor honest before G3 pseudo-labels.

**Non-goals for this experiment:** pseudo-labels, LLM labels, ensembles,
hyperparameter search, report at inference.

## D-010 Label strategy after RSNA-LABELS-001 audit (2026-10-02)

**Decision:** KEEP gold-only (A) as the imaging floor. REJECT regex-derived
silver strategies B–E from the current extractor (report→gold soft macro AUC
0.621). Do **not** abandon G3 — next gate is a stronger offline label source
(RSNA-LABELS-002: train-only LLM labels), then imaging OOF for D/E.

**Why:** Label quality is the binding constraint, but *this* silver source is
too noisy (Synovitis/OA ≤ chance; 218 gold contradictions). Community LLM
labels (~0.89 vs gold) are a different hypothesis.

**Inference:** unchanged — vision-only; no test reports.

