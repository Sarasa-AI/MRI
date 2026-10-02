"""Kaggle G1b: metadata-only CSV audit (no DICOM decode / no training).

Attach competition `rsna-knee-abnormality-detection` and run this script
(or the companion notebook). Reads ONLY:

  train.csv, train_series.csv, test.csv, test_series.csv, sample_submission.csv

Never recursively enumerates DICOM trees. Never opens .dcm files.

Outputs under /kaggle/working/ (or RSNA_WORKING_DIR):
  - runtime_paths.json
  - study_summary.json
  - series_summary.json
  - label_coverage.csv
  - study_folds.csv
  - GATE_DECISION.json
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

COMPETITION_SLUG = "rsna-knee-abnormality-detection"
TARGET_COLUMNS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
REQUIRED_CSVS = (
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
)
PATIENT_ID_PATTERNS = (
    r"^patientid$",
    r"^patient_id$",
    r"^patient$",
    r"^subjectid$",
    r"^subject_id$",
    r"^subject$",
    r"^caseid$",
    r"^case_id$",
)
LATERALITY_PATTERNS = (
    r"^laterality$",
    r"^patientlaterality$",
    r"^patient_laterality$",
    r"^side$",
    r"^kneeside$",
    r"^knee_side$",
    r"^left_right$",
    r"^lr$",
)
ORIENTATION_HINTS = (
    "orientation",
    "imageorientation",
    "image_orientation",
    "patientorientation",
    "ipp",
    "iop",
    "rows",
    "columns",
    "slice",
    "spacing",
)
SEQUENCE_HINTS = (
    "sequence",
    "seriesdescription",
    "series_description",
    "protocol",
    "pulse",
    "weighting",
    "contrast",
    "fluid",
    "fat",
    "plane",
    "anatomical",
)
N_SPLITS = 5
SEED = 42


def _working_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    out = Path(os.environ.get("RSNA_WORKING_DIR", "outputs/RSNA-DATA-002"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _top_level_entries(root: Path) -> list[str]:
    if not root.exists():
        return []
    # Non-recursive listing only — never walk train_series/test_series.
    return sorted(p.name for p in root.iterdir())


def resolve_competition_root() -> tuple[Path, dict[str, Any]]:
    """Resolve competition mount without DICOM recursion."""
    input_root = Path("/kaggle/input")
    candidates = [
        input_root / COMPETITION_SLUG,
        input_root / "competitions" / COMPETITION_SLUG,
    ]
    probed: list[dict[str, Any]] = []
    for cand in candidates:
        exists = cand.exists()
        has_train = (cand / "train.csv").exists() if exists else False
        probed.append(
            {
                "path": str(cand),
                "exists": exists,
                "has_train_csv": has_train,
                "top_level": _top_level_entries(cand) if exists else [],
            }
        )
        if has_train:
            return cand, {
                "input_root": str(input_root),
                "input_top_level": _top_level_entries(input_root),
                "candidates_probed": probed,
                "resolved_root": str(cand),
                "resolution_method": "known_candidate",
            }

    # Fallback: shallow search for train.csv among top-level mounts only.
    if input_root.exists():
        for child in sorted(input_root.iterdir()):
            if not child.is_dir():
                continue
            train_csv = child / "train.csv"
            probed.append(
                {
                    "path": str(child),
                    "exists": True,
                    "has_train_csv": train_csv.exists(),
                    "top_level": _top_level_entries(child),
                }
            )
            if train_csv.exists() and (child / "train_series.csv").exists():
                return child, {
                    "input_root": str(input_root),
                    "input_top_level": _top_level_entries(input_root),
                    "candidates_probed": probed,
                    "resolved_root": str(child),
                    "resolution_method": "shallow_top_level_search",
                }
            nested = child / "competitions" / COMPETITION_SLUG
            if (nested / "train.csv").exists():
                return nested, {
                    "input_root": str(input_root),
                    "input_top_level": _top_level_entries(input_root),
                    "candidates_probed": probed,
                    "resolved_root": str(nested),
                    "resolution_method": "nested_competitions_search",
                }

    raise FileNotFoundError(
        "Could not locate competition CSVs under /kaggle/input without "
        "recursing into DICOM trees. Attach competition data source."
    )


def schema_report(df: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
    }


def _match_columns(columns: list[str], patterns: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for col in columns:
        low = col.strip().lower().replace(" ", "")
        for pat in patterns:
            if re.fullmatch(pat, low, flags=re.IGNORECASE):
                out.append(col)
                break
    return out


def _hint_columns(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for col in columns:
        low = re.sub(r"[^a-z0-9]+", "", col.lower())
        if any(h.replace("_", "") in low for h in hints):
            hits.append(col)
    return hits


def series_per_study_stats(series_df: pd.DataFrame) -> dict[str, Any]:
    counts = series_df.groupby("StudyInstanceUID").size()
    return {
        "n_studies_with_series": int(counts.shape[0]),
        "min": int(counts.min()) if len(counts) else None,
        "max": int(counts.max()) if len(counts) else None,
        "mean": float(counts.mean()) if len(counts) else None,
        "median": float(counts.median()) if len(counts) else None,
        "p25": float(counts.quantile(0.25)) if len(counts) else None,
        "p75": float(counts.quantile(0.75)) if len(counts) else None,
        "histogram": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
    }


def make_study_folds(study_ids: list[str], n_splits: int = N_SPLITS, seed: int = SEED) -> pd.DataFrame:
    ids = np.asarray(sorted(map(str, study_ids)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    ids_perm = ids[order]
    X = np.zeros(len(ids_perm))
    groups = ids_perm
    fold_map: dict[str, int] = {}
    splitter = GroupKFold(n_splits=n_splits)
    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X, groups=groups)):
        train_ids = set(ids_perm[train_idx])
        val_ids = set(ids_perm[val_idx])
        overlap = train_ids & val_ids
        if overlap:
            raise AssertionError(f"Fold {fold_i} StudyInstanceUID overlap: {sorted(overlap)[:5]}")
        for sid in val_ids:
            fold_map[sid] = fold_i
    if set(fold_map) != set(ids):
        missing = sorted(set(ids) - set(fold_map))
        raise AssertionError(f"Fold assignment incomplete; missing {len(missing)} studies")
    return pd.DataFrame(
        {"StudyInstanceUID": ids, "fold": [fold_map[s] for s in ids]}
    ).sort_values("StudyInstanceUID").reset_index(drop=True)


def fold_label_prevalence(train: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    gold_mask = train[list(TARGET_COLUMNS)].notna().all(axis=1)
    gold = train.loc[gold_mask].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    merged = gold.merge(folds, on="StudyInstanceUID", how="left", validate="one_to_one")
    if merged["fold"].isna().any():
        raise AssertionError("Gold study missing fold assignment")
    # Each gold study exactly once
    if merged["StudyInstanceUID"].duplicated().any():
        raise AssertionError("Duplicate gold StudyInstanceUID in fold merge")

    rows = []
    zero_pos = []
    zero_neg = []
    for fold_i in range(N_SPLITS):
        part = merged[merged["fold"] == fold_i]
        for col in TARGET_COLUMNS:
            n_pos = int((part[col] == 1).sum())
            n_neg = int((part[col] == 0).sum())
            rows.append(
                {
                    "fold": fold_i,
                    "label": col,
                    "n_studies": int(len(part)),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "prevalence": float(n_pos / len(part)) if len(part) else None,
                }
            )
            if n_pos == 0:
                zero_pos.append({"fold": fold_i, "label": col})
            if n_neg == 0:
                zero_neg.append({"fold": fold_i, "label": col})

    audits = {
        "n_gold_studies": int(len(gold)),
        "n_assigned": int(len(merged)),
        "each_gold_exactly_once": True,
        "zero_study_overlap_between_folds": True,
        "labels_with_zero_positives_in_a_fold": zero_pos,
        "labels_with_zero_negatives_in_a_fold": zero_neg,
        "fold_sizes": {
            str(i): int((merged["fold"] == i).sum()) for i in range(N_SPLITS)
        },
    }
    return pd.DataFrame(rows), audits


def decide_gate(
    *,
    mount_ok: bool,
    schemas_ok: bool,
    folds_ok: bool,
    report_train: bool,
    report_test: bool,
    patient_id_present: bool,
    laterality_present: bool,
    zero_class_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Gate decision for G1b — do not change baseline from speculation."""
    blockers: list[str] = []
    warnings: list[str] = []
    if not mount_ok:
        blockers.append("competition mount path unresolved")
    if not schemas_ok:
        blockers.append("required CSV schemas incomplete")
    if not folds_ok:
        blockers.append("StudyInstanceUID GroupKFold leakage/coverage failed")
    if report_test:
        blockers.append("Report unexpectedly present in test.csv (inference leak risk)")
    if not report_train:
        warnings.append("Report missing from train.csv — G3 offline labels unavailable")
    if patient_id_present is False:
        warnings.append("No PatientID in CSV metadata; StudyInstanceUID is grouping key")
    if laterality_present is False:
        warnings.append("No Laterality in CSV metadata")
    if zero_class_issues:
        warnings.append(
            f"{len(zero_class_issues)} fold×label cells have zero pos or zero neg "
            "(StratifiedGroupKFold may be needed later; baseline GroupKFold still feasible)"
        )

    if blockers:
        decision = "BLOCKED"
    elif zero_class_issues:
        # Fold map exists and is leakage-safe; rare-label imbalance is expected with n_gold≈58.
        decision = "PARTIALLY_READY"
    else:
        decision = "READY_FOR_BASELINE"

    return {
        "gate": "G1b",
        "experiment_id": "RSNA-DATA-002",
        "decision": decision,
        "blockers": blockers,
        "warnings": warnings,
        "baseline_unchanged": True,
        "notes": (
            "Baseline architecture must not be changed based on this metadata audit. "
            "DICOM header/pixel facts remain UNKNOWN."
        ),
    }


def main() -> None:
    work = _working_dir()
    root, path_meta = resolve_competition_root()

    runtime_paths = {
        **path_meta,
        "working_dir": str(work),
        "csv_paths": {name: str(root / name) for name in REQUIRED_CSVS},
        "train_series_dir_exists": (root / "train_series").exists(),
        "test_series_dir_exists": (root / "test_series").exists(),
        "dicom_enumeration": "SKIPPED_BY_POLICY",
        "dicom_decode": "SKIPPED_BY_POLICY",
    }
    (work / "runtime_paths.json").write_text(json.dumps(runtime_paths, indent=2))

    frames: dict[str, pd.DataFrame] = {}
    schemas: dict[str, Any] = {}
    for name in REQUIRED_CSVS:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        frames[name] = df
        schemas[name] = schema_report(df, name)

    train = frames["train.csv"]
    train_series = frames["train_series.csv"]
    test = frames["test.csv"]
    test_series = frames["test_series.csv"]
    sample = frames["sample_submission.csv"]

    all_cols = []
    for df in frames.values():
        all_cols.extend(map(str, df.columns))
    unique_cols = sorted(set(all_cols))

    patient_cols = _match_columns(unique_cols, PATIENT_ID_PATTERNS)
    laterality_cols = _match_columns(unique_cols, LATERALITY_PATTERNS)
    orientation_cols = _hint_columns(unique_cols, ORIENTATION_HINTS)
    sequence_cols = _hint_columns(unique_cols, SEQUENCE_HINTS)

    present_flags = {
        "StudyInstanceUID": "StudyInstanceUID" in unique_cols,
        "SeriesInstanceUID": "SeriesInstanceUID" in unique_cols,
        "SOPInstanceUID": "SOPInstanceUID" in unique_cols,
        "Report": "Report" in unique_cols,
        "sequence_or_series_descriptors": bool(sequence_cols),
        "orientation_related_metadata": bool(orientation_cols),
        "patient_identifier": bool(patient_cols),
        "laterality": bool(laterality_cols),
    }

    gold_mask = train[list(TARGET_COLUMNS)].notna().all(axis=1) if set(TARGET_COLUMNS).issubset(train.columns) else pd.Series(False, index=train.index)
    any_label_mask = train[list(TARGET_COLUMNS)].notna().any(axis=1) if set(TARGET_COLUMNS).issubset(train.columns) else pd.Series(False, index=train.index)

    # Series-type / plane distributions (CSV metadata only).
    plane_counts = {}
    if "Anatomical_Plane" in train_series.columns:
        plane_counts = {
            str(k): int(v) for k, v in train_series["Anatomical_Plane"].value_counts(dropna=False).items()
        }
    combo = None
    group_cols = [c for c in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression") if c in train_series.columns]
    if group_cols:
        combo = (
            train_series.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="n_series")
            .to_dict(orient="records")
        )
        studies_by_plane = (
            train_series.groupby("Anatomical_Plane")["StudyInstanceUID"]
            .nunique()
            .to_dict()
            if "Anatomical_Plane" in train_series.columns
            else {}
        )
    else:
        studies_by_plane = {}

    series_summary = {
        "n_train_series": int(len(train_series)),
        "n_test_series_example": int(len(test_series)),
        "n_unique_train_series_uids": int(train_series["SeriesInstanceUID"].nunique())
        if "SeriesInstanceUID" in train_series.columns
        else None,
        "series_per_study": series_per_study_stats(train_series)
        if "StudyInstanceUID" in train_series.columns
        else None,
        "anatomical_plane_series_counts": plane_counts,
        "studies_per_anatomical_plane": {str(k): int(v) for k, v in studies_by_plane.items()},
        "plane_fluid_fatsat_counts": combo,
        "columns": schemas["train_series.csv"]["columns"],
        "schema": schemas["train_series.csv"],
        "test_series_schema": schemas["test_series.csv"],
    }
    (work / "series_summary.json").write_text(json.dumps(series_summary, indent=2, default=str))

    label_rows = []
    if set(TARGET_COLUMNS).issubset(train.columns):
        for col in TARGET_COLUMNS:
            s = train[col]
            label_rows.append(
                {
                    "label": col,
                    "n_non_null": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "n_neg": int((s == 0).sum()),
                    "n_null": int(s.isna().sum()),
                    "prevalence_among_non_null": float((s == 1).sum() / s.notna().sum())
                    if s.notna().any()
                    else None,
                }
            )
    label_coverage = pd.DataFrame(label_rows)
    label_coverage.to_csv(work / "label_coverage.csv", index=False)

    # Deterministic StudyInstanceUID GroupKFold on ALL train studies; gold subset audited.
    all_study_ids = train["StudyInstanceUID"].astype(str).drop_duplicates().tolist()
    folds = make_study_folds(all_study_ids, n_splits=N_SPLITS, seed=SEED)
    gold_ids = set(train.loc[gold_mask, "StudyInstanceUID"].astype(str))
    folds["has_gold_labels"] = folds["StudyInstanceUID"].isin(gold_ids).astype(int)
    folds.to_csv(work / "study_folds.csv", index=False)

    fold_prev, fold_audits = fold_label_prevalence(train, folds[["StudyInstanceUID", "fold"]])
    fold_prev.to_csv(work / "fold_label_prevalence.csv", index=False)

    # Cross-fold overlap check on full study map
    for i in range(N_SPLITS):
        for j in range(i + 1, N_SPLITS):
            a = set(folds.loc[folds["fold"] == i, "StudyInstanceUID"])
            b = set(folds.loc[folds["fold"] == j, "StudyInstanceUID"])
            if a & b:
                raise AssertionError(f"Study overlap between folds {i} and {j}")

    report_in_train = "Report" in train.columns
    report_in_test = "Report" in test.columns
    n_report = int(train["Report"].notna().sum()) if report_in_train else 0

    train_test_diff = {
        "train_only_columns": sorted(set(train.columns) - set(test.columns)),
        "test_only_columns": sorted(set(test.columns) - set(train.columns)),
        "shared_columns": sorted(set(train.columns) & set(test.columns)),
        "train_series_only_columns": sorted(set(train_series.columns) - set(test_series.columns)),
        "test_series_only_columns": sorted(set(test_series.columns) - set(train_series.columns)),
        "sample_vs_train_label_columns_match": sorted(sample.columns) == sorted(["StudyInstanceUID", *TARGET_COLUMNS])
        or list(sample.columns) == ["StudyInstanceUID", *TARGET_COLUMNS],
    }

    study_summary = {
        "n_train_studies": int(len(train)),
        "n_unique_train_study_uids": int(train["StudyInstanceUID"].nunique()),
        "n_test_studies_example": int(len(test)),
        "n_sample_submission_rows": int(len(sample)),
        "n_gold_label_rows": int(gold_mask.sum()),
        "n_any_label_rows": int(any_label_mask.sum()),
        "n_report_non_null": n_report,
        "report_in_train": report_in_train,
        "report_in_test": report_in_test,
        "target_columns_present": [c for c in TARGET_COLUMNS if c in train.columns],
        "target_columns_missing": [c for c in TARGET_COLUMNS if c not in train.columns],
        "missingness_train": schemas["train.csv"]["null_counts"],
        "patient_id_columns": patient_cols,
        "laterality_columns": laterality_cols,
        "orientation_hint_columns": orientation_cols,
        "sequence_hint_columns": sequence_cols,
        "field_presence": present_flags,
        "train_test_schema_diff": train_test_diff,
        "schemas": schemas,
        "fold_audits": fold_audits,
        "n_splits": N_SPLITS,
        "seed": SEED,
    }
    (work / "study_summary.json").write_text(json.dumps(study_summary, indent=2, default=str))

    schemas_ok = (
        present_flags["StudyInstanceUID"]
        and present_flags["SeriesInstanceUID"]
        and set(TARGET_COLUMNS).issubset(train.columns)
        and "StudyInstanceUID" in test.columns
        and not report_in_test
    )
    folds_ok = fold_audits["each_gold_exactly_once"] and fold_audits["zero_study_overlap_between_folds"]
    zero_issues = (
        fold_audits["labels_with_zero_positives_in_a_fold"]
        + fold_audits["labels_with_zero_negatives_in_a_fold"]
    )
    gate = decide_gate(
        mount_ok=True,
        schemas_ok=schemas_ok,
        folds_ok=folds_ok,
        report_train=report_in_train,
        report_test=report_in_test,
        patient_id_present=bool(patient_cols),
        laterality_present=bool(laterality_cols),
        zero_class_issues=zero_issues,
    )
    (work / "GATE_DECISION.json").write_text(json.dumps(gate, indent=2))

    print("RESOLVED_ROOT", root)
    print("N_TRAIN_STUDIES", study_summary["n_train_studies"])
    print("N_TRAIN_SERIES", series_summary["n_train_series"])
    print("N_GOLD", study_summary["n_gold_label_rows"])
    print("REPORT_TRAIN", report_in_train, "REPORT_TEST", report_in_test)
    print("PATIENT_ID_COLS", patient_cols)
    print("LATERALITY_COLS", laterality_cols)
    print("GATE", gate["decision"])
    print("Wrote artifacts to", work)


if __name__ == "__main__":
    main()
