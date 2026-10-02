"""Post-hoc analysis helpers for baseline experiment reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.labels.targets import TARGET_COLUMNS


def label_correlation(y_score: np.ndarray) -> pd.DataFrame:
    """Pearson correlation across label prediction columns."""
    df = pd.DataFrame(y_score, columns=list(TARGET_COLUMNS))
    return df.corr(method="pearson")


def fold_stability(fold_macros: list[float]) -> dict[str, float]:
    arr = np.asarray([m for m in fold_macros if m == m], dtype=float)  # drop NaN
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
    }


def strongest_weakest_labels(per_label_auc: dict[str, float], k: int = 3) -> dict[str, Any]:
    ranked = sorted(
        ((name, auc) for name, auc in per_label_auc.items() if auc == auc),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "strongest": ranked[:k],
        "weakest": list(reversed(ranked[-k:])) if ranked else [],
    }


def infer_bottleneck(
    *,
    n_gold: int,
    fold_std: float,
    max_abs_pred_corr: float,
    mean_train_auc: float | None = None,
    oof_macro: float | None = None,
) -> str:
    """Heuristic bottleneck call for the experiment report.

    Priority order reflects competition structure: label sparsity dominates
    until pseudo-labels are introduced.
    """
    if n_gold < 100:
        return (
            "labels — only ~58 gold studies; architecture/aug gains are "
            "secondary until report-derived training labels (training-time only) "
            "expand supervision"
        )
    if fold_std > 0.05:
        return "validation — fold macro-AUC unstable; check stratification / sample size"
    if max_abs_pred_corr > 0.95:
        return (
            "architecture/head — predictions nearly collinear across labels; "
            "model may be collapsing to a shared abnormality prior"
        )
    if (
        mean_train_auc is not None
        and oof_macro is not None
        and mean_train_auc - oof_macro > 0.15
    ):
        return "augmentation/regularization — large train/OOF gap suggests overfitting"
    return "data selection — revisit series/plane policy and input geometry next"


def leakage_checklist(
    *,
    n_fold_overlaps: int,
    report_used_at_inference: bool,
    patient_id_available: bool,
) -> dict[str, Any]:
    return {
        "study_uid_fold_overlaps": n_fold_overlaps,
        "report_used_at_inference": report_used_at_inference,
        "patient_id_available": patient_id_available,
        "notes": (
            "PatientID absent from official CSVs — StudyInstanceUID grouping is "
            "the strongest leakage guard available in public metadata."
        ),
        "leakage_signs": (
            []
            if n_fold_overlaps == 0 and not report_used_at_inference
            else ["see flags"]
        ),
    }
