"""Leakage audit utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from rsna_knee.evaluation.splits import assert_no_group_overlap
from rsna_knee.labels.targets import SERIES_ID_COL, STUDY_ID_COL


def audit_fold_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    study_col: str = STUDY_ID_COL,
    series_col: str = SERIES_ID_COL,
) -> dict[str, int]:
    """Assert zero study (and series, if present) overlap; return counts."""
    assert_no_group_overlap(
        train_df[study_col],
        val_df[study_col],
        context="study audit",
    )
    result = {
        "train_studies": int(train_df[study_col].nunique()),
        "val_studies": int(val_df[study_col].nunique()),
        "study_overlap": 0,
    }
    if series_col in train_df.columns and series_col in val_df.columns:
        assert_no_group_overlap(
            train_df[series_col],
            val_df[series_col],
            context="series audit",
        )
        result["train_series"] = int(train_df[series_col].nunique())
        result["val_series"] = int(val_df[series_col].nunique())
        result["series_overlap"] = 0
    return result


def assert_unique_study_ids(ids: Iterable[str]) -> None:
    values = list(map(str, ids))
    if len(values) != len(set(values)):
        raise AssertionError("Duplicate StudyInstanceUID values detected")
