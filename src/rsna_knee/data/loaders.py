"""Metadata CSV loaders (no DICOM decode)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_knee.labels.targets import (
    REPORT_COL,
    SERIES_META_COLUMNS,
    STUDY_ID_COL,
    TARGET_COLUMNS,
)
from rsna_knee.runtime.adapter import DataPaths


def load_train_metadata(path: str | Path) -> pd.DataFrame:
    """Load train.csv (study-level labels + optional Report)."""
    df = pd.read_csv(path)
    required = [STUDY_ID_COL, *TARGET_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    df[STUDY_ID_COL] = df[STUDY_ID_COL].astype(str)
    return df


def load_series_metadata(path: str | Path) -> pd.DataFrame:
    """Load train_series.csv or test_series.csv."""
    df = pd.read_csv(path)
    missing = [c for c in SERIES_META_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"series CSV missing columns: {missing}")
    df[STUDY_ID_COL] = df[STUDY_ID_COL].astype(str)
    return df


def load_test_metadata(path: str | Path) -> pd.DataFrame:
    """Load test.csv — StudyInstanceUID only; must not require Report."""
    df = pd.read_csv(path)
    if STUDY_ID_COL not in df.columns:
        raise ValueError(f"test.csv missing '{STUDY_ID_COL}'")
    if REPORT_COL in df.columns:
        # Example fixtures may omit Report; if present at score-time it would
        # still be invalid to depend on it. Keep the column but callers must
        # not use it for inference features.
        pass
    df[STUDY_ID_COL] = df[STUDY_ID_COL].astype(str)
    return df


def load_all_metadata(paths: DataPaths) -> dict[str, pd.DataFrame]:
    """Load all competition CSVs from a resolved DataPaths object."""
    return {
        "train": load_train_metadata(paths.train_csv),
        "train_series": load_series_metadata(paths.train_series_csv),
        "test": load_test_metadata(paths.test_csv),
        "test_series": load_series_metadata(paths.test_series_csv),
        "sample_submission": pd.read_csv(paths.sample_submission_csv),
    }


def study_label_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Return (id frame, float label matrix with NaN for missing)."""
    ids = df[[STUDY_ID_COL]].copy()
    y = df[list(TARGET_COLUMNS)].astype(float).to_numpy()
    return ids, y
