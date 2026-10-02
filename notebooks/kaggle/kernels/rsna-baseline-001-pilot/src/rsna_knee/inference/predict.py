"""Submission construction helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.models.stub import UniformHalfModel
from rsna_knee.submission.validate import validate_submission


def build_uniform_submission(test_df: pd.DataFrame) -> pd.DataFrame:
    """Build a valid all-0.5 submission from test StudyInstanceUIDs."""
    model = UniformHalfModel().fit()
    return model.predict_df(test_df)


def write_submission(
    df: pd.DataFrame,
    path: str | Path,
    *,
    expected_ids: list[str] | None = None,
) -> Path:
    """Validate and write submission.csv."""
    path = Path(path)
    result = validate_submission(df, expected_ids=expected_ids)
    if not result.ok:
        raise ValueError(f"Invalid submission: {result.errors}")
    # Enforce column order
    ordered = df[[STUDY_ID_COL, *TARGET_COLUMNS]].copy()
    ordered.to_csv(path, index=False)
    return path
