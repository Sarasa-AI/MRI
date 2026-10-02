"""Submission CSV validation for RSNA Knee 2026."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from rsna_knee.labels.targets import (
    N_TARGETS,
    STUDY_ID_COL,
    SUBMISSION_REQUIRED_COLUMNS,
    TARGET_COLUMNS,
)


@dataclass
class SubmissionValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def validate_submission(
    df: pd.DataFrame | str | Path,
    *,
    expected_ids: Sequence[str] | None = None,
    expected_row_count: int | None = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> SubmissionValidationResult:
    """Validate a submission against competition schema constraints.

    Checks:
      - required columns (StudyInstanceUID + 12 targets)
      - no unexpected missing required columns
      - prediction range [min_value, max_value]
      - no NaN / infinite values
      - row count (optional expected)
      - duplicate StudyInstanceUID
      - optional exact ID set match vs test.csv
    """
    errors: list[str] = []
    warnings: list[str] = []

    if isinstance(df, (str, Path)):
        try:
            df = pd.read_csv(df)
        except Exception as exc:  # noqa: BLE001
            return SubmissionValidationResult(ok=False, errors=[f"Failed to read CSV: {exc}"])

    # Column presence / order awareness
    missing = [c for c in SUBMISSION_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in SUBMISSION_REQUIRED_COLUMNS]
    if extra:
        warnings.append(f"Extra columns present (ignored by scorer?): {extra}")

    if missing:
        return SubmissionValidationResult(ok=False, errors=errors, warnings=warnings, n_rows=len(df))

    n_rows = len(df)
    if expected_row_count is not None and n_rows != expected_row_count:
        errors.append(f"Row count {n_rows} != expected {expected_row_count}")

    # Duplicate IDs
    ids = df[STUDY_ID_COL].astype(str)
    n_dupes = int(ids.duplicated().sum())
    if n_dupes:
        errors.append(f"Found {n_dupes} duplicate StudyInstanceUID values")

    if expected_ids is not None:
        expected_set = set(map(str, expected_ids))
        actual_set = set(ids.tolist())
        missing_ids = expected_set - actual_set
        unexpected_ids = actual_set - expected_set
        if missing_ids:
            errors.append(f"Missing {len(missing_ids)} StudyInstanceUID(s) vs expected set")
        if unexpected_ids:
            errors.append(f"Unexpected {len(unexpected_ids)} StudyInstanceUID(s) vs expected set")
        if expected_row_count is None and len(expected_set) != n_rows and not n_dupes:
            # If sets match but counts differ only via dupes (already flagged).
            pass

    # Value checks for targets
    for col in TARGET_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            # Distinguish original NaN vs coercion failure
            n_nan = int(series.isna().sum())
            errors.append(f"Column '{col}' has {n_nan} NaN/non-numeric values")
            continue
        values = series.to_numpy(dtype=float)
        if np.isinf(values).any():
            errors.append(f"Column '{col}' contains infinite values")
        if (values < min_value).any() or (values > max_value).any():
            errors.append(
                f"Column '{col}' has values outside [{min_value}, {max_value}]"
            )

    if len(TARGET_COLUMNS) != N_TARGETS:
        errors.append("Internal target count mismatch")

    return SubmissionValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_rows=n_rows,
    )
