"""Submission validator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.datasets.synthetic import build_fixture_frames
from rsna_knee.inference.predict import build_uniform_submission, write_submission
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.submission.validate import validate_submission


def _valid_submission() -> pd.DataFrame:
    frames = build_fixture_frames()
    return build_uniform_submission(frames["test"])


def test_valid_submission_passes(tmp_path):
    df = _valid_submission()
    result = validate_submission(df, expected_ids=df[STUDY_ID_COL].tolist())
    assert result.ok
    path = write_submission(df, tmp_path / "submission.csv", expected_ids=df[STUDY_ID_COL].tolist())
    assert path.exists()


def test_missing_column_fails():
    df = _valid_submission().drop(columns=[TARGET_COLUMNS[0]])
    result = validate_submission(df)
    assert not result.ok
    assert any("Missing required columns" in e for e in result.errors)


def test_nan_and_inf_fail():
    df = _valid_submission()
    df.loc[0, TARGET_COLUMNS[0]] = np.nan
    assert not validate_submission(df).ok
    df = _valid_submission()
    df.loc[0, TARGET_COLUMNS[1]] = np.inf
    assert not validate_submission(df).ok


def test_out_of_range_fails():
    df = _valid_submission()
    df.loc[0, TARGET_COLUMNS[2]] = 1.5
    assert not validate_submission(df).ok


def test_duplicate_ids_fail():
    df = _valid_submission()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = validate_submission(df)
    assert not result.ok
    assert any("duplicate" in e.lower() for e in result.errors)


def test_row_count_mismatch():
    df = _valid_submission()
    result = validate_submission(df, expected_row_count=999)
    assert not result.ok
