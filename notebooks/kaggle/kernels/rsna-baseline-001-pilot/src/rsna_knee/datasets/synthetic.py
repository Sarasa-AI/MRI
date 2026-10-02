"""Synthetic fixture builders for local metadata-only tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS


def build_fixture_frames(n_studies: int = 20, n_test: int = 3, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Create tiny synthetic CSVs matching competition schemas."""
    rng_labels = pd.Series(range(n_studies))
    study_ids = [f"study_{i:04d}" for i in range(n_studies)]

    # First 8 studies are "gold" labeled; rest have NaN labels + report text.
    train_rows = []
    for i, sid in enumerate(study_ids):
        row = {STUDY_ID_COL: sid, "Report": f"Synthetic report for {sid}."}
        if i < 8:
            for j, col in enumerate(TARGET_COLUMNS):
                # Deterministic binary labels with both classes for ACL/MCL.
                row[col] = float((i + j) % 2)
        else:
            for col in TARGET_COLUMNS:
                row[col] = None
        train_rows.append(row)
    train = pd.DataFrame(train_rows)

    series_rows = []
    planes = ["Sagittal", "Coronal", "Axial"]
    for i, sid in enumerate(study_ids):
        for p, plane in enumerate(planes):
            series_rows.append(
                {
                    STUDY_ID_COL: sid,
                    "SeriesInstanceUID": f"{sid}_series_{p}",
                    "Fluid_Sensitive": int(p != 2),
                    "Fat_Suppression": int(p == 0),
                    "Anatomical_Plane": plane,
                }
            )
    train_series = pd.DataFrame(series_rows)

    test_ids = [f"test_{i:04d}" for i in range(n_test)]
    test = pd.DataFrame({STUDY_ID_COL: test_ids})
    test_series = pd.DataFrame(
        [
            {
                STUDY_ID_COL: tid,
                "SeriesInstanceUID": f"{tid}_series_0",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Sagittal",
            }
            for tid in test_ids
        ]
    )
    sample = test[[STUDY_ID_COL]].copy()
    for col in TARGET_COLUMNS:
        sample[col] = 0.5

    _ = rng_labels  # reserved for future stochastic fixtures
    return {
        "train": train,
        "train_series": train_series,
        "test": test,
        "test_series": test_series,
        "sample_submission": sample,
    }


def write_fixtures(directory: str | Path, **kwargs) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = build_fixture_frames(**kwargs)
    frames["train"].to_csv(directory / "train.csv", index=False)
    frames["train_series"].to_csv(directory / "train_series.csv", index=False)
    frames["test"].to_csv(directory / "test.csv", index=False)
    frames["test_series"].to_csv(directory / "test_series.csv", index=False)
    frames["sample_submission"].to_csv(directory / "sample_submission.csv", index=False)
    return directory
