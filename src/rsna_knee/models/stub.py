"""Intentionally trivial models for platform smoke tests only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.labels.targets import N_TARGETS, STUDY_ID_COL, TARGET_COLUMNS


class ConstantPriorModel:
    """Predicts each label's training prevalence (fallback 0.5).

    Not a competition model — exists so the experiment platform can be
    exercised end-to-end without DICOMs or a neural net.
    """

    def __init__(self) -> None:
        self.priors_: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> "ConstantPriorModel":
        priors = []
        for col in TARGET_COLUMNS:
            series = train_df[col].astype(float)
            valid = series.dropna()
            priors.append(float(valid.mean()) if len(valid) else 0.5)
        self.priors_ = np.asarray(priors, dtype=float)
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.priors_ is None:
            raise RuntimeError("Model is not fitted")
        return np.tile(self.priors_, (n, 1))

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = self.predict(len(df))
        out = df[[STUDY_ID_COL]].copy()
        for i, col in enumerate(TARGET_COLUMNS):
            out[col] = scores[:, i]
        return out


class UniformHalfModel:
    """Predicts 0.5 for every label — sample_submission equivalent."""

    def fit(self, train_df: pd.DataFrame | None = None) -> "UniformHalfModel":
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full((n, N_TARGETS), 0.5, dtype=float)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[[STUDY_ID_COL]].copy()
        for col in TARGET_COLUMNS:
            out[col] = 0.5
        return out
