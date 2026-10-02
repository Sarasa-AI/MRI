"""Label supervision strategies A–E for training-time report use only.

A. Gold only
B. Report-derived hard labels
C. Report-derived soft probabilities
D. Confidence-weighted soft labels
E. Gold + high-confidence silver labels

All strategies produce study-level supervision frames consumed by the imaging
OOF loop. Inference never reads Report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rsna_knee.evaluation.splits import gold_label_mask
from rsna_knee.labels.report_parser import (
    STATE_MISSING,
    STATE_NEGATIVE,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    conf_matrix,
    extract_reports_frame,
    soft_matrix,
    state_matrix,
)
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS

StrategyName = Literal["A_gold", "B_hard", "C_soft", "D_conf_weighted", "E_gold_plus_silver"]

STRATEGY_NAMES: tuple[StrategyName, ...] = (
    "A_gold",
    "B_hard",
    "C_soft",
    "D_conf_weighted",
    "E_gold_plus_silver",
)


@dataclass(frozen=True)
class StrategySpec:
    name: StrategyName
    description: str
    hard_threshold: float = 0.55
    min_confidence: float = 0.55
    missing_policy: str = "mask"  # mask | prior


@dataclass
class SupervisionBundle:
    """Labels + per-label loss weights for one strategy."""

    strategy: StrategyName
    studies: pd.DataFrame  # StudyInstanceUID + target cols (float, NaN=masked)
    weights: pd.DataFrame  # same shape; 0 = ignore in loss
    meta: dict


DEFAULT_SPECS: dict[StrategyName, StrategySpec] = {
    "A_gold": StrategySpec("A_gold", "Gold-only hard labels (58 studies)"),
    "B_hard": StrategySpec(
        "B_hard",
        "Report hard labels: P→1, N→0, U/M masked",
        hard_threshold=0.55,
        min_confidence=0.0,
    ),
    "C_soft": StrategySpec(
        "C_soft",
        "Report soft probabilities; M kept at prior with tiny weight",
    ),
    "D_conf_weighted": StrategySpec(
        "D_conf_weighted",
        "Report soft labels weighted by extractor confidence",
    ),
    "E_gold_plus_silver": StrategySpec(
        "E_gold_plus_silver",
        "Gold where available; else high-confidence report hard labels",
        hard_threshold=0.55,
        min_confidence=0.60,
    ),
}


def _empty_like(ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame({STUDY_ID_COL: ids.astype(str).values})
    weights = pd.DataFrame({STUDY_ID_COL: ids.astype(str).values})
    for col in TARGET_COLUMNS:
        labels[col] = np.nan
        weights[col] = 0.0
    return labels, weights


def build_supervision(
    train_df: pd.DataFrame,
    *,
    strategy: StrategyName,
    extracted: pd.DataFrame | None = None,
    spec: StrategySpec | None = None,
) -> SupervisionBundle:
    """Build label + weight frames for a named strategy."""
    spec = spec or DEFAULT_SPECS[strategy]
    if extracted is None:
        extracted = extract_reports_frame(train_df)

    # Align extracted to train order
    merged = train_df[[STUDY_ID_COL]].copy()
    merged[STUDY_ID_COL] = merged[STUDY_ID_COL].astype(str)
    extracted = extracted.copy()
    extracted[STUDY_ID_COL] = extracted[STUDY_ID_COL].astype(str)
    extracted = merged.merge(extracted, on=STUDY_ID_COL, how="left", validate="one_to_one")

    soft = soft_matrix(extracted)
    conf = conf_matrix(extracted)
    state = state_matrix(extracted)
    gold_mask = gold_label_mask(train_df).to_numpy()
    gold_y = train_df[list(TARGET_COLUMNS)].to_numpy(dtype=float)

    labels, weights = _empty_like(train_df[STUDY_ID_COL])
    y = labels[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    w = weights[list(TARGET_COLUMNS)].to_numpy(dtype=float)

    if strategy == "A_gold":
        y[gold_mask] = gold_y[gold_mask]
        w[gold_mask] = 1.0
        keep = gold_mask
    elif strategy == "B_hard":
        for i in range(len(train_df)):
            for j in range(len(TARGET_COLUMNS)):
                st = state[i, j]
                if st == STATE_POSITIVE:
                    y[i, j] = 1.0
                    w[i, j] = 1.0
                elif st == STATE_NEGATIVE:
                    y[i, j] = 0.0
                    w[i, j] = 1.0
                else:
                    y[i, j] = np.nan
                    w[i, j] = 0.0
        # Prefer gold overwrite when available (still strategy B corpus-wide hard,
        # but never contradict expert labels on the 58).
        y[gold_mask] = gold_y[gold_mask]
        w[gold_mask] = 1.0
        keep = (w > 0).any(axis=1)
    elif strategy == "C_soft":
        y[:] = soft
        w[:] = np.where(state == STATE_MISSING, 0.15, 1.0)
        y[gold_mask] = gold_y[gold_mask]
        w[gold_mask] = 1.0
        keep = np.ones(len(train_df), dtype=bool)
    elif strategy == "D_conf_weighted":
        y[:] = soft
        w[:] = conf
        w = np.where(state == STATE_MISSING, conf * 0.25, w)
        y[gold_mask] = gold_y[gold_mask]
        w[gold_mask] = 1.0
        keep = np.ones(len(train_df), dtype=bool)
    elif strategy == "E_gold_plus_silver":
        for i in range(len(train_df)):
            if gold_mask[i]:
                y[i] = gold_y[i]
                w[i] = 1.0
                continue
            for j in range(len(TARGET_COLUMNS)):
                st = state[i, j]
                c = float(conf[i, j])
                if st == STATE_POSITIVE and c >= spec.min_confidence:
                    y[i, j] = 1.0
                    w[i, j] = float(c)
                elif st == STATE_NEGATIVE and c >= spec.min_confidence:
                    y[i, j] = 0.0
                    w[i, j] = float(c)
                else:
                    y[i, j] = np.nan
                    w[i, j] = 0.0
        keep = (w > 0).any(axis=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for j, col in enumerate(TARGET_COLUMNS):
        labels[col] = y[:, j]
        weights[col] = w[:, j]

    # Drop studies with no supervised targets for this strategy.
    labels = labels.loc[keep].reset_index(drop=True)
    weights = weights.loc[keep].reset_index(drop=True)

    n_gold_in = int(labels[STUDY_ID_COL].isin(train_df.loc[gold_mask, STUDY_ID_COL].astype(str)).sum())
    meta = {
        "strategy": strategy,
        "description": spec.description,
        "n_studies": int(len(labels)),
        "n_gold_included": n_gold_in,
        "mean_weight": float(np.nanmean(weights[list(TARGET_COLUMNS)].to_numpy())),
        "frac_masked_cells": float(np.isnan(labels[list(TARGET_COLUMNS)].to_numpy()).mean()),
        "hard_threshold": spec.hard_threshold,
        "min_confidence": spec.min_confidence,
    }
    return SupervisionBundle(strategy=strategy, studies=labels, weights=weights, meta=meta)


def build_all_strategies(
    train_df: pd.DataFrame,
    *,
    extracted: pd.DataFrame | None = None,
) -> dict[StrategyName, SupervisionBundle]:
    if extracted is None:
        extracted = extract_reports_frame(train_df)
    return {
        name: build_supervision(train_df, strategy=name, extracted=extracted)
        for name in STRATEGY_NAMES
    }


def report_labels_as_predictions(extracted: pd.DataFrame) -> pd.DataFrame:
    """Soft report labels shaped as an OOF-style prediction frame."""
    out = extracted[[STUDY_ID_COL]].copy()
    soft = soft_matrix(extracted)
    for j, col in enumerate(TARGET_COLUMNS):
        out[col] = soft[:, j]
    return out


def hard_report_predictions(extracted: pd.DataFrame) -> pd.DataFrame:
    """Hard 0/1/NaN from states — NaN for U/M so metrics skip undefined."""
    out = extracted[[STUDY_ID_COL]].copy()
    state = state_matrix(extracted)
    for j, col in enumerate(TARGET_COLUMNS):
        vals = np.full(len(extracted), np.nan, dtype=float)
        vals[state[:, j] == STATE_POSITIVE] = 1.0
        vals[state[:, j] == STATE_NEGATIVE] = 0.0
        # Uncertain → soft mid for ranking metrics
        vals[state[:, j] == STATE_UNCERTAIN] = 0.5
        out[col] = vals
    return out
