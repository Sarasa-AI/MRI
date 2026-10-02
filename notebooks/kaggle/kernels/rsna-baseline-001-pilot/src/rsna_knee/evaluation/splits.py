"""Leakage-safe StudyInstanceUID-level CV splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS


@dataclass(frozen=True)
class FoldAssignment:
    fold: int
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]

    def assert_no_leakage(self) -> None:
        overlap = set(self.train_ids) & set(self.val_ids)
        if overlap:
            raise AssertionError(
                f"Fold {self.fold} leaks StudyInstanceUID(s): {sorted(overlap)[:5]}"
            )


def _study_groups(df: pd.DataFrame, group_col: str = STUDY_ID_COL) -> np.ndarray:
    if group_col not in df.columns:
        raise KeyError(f"Missing grouping column '{group_col}'")
    return df[group_col].astype(str).to_numpy()


def assert_no_group_overlap(
    train_ids: set[str] | list[str] | np.ndarray,
    val_ids: set[str] | list[str] | np.ndarray,
    *,
    context: str = "",
) -> None:
    overlap = set(map(str, train_ids)) & set(map(str, val_ids))
    if overlap:
        sample = sorted(overlap)[:10]
        raise AssertionError(
            f"Group leakage detected{(' in ' + context) if context else ''}: {sample}"
        )


def make_study_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = STUDY_ID_COL,
    stratify_col: str | None = None,
) -> list[FoldAssignment]:
    """Create GroupKFold (or StratifiedGroupKFold) assignments by study UID.

    PatientID is not present in official competition CSVs. StudyInstanceUID is
    the finest leakage-safe group available in released metadata.
    """
    if df[group_col].duplicated().any():
        # Expect one row per study for fold assignment. Series-level frames
        # must be collapsed first.
        raise ValueError(
            f"Fold assignment requires unique '{group_col}' rows. "
            "Collapse series-level data to study level first."
        )

    groups = _study_groups(df, group_col)
    X = np.zeros(len(df))
    indices = np.arange(len(df))
    study_ids = groups

    if stratify_col is not None:
        y = df[stratify_col].to_numpy()
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = splitter.split(X, y, groups)
    else:
        # GroupKFold has no shuffle; we permute studies deterministically.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
        X = X[order]
        groups_perm = groups[order]
        indices = indices[order]
        study_ids = groups_perm
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, groups=groups_perm)

    folds: list[FoldAssignment] = []
    for fold_i, (train_pos, val_pos) in enumerate(splits):
        train_idx = indices[train_pos]
        val_idx = indices[val_pos]
        train_ids = tuple(study_ids[train_pos])
        val_ids = tuple(study_ids[val_pos])
        assignment = FoldAssignment(fold=fold_i, train_ids=train_ids, val_ids=val_ids)
        assignment.assert_no_leakage()
        assert_no_group_overlap(train_ids, val_ids, context=f"fold {fold_i}")
        # Also verify via original dataframe index mapping
        assert_no_group_overlap(
            df.iloc[train_idx][group_col].astype(str),
            df.iloc[val_idx][group_col].astype(str),
            context=f"fold {fold_i} index map",
        )
        folds.append(assignment)
    return folds


def iter_fold_frames(
    df: pd.DataFrame,
    folds: list[FoldAssignment],
    *,
    group_col: str = STUDY_ID_COL,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train_df, val_df) with leakage assertions."""
    for fold in folds:
        fold.assert_no_leakage()
        train_df = df[df[group_col].astype(str).isin(fold.train_ids)].copy()
        val_df = df[df[group_col].astype(str).isin(fold.val_ids)].copy()
        assert_no_group_overlap(
            train_df[group_col],
            val_df[group_col],
            context=f"iter fold {fold.fold}",
        )
        yield fold.fold, train_df, val_df


def has_any_gold_label(df: pd.DataFrame) -> pd.Series:
    """True where at least one of the 12 targets is non-null."""
    return df[list(TARGET_COLUMNS)].notna().any(axis=1)


def gold_label_mask(df: pd.DataFrame) -> pd.Series:
    """True where all 12 targets are non-null (official gold rows)."""
    return df[list(TARGET_COLUMNS)].notna().all(axis=1)
