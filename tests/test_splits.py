"""Leakage-safe StudyInstanceUID split tests."""

from __future__ import annotations

import pytest

from rsna_knee.datasets.synthetic import build_fixture_frames
from rsna_knee.evaluation.leakage import audit_fold_leakage
from rsna_knee.evaluation.splits import (
    assert_no_group_overlap,
    gold_label_mask,
    iter_fold_frames,
    make_study_folds,
)
from rsna_knee.labels.targets import STUDY_ID_COL


def test_study_folds_zero_overlap():
    frames = build_fixture_frames(n_studies=20)
    train = frames["train"]
    gold = train.loc[gold_label_mask(train)].copy()
    folds = make_study_folds(gold, n_splits=4, seed=42)
    assert len(folds) == 4
    for fold in folds:
        fold.assert_no_leakage()
        assert len(fold.train_ids) + len(fold.val_ids) == len(gold)


def test_iter_fold_frames_no_leakage():
    frames = build_fixture_frames(n_studies=20)
    train = frames["train"]
    gold = train.loc[gold_label_mask(train)].copy()
    folds = make_study_folds(gold, n_splits=4, seed=7)
    for fold_i, tr, va in iter_fold_frames(gold, folds):
        assert_no_group_overlap(tr[STUDY_ID_COL], va[STUDY_ID_COL], context=str(fold_i))
        stats = audit_fold_leakage(tr, va)
        assert stats["study_overlap"] == 0


def test_rejects_duplicate_study_rows_for_folding():
    frames = build_fixture_frames(n_studies=8)
    series = frames["train_series"]
    with pytest.raises(ValueError, match="unique"):
        make_study_folds(series, n_splits=2, seed=0)
