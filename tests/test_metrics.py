"""Tests for competition metrics."""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.evaluation.metrics import macro_roc_auc, per_label_roc_auc
from rsna_knee.labels.targets import N_TARGETS, TARGET_COLUMNS


def test_macro_auc_perfect():
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
    # Pad to 12 labels with alternating patterns that remain separable.
    y_true_full = np.zeros((4, N_TARGETS), dtype=float)
    y_score_full = np.zeros((4, N_TARGETS), dtype=float)
    for i in range(N_TARGETS):
        y_true_full[:, i] = (np.arange(4) + i) % 2
        y_score_full[:, i] = y_true_full[:, i]
    result = macro_roc_auc(y_true_full, y_score_full)
    assert result.macro_auc == pytest.approx(1.0)
    assert len(result.evaluated_labels) == N_TARGETS
    assert result.skipped_labels == ()


def test_per_label_skips_undefined():
    y_true = np.zeros((4, N_TARGETS), dtype=float)
    y_score = np.linspace(0, 1, 4 * N_TARGETS).reshape(4, N_TARGETS)
    # All zeros → undefined AUC for every label
    per = per_label_roc_auc(y_true, y_score)
    assert all(v is None for v in per.values())
    result = macro_roc_auc(y_true, y_score)
    assert np.isnan(result.macro_auc)
    assert set(result.skipped_labels) == set(TARGET_COLUMNS)
