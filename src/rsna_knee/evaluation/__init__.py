"""Evaluation exports."""

from rsna_knee.evaluation.leakage import assert_unique_study_ids, audit_fold_leakage
from rsna_knee.evaluation.metrics import MetricResult, macro_roc_auc, per_label_roc_auc
from rsna_knee.evaluation.splits import (
    FoldAssignment,
    assert_no_group_overlap,
    gold_label_mask,
    has_any_gold_label,
    iter_fold_frames,
    make_study_folds,
)

__all__ = [
    "FoldAssignment",
    "MetricResult",
    "assert_no_group_overlap",
    "assert_unique_study_ids",
    "audit_fold_leakage",
    "gold_label_mask",
    "has_any_gold_label",
    "iter_fold_frames",
    "macro_roc_auc",
    "make_study_folds",
    "per_label_roc_auc",
]
