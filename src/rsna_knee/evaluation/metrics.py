"""Competition metrics: macro ROC-AUC and per-label AUC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from rsna_knee.labels.targets import TARGET_COLUMNS


@dataclass(frozen=True)
class MetricResult:
    macro_auc: float
    per_label_auc: dict[str, float]
    evaluated_labels: tuple[str, ...]
    skipped_labels: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "macro_auc": self.macro_auc,
            "per_label_auc": dict(self.per_label_auc),
            "evaluated_labels": list(self.evaluated_labels),
            "skipped_labels": list(self.skipped_labels),
        }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return AUC or None if the label is undefined (single class / all-NaN).

    Also drops rows where ``y_score`` is NaN so ``pilot_folds`` partial OOF
    (unrun folds left as NaN) does not crash sklearn. Full 5-fold runs have no
    score NaNs, so this does not change the official baseline metric contract.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    if mask.sum() < 2:
        return None
    yt = y_true[mask]
    ys = y_score[mask]
    if np.unique(yt).size < 2:
        return None
    return float(roc_auc_score(yt, ys))


def per_label_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str] = TARGET_COLUMNS,
) -> dict[str, float | None]:
    """Compute ROC-AUC for each label; None when undefined."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != len(label_names):
        raise ValueError(
            f"Expected (N, {len(label_names)}), got y_true={y_true.shape}"
        )

    out: dict[str, float | None] = {}
    for i, name in enumerate(label_names):
        out[name] = _safe_auc(y_true[:, i], y_score[:, i])
    return out


def macro_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str] = TARGET_COLUMNS,
) -> MetricResult:
    """Macro-average ROC-AUC over labels that have a defined AUC.

    Competition metric is macro-average ROC AUC over the 12 targets.
    Labels with undefined AUC (no positive/negative examples in the eval
    slice) are skipped and reported separately so folds with sparse gold
    labels remain interpretable.
    """
    per_label = per_label_roc_auc(y_true, y_score, label_names=label_names)
    evaluated = {k: v for k, v in per_label.items() if v is not None}
    skipped = tuple(k for k, v in per_label.items() if v is None)

    if not evaluated:
        macro = float("nan")
    else:
        macro = float(np.mean(list(evaluated.values())))

    return MetricResult(
        macro_auc=macro,
        per_label_auc={k: (float("nan") if v is None else v) for k, v in per_label.items()},
        evaluated_labels=tuple(evaluated.keys()),
        skipped_labels=skipped,
    )
