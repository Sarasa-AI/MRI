"""Bootstrap confidence intervals for macro / per-label ROC-AUC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rsna_knee.evaluation.metrics import macro_roc_auc
from rsna_knee.labels.targets import TARGET_COLUMNS


@dataclass(frozen=True)
class BootstrapAUC:
    point: float
    low: float
    high: float
    n_boot: int
    n_effective: int

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "ci95_low": self.low,
            "ci95_high": self.high,
            "n_boot": self.n_boot,
            "n_effective": self.n_effective,
        }


def bootstrap_macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    label_names: Sequence[str] = TARGET_COLUMNS,
) -> BootstrapAUC:
    """Study-level bootstrap of macro ROC-AUC."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = y_true.shape[0]
    point = macro_roc_auc(y_true, y_score, label_names=label_names).macro_auc
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = macro_roc_auc(y_true[idx], y_score[idx], label_names=label_names).macro_auc
        if m == m:  # not NaN
            samples.append(float(m))
    if not samples:
        return BootstrapAUC(point=float(point), low=float("nan"), high=float("nan"), n_boot=n_boot, n_effective=0)
    arr = np.asarray(samples, dtype=float)
    return BootstrapAUC(
        point=float(point),
        low=float(np.quantile(arr, 0.025)),
        high=float(np.quantile(arr, 0.975)),
        n_boot=n_boot,
        n_effective=int(arr.size),
    )


def bootstrap_per_label_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    label_names: Sequence[str] = TARGET_COLUMNS,
) -> dict[str, BootstrapAUC]:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = y_true.shape[0]
    base = macro_roc_auc(y_true, y_score, label_names=label_names).per_label_auc
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[float]] = {name: [] for name in label_names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        per = macro_roc_auc(y_true[idx], y_score[idx], label_names=label_names).per_label_auc
        for name, auc in per.items():
            if auc == auc:
                buckets[name].append(float(auc))
    out: dict[str, BootstrapAUC] = {}
    for name in label_names:
        arr = np.asarray(buckets[name], dtype=float)
        point = float(base[name]) if base[name] == base[name] else float("nan")
        if arr.size == 0:
            out[name] = BootstrapAUC(point, float("nan"), float("nan"), n_boot, 0)
        else:
            out[name] = BootstrapAUC(
                point=point,
                low=float(np.quantile(arr, 0.025)),
                high=float(np.quantile(arr, 0.975)),
                n_boot=n_boot,
                n_effective=int(arr.size),
            )
    return out
