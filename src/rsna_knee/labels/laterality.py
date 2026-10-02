"""Label-policy helpers (laterality / compartment flips, report boundary)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from rsna_knee.labels.targets import (
    COMPARTMENT_FLIP_PAIRS,
    REPORT_ALLOWED_AT_INFERENCE,
    REPORT_COL,
    TARGET_COLUMNS,
)


def assert_vision_only_inference_inputs(columns: list[str] | tuple[str, ...]) -> None:
    """Reject any inference feature set that depends on report text."""
    if REPORT_ALLOWED_AT_INFERENCE:
        return
    if REPORT_COL in columns:
        raise ValueError(
            f"Inference inputs must not include '{REPORT_COL}'. "
            "Reports are training-time only (pseudo-labels / distillation)."
        )


def swap_compartment_labels(row: Mapping[str, float] | pd.Series) -> dict[str, float]:
    """Swap medial↔lateral compartment labels after a horizontal flip."""
    out = {name: float(row[name]) for name in TARGET_COLUMNS}
    for left, right in COMPARTMENT_FLIP_PAIRS:
        out[left], out[right] = out[right], out[left]
    return out


def swap_compartment_label_matrix(y: np.ndarray) -> np.ndarray:
    """Swap medial↔lateral columns in a (N, 12) label matrix."""
    if y.ndim != 2 or y.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"Expected shape (N, {len(TARGET_COLUMNS)}), got {y.shape}")
    out = y.copy()
    name_to_idx = {name: i for i, name in enumerate(TARGET_COLUMNS)}
    for left, right in COMPARTMENT_FLIP_PAIRS:
        i, j = name_to_idx[left], name_to_idx[right]
        out[:, [i, j]] = out[:, [j, i]]
    return out


# Explicit augmentation policy for this project (documented decision D-007):
# horizontal flips are ALLOWED only if compartment pairs are swapped.
HORIZONTAL_FLIP_POLICY: str = (
    "allow_with_compartment_swap"
)
