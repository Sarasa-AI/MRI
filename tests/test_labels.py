"""Report-boundary and compartment-flip policy tests."""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.labels.laterality import (
    assert_vision_only_inference_inputs,
    swap_compartment_label_matrix,
    swap_compartment_labels,
)
from rsna_knee.labels.targets import TARGET_COLUMNS


def test_report_rejected_at_inference():
    with pytest.raises(ValueError, match="Report"):
        assert_vision_only_inference_inputs(["StudyInstanceUID", "Report", "ACL"])


def test_compartment_swap():
    row = {name: float(i) for i, name in enumerate(TARGET_COLUMNS)}
    swapped = swap_compartment_labels(row)
    assert swapped["Medial Meniscus"] == row["Lateral Meniscus"]
    assert swapped["Lateral Meniscus"] == row["Medial Meniscus"]
    assert swapped["Medial OA"] == row["Lateral OA"]
    assert swapped["ACL"] == row["ACL"]


def test_compartment_swap_matrix():
    y = np.arange(24, dtype=float).reshape(2, 12)
    out = swap_compartment_label_matrix(y)
    idx = {n: i for i, n in enumerate(TARGET_COLUMNS)}
    assert out[0, idx["Medial Meniscus"]] == y[0, idx["Lateral Meniscus"]]
    assert out[0, idx["Lateral OA"]] == y[0, idx["Medial OA"]]
