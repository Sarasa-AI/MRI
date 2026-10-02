"""Unit tests for read-only DICOM audit helpers (no competition DICOMs)."""

from __future__ import annotations

import numpy as np

from rsna_knee.data.dicom_audit import (
    classify_array,
    compare_instance_vs_ipp_order,
    decide_dicom_gate,
    _spearman,
)


def test_classify_array_statuses():
    assert classify_array(np.zeros((8, 8), dtype=np.float32)).status == "zero"
    valid = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    assert classify_array(valid).status == "valid"
    nan = valid.copy()
    nan[0, 0] = np.nan
    assert classify_array(nan).status == "nan"


def test_ordering_agreement_and_disagreement():
    # Perfect agreement: InstanceNumber increases with IPP projection.
    agree = []
    for i in range(10):
        agree.append(
            {
                "InstanceNumber": i + 1,
                "ImagePositionPatient": [0.0, 0.0, float(i)],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
            }
        )
    res = compare_instance_vs_ipp_order(agree)
    assert res["comparable"] is True
    assert res["abs_spearman"] == 1.0
    assert res["pairwise_inversion_rate"] == 0.0

    # Scrambled InstanceNumber vs physical order → disagreement.
    scramble = []
    order = [0, 2, 1, 4, 3, 6, 5, 8, 7, 9]
    for i, z in enumerate(order):
        scramble.append(
            {
                "InstanceNumber": i + 1,
                "ImagePositionPatient": [0.0, 0.0, float(z)],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
            }
        )
    bad = compare_instance_vs_ipp_order(scramble)
    assert bad["comparable"] is True
    assert bad["pairwise_inversion_rate"] is not None
    assert bad["pairwise_inversion_rate"] > 0.05


def test_spearman_and_gate_helper():
    assert abs(_spearman(np.arange(5.0), np.arange(5.0)) - 1.0) < 1e-9
    mid = {
        "success_rate": 1.0,
        "degenerate_rate": 0.0,
        "failure_counts": {},
    }
    header = {
        "ordering_summary": {
            "n_comparable": 10,
            "disagreement_rate": 0.0,
        },
        "orientation_note_counts": {},
        "series": [{"orientation": {"n_with_iop": 1}}],
    }
    gate = decide_dicom_gate(mid, header)
    assert gate["blockers"] == []
    assert gate["decode_ok"] and gate["ordering_ok"]
