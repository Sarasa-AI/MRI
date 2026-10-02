"""Unit tests for offline report→label extraction and strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.evaluation.splits import gold_label_mask
from rsna_knee.labels.audit import run_label_audit, score_against_gold
from rsna_knee.labels.laterality import assert_vision_only_inference_inputs
from rsna_knee.labels.report_parser import (
    STATE_MISSING,
    STATE_NEGATIVE,
    STATE_POSITIVE,
    extract_report,
    extract_reports_frame,
    normalize_report,
)
from rsna_knee.labels.strategies import STRATEGY_NAMES, build_all_strategies
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS


def test_normalize_turkish_i():
    assert "iliak" in normalize_report("İLİAK")


def test_acl_positive_and_negation_english():
    pos = extract_report("Findings: Complete tear of the ACL. MCL is intact. No fracture is seen.")
    assert pos["ACL"].state == STATE_POSITIVE
    assert pos["MCL"].state == STATE_NEGATIVE
    assert pos["Fracture"].state == STATE_NEGATIVE
    assert pos["Synovitis"].state == STATE_MISSING


def test_spanish_meniscus_and_effusion():
    text = (
        "Hallazgos: Rotura del menisco interno. Menisco lateral sin signos de rotura. "
        "Derrame articular. Pas de fracture."
    )
    # mixed — Spanish meniscus/effusion
    ex = extract_report(
        "Hallazgos: Rotura del menisco interno. Menisco lateral sin signos de rotura. Derrame articular."
    )
    assert ex["Medial Meniscus"].state == STATE_POSITIVE
    assert ex["Effusion"].state == STATE_POSITIVE


def test_baker_and_uncertain():
    ex = extract_report(
        "There is a small Baker's cyst. Possible synovitis. No evidence of fracture."
    )
    assert ex["Baker's"].state == STATE_POSITIVE
    assert ex["Fracture"].state == STATE_NEGATIVE
    assert ex["Synovitis"].uncertain or ex["Synovitis"].state in {
        STATE_POSITIVE,
        "U",
    }


def test_strategies_gold_only_and_vision_boundary(tmp_path):
    rows = []
    for i in range(12):
        row = {STUDY_ID_COL: f"s{i}", "Report": "ACL is intact. No effusion."}
        if i < 6:
            for j, c in enumerate(TARGET_COLUMNS):
                row[c] = float((i + j) % 2)
        else:
            for c in TARGET_COLUMNS:
                row[c] = None
        rows.append(row)
    # Make one silver report clearly positive ACL
    rows[6]["Report"] = "Complete rupture of the ACL with bone contusion. Large effusion."
    df = pd.DataFrame(rows)
    extracted = extract_reports_frame(df)
    bundles = build_all_strategies(df, extracted=extracted)
    assert set(bundles) == set(STRATEGY_NAMES)
    assert bundles["A_gold"].meta["n_studies"] == 6
    assert bundles["B_hard"].meta["n_studies"] >= 6
    # Inference must reject Report
    with pytest.raises(ValueError, match="Report"):
        assert_vision_only_inference_inputs([STUDY_ID_COL, "Report"])


def test_label_audit_smoke(tmp_path):
    rows = []
    for i in range(16):
        if i % 2 == 0:
            report = "Complete tear of the ACL. Medial meniscus tear. Large effusion. No fracture."
            labels = {c: 0.0 for c in TARGET_COLUMNS}
            labels["ACL"] = 1.0
            labels["Medial Meniscus"] = 1.0
            labels["Effusion"] = 1.0
        else:
            report = "ACL is intact. Medial meniscus is not torn. No effusion. No fracture."
            labels = {c: 0.0 for c in TARGET_COLUMNS}
        row = {STUDY_ID_COL: f"g{i}", "Report": report, **labels}
        rows.append(row)
    # silver
    for i in range(8):
        rows.append(
            {
                STUDY_ID_COL: f"s{i}",
                "Report": "Baker cyst and mild effusion. ACL intact.",
                **{c: None for c in TARGET_COLUMNS},
            }
        )
    df = pd.DataFrame(rows)
    assert int(gold_label_mask(df).sum()) == 16
    summary = run_label_audit(df, output_dir=tmp_path / "audit", n_boot=50, seed=0)
    assert summary["n_gold_studies"] == 16
    assert summary["report_soft_vs_gold"]["macro_auc"] > 0.6
    assert (tmp_path / "audit" / "label_audit_summary.json").exists()
    soft = extract_reports_frame(df)
    from rsna_knee.labels.strategies import report_labels_as_predictions

    scored = score_against_gold(df, report_labels_as_predictions(soft), n_boot=20, seed=0)
    assert scored["macro_auc"] == scored["macro_auc"]
