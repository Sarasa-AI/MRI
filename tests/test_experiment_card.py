"""Experiment card schema tests."""

from __future__ import annotations

import pytest

from rsna_knee.utils.experiment import ExperimentCard, load_experiment_card, save_experiment_card


def test_experiment_card_roundtrip(tmp_path):
    card = ExperimentCard(
        experiment_id="EXP-000-smoke",
        configuration="configs/experiment/exp000_smoke.yaml",
        seed=42,
        model="ConstantPriorModel",
        input_geometry="metadata_only",
        data_selection="gold_labels_only",
        labels="official_12_targets",
        folds="GroupKFold",
        augmentation="none",
        optimizer="none",
        scheduler="none",
        metric="macro_roc_auc=nan",
        runtime="local",
        oof_predictions="outputs/oof.csv",
        conclusion="platform smoke",
    )
    path = save_experiment_card(card, tmp_path / "card.json")
    loaded = load_experiment_card(path)
    assert loaded.experiment_id == "EXP-000-smoke"


def test_missing_field_rejected():
    card = ExperimentCard(
        experiment_id="",
        configuration="x",
        seed=1,
        model="m",
        input_geometry="g",
        data_selection="d",
        labels="l",
        folds="f",
        augmentation="a",
        optimizer="o",
        scheduler="s",
        metric="m",
        runtime="r",
        oof_predictions="p",
        conclusion="c",
    )
    with pytest.raises(ValueError, match="missing required"):
        card.validate()
