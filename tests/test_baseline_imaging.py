"""Unit tests for series selection, compartment-safe aug, and imaging smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.data.series_selection import select_series_for_study
from rsna_knee.labels.targets import TARGET_COLUMNS
from rsna_knee.training.augment import AugmentConfig, augment_study


def test_select_prefers_fluid_sensitive():
    rows = [
        {
            "StudyInstanceUID": "s1",
            "SeriesInstanceUID": "a",
            "Fluid_Sensitive": 0,
            "Fat_Suppression": 1,
            "Anatomical_Plane": "Sagittal",
        },
        {
            "StudyInstanceUID": "s1",
            "SeriesInstanceUID": "b",
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 0,
            "Anatomical_Plane": "Sagittal",
        },
        {
            "StudyInstanceUID": "s1",
            "SeriesInstanceUID": "c",
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
            "Anatomical_Plane": "Coronal",
        },
    ]
    df = pd.DataFrame(rows)
    picks = select_series_for_study(df, "s1")
    by_plane = {p.plane: p.series_uid for p in picks}
    assert by_plane["Sagittal"] == "b"
    assert by_plane["Coronal"] == "c"


def test_hflip_swaps_compartment_labels():
    rng = np.random.default_rng(0)
    image = np.zeros((3, 32, 32), dtype=np.float32)
    image[:, :, :16] = 1.0  # left half bright
    labels = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(TARGET_COLUMNS)}
    labels[name_to_idx["Medial Meniscus"]] = 1.0
    labels[name_to_idx["Lateral Meniscus"]] = 0.0
    labels[name_to_idx["Medial OA"]] = 1.0
    labels[name_to_idx["Lateral OA"]] = 0.0
    labels[name_to_idx["ACL"]] = 1.0

    cfg = AugmentConfig(
        hflip_p=1.0,
        rotate_deg=0.0,
        brightness=0.0,
        contrast=0.0,
        scale_min=1.0,
        scale_max=1.0,
    )
    out_img, out_y, meta = augment_study(image, labels, cfg=cfg, rng=rng)
    assert meta["hflip"] is True
    assert out_img[0, 0, -1] == pytest.approx(1.0)  # flipped
    assert out_y[name_to_idx["Medial Meniscus"]] == pytest.approx(0.0)
    assert out_y[name_to_idx["Lateral Meniscus"]] == pytest.approx(1.0)
    assert out_y[name_to_idx["Medial OA"]] == pytest.approx(0.0)
    assert out_y[name_to_idx["Lateral OA"]] == pytest.approx(1.0)
    assert out_y[name_to_idx["ACL"]] == pytest.approx(1.0)


@pytest.mark.slow
def test_imaging_baseline_synthetic_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    _ = torch
    from rsna_knee.datasets.synthetic import write_fixtures
    from rsna_knee.data.loaders import (
        load_series_metadata,
        load_test_metadata,
        load_train_metadata,
    )
    from rsna_knee.training.imaging_loop import BaselineConfig, run_imaging_baseline

    fixtures = tmp_path / "fixtures"
    write_fixtures(fixtures, n_studies=12, n_test=2)
    cfg = BaselineConfig(
        experiment_id="RSNA-BASELINE-001-test",
        n_folds=2,
        epochs=1,
        batch_size=2,
        pretrained=False,  # avoid network in CI
        force_synthetic=True,
        image_size=64,
        early_stopping_patience=1,
        amp=False,
        num_workers=0,
    )
    result = run_imaging_baseline(
        load_train_metadata(fixtures / "train.csv"),
        load_series_metadata(fixtures / "train_series.csv"),
        load_test_metadata(fixtures / "test.csv"),
        load_series_metadata(fixtures / "test_series.csv"),
        train_series_root=None,
        test_series_root=None,
        output_dir=tmp_path / "out",
        cfg=cfg,
    )
    assert (tmp_path / "out" / "oof_predictions.csv").exists()
    assert (tmp_path / "out" / "submission.csv").exists()
    assert (tmp_path / "out" / "experiment_card.json").exists()
    assert (tmp_path / "out" / "metrics_summary.json").exists()
    assert result.experiment_id == "RSNA-BASELINE-001-test"
