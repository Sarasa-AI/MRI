"""End-to-end metadata smoke (local fixtures only)."""

from __future__ import annotations

from rsna_knee.data.loaders import load_train_metadata
from rsna_knee.datasets.synthetic import write_fixtures
from rsna_knee.runtime.adapter import resolve_data_paths
from rsna_knee.training.loop import run_metadata_smoke


def test_metadata_smoke(tmp_path):
    fixtures = tmp_path / "fixtures"
    write_fixtures(fixtures, n_studies=20)
    paths = resolve_data_paths(mode="metadata_only", fixtures_dir=fixtures, working_dir=tmp_path / "work")
    train = load_train_metadata(paths.train_csv)
    result = run_metadata_smoke(
        train,
        experiment_id="EXP-000-smoke",
        n_splits=4,
        seed=42,
        output_dir=tmp_path / "out",
    )
    assert result.oof_path is not None
    assert (tmp_path / "out" / "experiment_card.json").exists()
