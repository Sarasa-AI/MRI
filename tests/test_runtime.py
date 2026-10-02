"""Runtime adapter tests (no Kaggle mount required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna_knee.datasets.synthetic import write_fixtures
from rsna_knee.runtime.adapter import RuntimeMode, detect_runtime, resolve_data_paths


def test_default_local_is_metadata_only(monkeypatch):
    monkeypatch.delenv("RSNA_RUNTIME_MODE", raising=False)
    monkeypatch.setattr(
        "rsna_knee.runtime.adapter.Path.exists",
        lambda self: False if str(self).startswith("/kaggle") else Path.exists(self),
    )
    # Force detection without relying on monkeypatch of Path.exists globally
    mode = detect_runtime(force_mode="metadata_only")
    assert mode == RuntimeMode.METADATA_ONLY


def test_resolve_fixtures(tmp_path):
    write_fixtures(tmp_path / "fixtures")
    paths = resolve_data_paths(mode="metadata_only", fixtures_dir=tmp_path / "fixtures")
    assert paths.train_csv.exists()
    assert paths.mode == RuntimeMode.METADATA_ONLY
    assert paths.dicom_available is False


def test_env_root_override(tmp_path, monkeypatch):
    write_fixtures(tmp_path / "fixtures")
    monkeypatch.setenv("RSNA_DATA_ROOT", str(tmp_path / "fixtures"))
    paths = resolve_data_paths(mode="metadata_only")
    assert paths.root == tmp_path / "fixtures"


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_paths(mode="metadata_only", fixtures_dir=tmp_path)
