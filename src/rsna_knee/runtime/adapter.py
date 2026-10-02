"""Runtime environment detection and competition path resolution.

Never hard-code a local Mac dataset path. Paths come from:
  1) explicit config / env overrides
  2) Kaggle mount discovery under /kaggle/input
  3) local metadata-only fixtures for unit tests
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from rsna_knee.labels.targets import COMPETITION_SLUG


class RuntimeMode(str, Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True)
class DataPaths:
    """Resolved filesystem locations for competition artifacts."""

    mode: RuntimeMode
    root: Path
    train_csv: Path
    train_series_csv: Path
    test_csv: Path
    test_series_csv: Path
    sample_submission_csv: Path
    train_series_dir: Path | None
    test_series_dir: Path | None
    working_dir: Path

    @property
    def dicom_available(self) -> bool:
        return self.train_series_dir is not None and self.train_series_dir.exists()


def detect_runtime(
    force_mode: str | None = None,
    *,
    kaggle_marker: Path = Path("/kaggle/input"),
) -> RuntimeMode:
    """Detect whether we are on Kaggle, local, or metadata-only."""
    if force_mode:
        return RuntimeMode(force_mode)

    env_mode = os.environ.get("RSNA_RUNTIME_MODE")
    if env_mode:
        return RuntimeMode(env_mode)

    if kaggle_marker.exists() or Path("/kaggle/working").exists():
        return RuntimeMode.KAGGLE

    # Default local execution is metadata/fixtures only — no competition DICOMs.
    return RuntimeMode.METADATA_ONLY


def _candidate_kaggle_roots(slug: str = COMPETITION_SLUG) -> list[Path]:
    # RSNA-DATA-002 VERIFIED mount is /kaggle/input/competitions/<slug>
    # (plain /kaggle/input/<slug> was absent on the audit runtime).
    return [
        Path("/kaggle/input/competitions") / slug,
        Path("/kaggle/input") / slug,
    ]


def discover_kaggle_root(
    slug: str = COMPETITION_SLUG,
    *,
    input_root: Path = Path("/kaggle/input"),
) -> Path:
    """Find competition mount by locating train.csv under /kaggle/input."""
    for candidate in _candidate_kaggle_roots(slug):
        if (candidate / "train.csv").exists():
            return candidate

    if input_root.exists():
        matches = sorted(input_root.rglob("train.csv"))
        for match in matches:
            # Prefer competition-shaped trees (sibling train_series.csv).
            if (match.parent / "train_series.csv").exists():
                return match.parent
        if matches:
            return matches[0].parent

    raise FileNotFoundError(
        f"Could not locate competition data under {input_root}. "
        f"Tried: {[str(p) for p in _candidate_kaggle_roots(slug)]}"
    )


def _require_files(root: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files under {root}: {missing}")


def resolve_data_paths(
    *,
    mode: RuntimeMode | str | None = None,
    root: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    working_dir: str | Path | None = None,
    allow_missing_dicom: bool = True,
) -> DataPaths:
    """Resolve CSV / optional DICOM paths for the active runtime.

    Parameters
    ----------
    root:
        Explicit competition root. If set, overrides discovery.
        Must NOT be a hard-coded Mac path in committed configs.
    fixtures_dir:
        Local synthetic fixtures for metadata-only / unit-test mode.
    """
    runtime = detect_runtime(force_mode=mode if isinstance(mode, str) else (mode.value if mode else None))

    env_root = os.environ.get("RSNA_DATA_ROOT")
    resolved_root: Path
    train_series_dir: Path | None
    test_series_dir: Path | None

    if root is not None:
        resolved_root = Path(root)
    elif env_root:
        resolved_root = Path(env_root)
    elif runtime == RuntimeMode.KAGGLE:
        resolved_root = discover_kaggle_root()
    else:
        # metadata_only / local → fixtures only
        if fixtures_dir is not None:
            resolved_root = Path(fixtures_dir)
        else:
            # Relative to package repo: data/fixtures
            resolved_root = Path(__file__).resolve().parents[3] / "data" / "fixtures"

    required_csvs = [
        "train.csv",
        "train_series.csv",
        "test.csv",
        "test_series.csv",
        "sample_submission.csv",
    ]
    _require_files(resolved_root, required_csvs)

    train_series_candidate = resolved_root / "train_series"
    test_series_candidate = resolved_root / "test_series"
    train_series_dir = train_series_candidate if train_series_candidate.exists() else None
    test_series_dir = test_series_candidate if test_series_candidate.exists() else None

    if runtime == RuntimeMode.KAGGLE and not allow_missing_dicom:
        if train_series_dir is None or test_series_dir is None:
            raise FileNotFoundError(
                f"Kaggle mode expects DICOM dirs under {resolved_root}"
            )

    if working_dir is not None:
        work = Path(working_dir)
    elif runtime == RuntimeMode.KAGGLE:
        work = Path("/kaggle/working")
    else:
        work = Path(os.environ.get("RSNA_WORKING_DIR", "outputs"))

    work.mkdir(parents=True, exist_ok=True)

    return DataPaths(
        mode=runtime if root is None and not env_root else (
            RuntimeMode.KAGGLE if runtime == RuntimeMode.KAGGLE else RuntimeMode.METADATA_ONLY
        ),
        root=resolved_root,
        train_csv=resolved_root / "train.csv",
        train_series_csv=resolved_root / "train_series.csv",
        test_csv=resolved_root / "test.csv",
        test_series_csv=resolved_root / "test_series.csv",
        sample_submission_csv=resolved_root / "sample_submission.csv",
        train_series_dir=train_series_dir,
        test_series_dir=test_series_dir,
        working_dir=work,
    )
