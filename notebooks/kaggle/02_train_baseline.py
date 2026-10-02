"""Kaggle entrypoint: RSNA-BASELINE-001 imaging floor.

Attach competition `rsna-knee-abnormality-detection` + this repo as a Dataset
(or paste `src/` onto sys.path). Run on GPU. Internet ON only for the first
Save Version if ImageNet weights must download; subsequent / submission runs
should use cached checkpoints with Internet OFF.

NEVER download competition DICOMs to a local Mac from this notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- path bootstrap (Kaggle Dataset mount variants) ---
CANDIDATE_SRC = [
    Path("/kaggle/input/rsna-knee-control/src"),
    Path("/kaggle/input/rsna-knee/src"),
    Path("/kaggle/working/src"),
    Path(__file__).resolve().parents[2] / "src",
]
for p in CANDIDATE_SRC:
    if (p / "rsna_knee").exists():
        sys.path.insert(0, str(p))
        break

import pandas as pd

from rsna_knee.data.loaders import (
    load_series_metadata,
    load_test_metadata,
    load_train_metadata,
)
from rsna_knee.runtime.adapter import resolve_data_paths
from rsna_knee.training.imaging_loop import BaselineConfig, load_baseline_config, run_imaging_baseline
from rsna_knee.training.seed import seed_everything


def _find_config() -> Path | None:
    candidates = [
        Path("/kaggle/input/rsna-knee-control/configs/experiment/rsna_baseline_001.yaml"),
        Path("/kaggle/input/rsna-knee/configs/experiment/rsna_baseline_001.yaml"),
        Path(__file__).resolve().parents[2] / "configs" / "experiment" / "rsna_baseline_001.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    seed_everything(42)
    paths = resolve_data_paths(mode="kaggle", allow_missing_dicom=False)
    print("resolved_root=", paths.root)
    print("train_series_dir=", paths.train_series_dir)
    print("test_series_dir=", paths.test_series_dir)

    config_path = _find_config()
    cfg = load_baseline_config(config_path) if config_path else BaselineConfig()
    # Full 5-fold on ~58 gold studies is cheap for EfficientNet-B0@224.
    # Uncomment for a 1-fold I/O pilot first:
    # cfg.pilot_folds = 1

    train = load_train_metadata(paths.train_csv)
    train_series = load_series_metadata(paths.train_series_csv)
    test = load_test_metadata(paths.test_csv)
    test_series = load_series_metadata(paths.test_series_csv)

    out = paths.working_dir / cfg.experiment_id
    result = run_imaging_baseline(
        train,
        train_series,
        test,
        test_series,
        train_series_root=paths.train_series_dir,
        test_series_root=paths.test_series_dir,
        output_dir=out,
        cfg=cfg,
        config_path=config_path,
    )

    # Competition-visible submission path
    sub_src = Path(result.output_dir) / "submission.csv"
    sub_dst = paths.working_dir / "submission.csv"
    sub_dst.write_bytes(sub_src.read_bytes())

    print("=== RSNA-BASELINE-001 ===")
    print("oof_macro_auc=", result.oof_macro_auc)
    print("per_label_auc=", result.per_label_auc)
    print("train_runtime_sec=", result.train_runtime_sec)
    print("infer_runtime_sec=", result.infer_runtime_sec)
    print("peak_rss_mb=", result.peak_rss_mb)
    print("cuda_peak_mb=", result.cuda_peak_mb)
    print("output_dir=", result.output_dir)
    print("conclusion=", result.conclusion)


if __name__ == "__main__":
    main()
