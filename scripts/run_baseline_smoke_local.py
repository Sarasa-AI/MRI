#!/usr/bin/env python3
"""Local synthetic smoke for RSNA-BASELINE-001 (no competition DICOMs).

Runs 1-fold, few epochs, synthetic mid-slices on MPS/CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna_knee.data.loaders import (
    load_series_metadata,
    load_test_metadata,
    load_train_metadata,
)
from rsna_knee.datasets.synthetic import write_fixtures
from rsna_knee.training.imaging_loop import BaselineConfig, run_imaging_baseline
from rsna_knee.training.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/RSNA-BASELINE-001-smoke"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--n-studies", type=int, default=16)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Download ImageNet weights (needs network + torch hub cache).",
    )
    args = parser.parse_args()

    seed_everything(42)
    fixtures = args.output / "fixtures"
    write_fixtures(fixtures, n_studies=args.n_studies, n_test=3)

    train = load_train_metadata(fixtures / "train.csv")
    train_series = load_series_metadata(fixtures / "train_series.csv")
    test = load_test_metadata(fixtures / "test.csv")
    test_series = load_series_metadata(fixtures / "test_series.csv")

    cfg = BaselineConfig(
        experiment_id="RSNA-BASELINE-001-smoke",
        seed=42,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=4,
        pretrained=bool(args.pretrained),
        image_size=224,
        force_synthetic=True,
        num_workers=0,
        early_stopping_patience=5,
        amp=False,
    )
    result = run_imaging_baseline(
        train,
        train_series,
        test,
        test_series,
        train_series_root=None,
        test_series_root=None,
        output_dir=args.output,
        cfg=cfg,
    )
    print(result)


if __name__ == "__main__":
    main()
