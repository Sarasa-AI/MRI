#!/usr/bin/env python3
"""Local metadata-only smoke run (no DICOMs, no downloads)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.data.loaders import load_train_metadata
from rsna_knee.datasets.synthetic import write_fixtures
from rsna_knee.runtime.adapter import resolve_data_paths
from rsna_knee.training.loop import run_metadata_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", default=str(ROOT / "data" / "fixtures"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "EXP-000-smoke"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--experiment-id", default="EXP-000-smoke")
    args = parser.parse_args()

    fixtures = Path(args.fixtures_dir)
    if not (fixtures / "train.csv").exists():
        write_fixtures(fixtures)

    paths = resolve_data_paths(mode="metadata_only", fixtures_dir=fixtures)
    train_df = load_train_metadata(paths.train_csv)
    result = run_metadata_smoke(
        train_df,
        experiment_id=args.experiment_id,
        n_splits=args.n_folds,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(
        f"OK experiment={result.experiment_id} "
        f"macro_auc_mean={result.macro_auc_mean} oof={result.oof_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
