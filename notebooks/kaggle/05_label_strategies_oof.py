"""Kaggle GPU: imaging OOF for label strategies A–E.

Requires competition DICOMs. Trains EfficientNet-B0 multi-plane mid-slice with
each strategy; evaluates OOF on gold studies only. Inference remains vision-only.

Pass criterion vs A_gold: gold OOF macro AUC ↑ >0.005, broad lift, bootstrap CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for cand in (
    Path("/kaggle/input/rsna-knee-control/src"),
    Path("/kaggle/input/rsna-knee-control/MRI/src"),
    Path("src"),
):
    if cand.exists():
        sys.path.insert(0, str(cand))
        break

import pandas as pd

from rsna_knee.labels.laterality import assert_vision_only_inference_inputs
from rsna_knee.labels.report_parser import extract_reports_frame
from rsna_knee.labels.strategies import STRATEGY_NAMES
from rsna_knee.runtime.adapter import resolve_data_paths
from rsna_knee.training.imaging_loop import BaselineConfig
from rsna_knee.training.strategy_oof import run_all_strategy_imaging_oof


def main() -> None:
    paths = resolve_data_paths(mode="kaggle", allow_missing_dicom=False)
    train = pd.read_csv(paths.train_csv)
    series = pd.read_csv(paths.train_series_csv)
    assert_vision_only_inference_inputs(["StudyInstanceUID"])

    extracted = extract_reports_frame(train)
    cfg = BaselineConfig(
        experiment_id="RSNA-LABELS-001",
        epochs=25,
        n_folds=5,
        batch_size=8,
        force_synthetic=False,
    )
    out = paths.working_dir / "RSNA-LABELS-001-imaging"
    # Memory/runtime risk: 5 strategies × 5 folds × 25 epochs on ~4k studies.
    # Prefer running D and E first if GPU hours are tight.
    results = run_all_strategy_imaging_oof(
        train,
        series,
        train_series_root=paths.train_series_dir,
        output_dir=out,
        cfg=cfg,
        strategies=STRATEGY_NAMES,
        extracted=extracted,
    )
    print(json.dumps({k: v["macro_auc"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
