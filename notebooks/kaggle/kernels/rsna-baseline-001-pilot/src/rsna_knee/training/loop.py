"""Minimal training interface — stub only, no real model training yet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.evaluation.metrics import macro_roc_auc
from rsna_knee.evaluation.splits import gold_label_mask, iter_fold_frames, make_study_folds
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.models.stub import ConstantPriorModel
from rsna_knee.training.seed import seed_everything
from rsna_knee.utils.experiment import ExperimentCard, save_experiment_card
from rsna_knee.utils.logging import append_results_row, get_logger

logger = get_logger(__name__)


@dataclass
class SmokeTrainResult:
    experiment_id: str
    macro_auc_mean: float
    fold_metrics: list[dict[str, Any]]
    oof_path: str | None


def run_metadata_smoke(
    train_df: pd.DataFrame,
    *,
    experiment_id: str = "EXP-000-smoke",
    n_splits: int = 5,
    seed: int = 42,
    output_dir: str | Path = "outputs/EXP-000-smoke",
) -> SmokeTrainResult:
    """Metadata-only smoke: constant prior baseline on gold-labeled rows.

    Does not load DICOMs. Intended for local unit validation of the
    experiment platform (splits, metrics, OOF schema, logging).
    """
    seed_everything(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gold = train_df.loc[gold_label_mask(train_df)].copy()
    if gold.empty:
        raise ValueError("No gold-labeled rows available for smoke training")

    # Stratify on ACL if both classes exist; else plain GroupKFold.
    stratify_col = None
    if gold["ACL"].nunique(dropna=True) >= 2:
        stratify_col = "ACL"

    folds = make_study_folds(
        gold, n_splits=min(n_splits, len(gold)), seed=seed, stratify_col=stratify_col
    )

    oof = gold[[STUDY_ID_COL]].copy()
    for col in TARGET_COLUMNS:
        oof[col] = np.nan

    fold_metrics: list[dict[str, Any]] = []
    for fold_i, tr, va in iter_fold_frames(gold, folds):
        model = ConstantPriorModel()
        model.fit(tr)
        preds = model.predict_df(va)
        y_true = va[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        y_score = preds[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        metrics = macro_roc_auc(y_true, y_score)
        fold_metrics.append({"fold": fold_i, **metrics.to_dict()})
        for col in TARGET_COLUMNS:
            oof.loc[oof[STUDY_ID_COL].isin(va[STUDY_ID_COL]), col] = preds[col].to_numpy()
        logger.info("fold=%s macro_auc=%s", fold_i, metrics.macro_auc)

    oof_path = out / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    macros = [m["macro_auc"] for m in fold_metrics if not np.isnan(m["macro_auc"])]
    macro_mean = float(np.mean(macros)) if macros else float("nan")

    card = ExperimentCard(
        experiment_id=experiment_id,
        configuration="configs/experiment/exp000_smoke.yaml",
        seed=seed,
        model="ConstantPriorModel",
        input_geometry="metadata_only",
        data_selection="gold_labels_only",
        labels="official_12_targets",
        folds=f"GroupKFold n_splits={len(folds)} by StudyInstanceUID",
        augmentation="none",
        optimizer="none",
        scheduler="none",
        metric=f"macro_roc_auc={macro_mean:.6f}",
        runtime="local_metadata_only",
        oof_predictions=str(oof_path),
        conclusion="Platform smoke only — not a modeling result.",
    )
    save_experiment_card(card, out / "experiment_card.json")
    results_path = out / "results_row.csv"
    # Prefer repo results table when running from a full checkout.
    repo_results = Path(__file__).resolve().parents[3] / "experiments" / "results" / "results_table.csv"
    if repo_results.parent.exists():
        results_path = repo_results
    append_results_row(
        {
            "experiment_id": experiment_id,
            "macro_auc_mean": macro_mean,
            "n_folds": len(folds),
            "n_gold": len(gold),
            "seed": seed,
        },
        path=results_path,
    )

    return SmokeTrainResult(
        experiment_id=experiment_id,
        macro_auc_mean=macro_mean,
        fold_metrics=fold_metrics,
        oof_path=str(oof_path),
    )
