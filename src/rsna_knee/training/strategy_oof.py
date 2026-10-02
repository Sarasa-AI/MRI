"""Imaging OOF runner for label strategies A–E (vision-only inference).

Train on strategy supervision; always score OOF on gold studies using the same
StudyInstanceUID fold map as RSNA-BASELINE-001. Reports never enter inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.datasets.imaging import StudyMultiPlaneDataset, collate_studies
from rsna_knee.evaluation.bootstrap import bootstrap_macro_auc, bootstrap_per_label_auc
from rsna_knee.evaluation.metrics import macro_roc_auc
from rsna_knee.evaluation.splits import (
    assert_no_group_overlap,
    gold_label_mask,
    iter_fold_frames,
    make_study_folds,
)
from rsna_knee.labels.strategies import STRATEGY_NAMES, StrategyName, build_supervision
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.models.backbone_25d import BackboneConfig, EfficientNetB0MultiPlane25D
from rsna_knee.training.augment import AugmentConfig
from rsna_knee.training.imaging_loop import BaselineConfig, _make_loader, _predict_loader, _train_one_fold
from rsna_knee.training.seed import seed_everything
from rsna_knee.utils.logging import get_logger
from rsna_knee.utils.resources import pick_device

logger = get_logger(__name__)


def _gold_fold_map(gold: pd.DataFrame, *, n_folds: int, seed: int) -> list:
    stratify_col = "ACL" if gold["ACL"].nunique(dropna=True) >= 2 else None
    return make_study_folds(gold, n_splits=n_folds, seed=seed, stratify_col=stratify_col)


def run_strategy_imaging_oof(
    train_df: pd.DataFrame,
    train_series_df: pd.DataFrame,
    *,
    strategy: StrategyName,
    train_series_root: Path | None,
    output_dir: str | Path,
    cfg: BaselineConfig | None = None,
    extracted: pd.DataFrame | None = None,
    n_boot: int = 500,
) -> dict[str, Any]:
    """Train one strategy; evaluate OOF exclusively on gold rows."""
    cfg = cfg or BaselineConfig(experiment_id=f"RSNA-LABELS-001-{strategy}")
    seed_everything(cfg.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.force_synthetic:
        train_series_root = None

    gold = train_df.loc[gold_label_mask(train_df)].copy()
    if gold.empty:
        raise ValueError("No gold studies for evaluation")

    bundle = build_supervision(train_df, strategy=strategy, extracted=extracted)
    train_pool = bundle.studies
    weights = bundle.weights

    n_splits = min(cfg.n_folds, len(gold))
    folds = _gold_fold_map(gold, n_folds=n_splits, seed=cfg.seed)
    if cfg.pilot_folds is not None:
        folds = folds[: max(1, int(cfg.pilot_folds))]

    # Map gold val IDs → fold; training uses all strategy studies whose UID is
    # not in the gold validation fold (fold-safe silver).
    device = pick_device(cfg.device)
    amp = bool(cfg.amp and device.startswith("cuda"))
    cache_dir = out / cfg.cache_dirname
    aug_cfg = AugmentConfig(
        hflip_p=cfg.hflip_p,
        rotate_deg=cfg.rotate_deg,
        brightness=cfg.brightness,
        contrast=cfg.contrast,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    )

    oof = gold[[STUDY_ID_COL]].copy()
    for col in TARGET_COLUMNS:
        oof[col] = np.nan
    fold_metrics: list[dict] = []

    import torch

    for fold_i, tr_gold, va_gold in iter_fold_frames(gold, folds):
        val_ids = set(va_gold[STUDY_ID_COL].astype(str))
        # Fold-safe: drop any train_pool study that appears in gold val fold.
        tr = train_pool[~train_pool[STUDY_ID_COL].astype(str).isin(val_ids)].copy()
        w_tr = weights[~weights[STUDY_ID_COL].astype(str).isin(val_ids)].copy()
        assert_no_group_overlap(tr[STUDY_ID_COL], va_gold[STUDY_ID_COL], context=f"{strategy} fold {fold_i}")
        logger.info(
            "strategy=%s fold=%s train_studies=%s val_gold=%s",
            strategy,
            fold_i,
            len(tr),
            len(va_gold),
        )

        train_ds = StudyMultiPlaneDataset(
            tr,
            train_series_df,
            series_root=train_series_root,
            cache_dir=cache_dir,
            image_size=cfg.image_size,
            prefer_fat_suppression=cfg.prefer_fat_suppression,
            mid_frac=cfg.mid_frac,
            train=True,
            augment_cfg=aug_cfg,
            seed=cfg.seed + fold_i,
            weights=w_tr,
        )
        val_ds = StudyMultiPlaneDataset(
            va_gold,
            train_series_df,
            series_root=train_series_root,
            cache_dir=cache_dir,
            image_size=cfg.image_size,
            prefer_fat_suppression=cfg.prefer_fat_suppression,
            mid_frac=cfg.mid_frac,
            train=False,
            seed=cfg.seed + fold_i,
        )
        train_loader = _make_loader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        val_loader = _make_loader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        model = EfficientNetB0MultiPlane25D.build(
            BackboneConfig(
                backbone=cfg.backbone,
                pretrained=cfg.pretrained,
                dropout=cfg.dropout,
                num_classes=len(TARGET_COLUMNS),
            )
        )
        ckpt = out / "checkpoints" / f"{strategy}_fold{fold_i}.pt"
        hist = _train_one_fold(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            amp=amp,
            grad_clip=cfg.grad_clip,
            patience=cfg.early_stopping_patience,
            ckpt_path=ckpt,
        )
        # Reload best and predict val
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        metrics, scores = _predict_loader(model, val_loader, device=device, use_amp=amp)
        for uid, score in zip(va_gold[STUDY_ID_COL].astype(str), scores):
            idx = oof[STUDY_ID_COL].astype(str) == uid
            oof.loc[idx, list(TARGET_COLUMNS)] = score
        fold_metrics.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_val_gold": int(len(va_gold)),
                "macro_auc": metrics.macro_auc,
                "best_epoch": hist.get("best_epoch"),
            }
        )

    y_true = gold.set_index(STUDY_ID_COL).loc[oof[STUDY_ID_COL], list(TARGET_COLUMNS)].to_numpy(
        dtype=float
    )
    y_score = oof[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    oof_metrics = macro_roc_auc(y_true, y_score)
    boot = bootstrap_macro_auc(y_true, y_score, n_boot=n_boot, seed=cfg.seed)
    per_boot = bootstrap_per_label_auc(y_true, y_score, n_boot=n_boot, seed=cfg.seed)

    oof.to_csv(out / f"oof_{strategy}.csv", index=False)
    result = {
        "strategy": strategy,
        "macro_auc": oof_metrics.macro_auc,
        "per_label_auc": oof_metrics.per_label_auc,
        "bootstrap_macro": boot.to_dict(),
        "bootstrap_per_label": {k: v.to_dict() for k, v in per_boot.items()},
        "fold_metrics": fold_metrics,
        "supervision_meta": bundle.meta,
        "n_gold": int(len(gold)),
        "report_used_at_inference": False,
        "reproducible": True,
    }
    (out / f"metrics_{strategy}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_all_strategy_imaging_oof(
    train_df: pd.DataFrame,
    train_series_df: pd.DataFrame,
    *,
    train_series_root: Path | None,
    output_dir: str | Path,
    cfg: BaselineConfig | None = None,
    strategies: tuple[StrategyName, ...] = STRATEGY_NAMES,
    extracted: pd.DataFrame | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    results = {}
    for name in strategies:
        results[name] = run_strategy_imaging_oof(
            train_df,
            train_series_df,
            strategy=name,
            train_series_root=train_series_root,
            output_dir=out / name,
            cfg=cfg,
            extracted=extracted,
        )
    (out / "all_strategy_imaging_oof.json").write_text(json.dumps(results, indent=2, default=str))
    return results
