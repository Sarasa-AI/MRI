"""Imaging CV training loop for RSNA-BASELINE-001 (Kaggle-oriented)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rsna_knee.datasets.imaging import StudyMultiPlaneDataset, collate_studies
from rsna_knee.evaluation.analysis import (
    fold_stability,
    infer_bottleneck,
    label_correlation,
    leakage_checklist,
    strongest_weakest_labels,
)
from rsna_knee.evaluation.metrics import macro_roc_auc
from rsna_knee.evaluation.splits import (
    assert_no_group_overlap,
    gold_label_mask,
    iter_fold_frames,
    make_study_folds,
)
from rsna_knee.inference.predict import write_submission
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.models.backbone_25d import BackboneConfig, EfficientNetB0MultiPlane25D, count_parameters
from rsna_knee.training.augment import AugmentConfig
from rsna_knee.training.seed import seed_everything
from rsna_knee.utils.experiment import ExperimentCard, save_experiment_card
from rsna_knee.utils.logging import append_results_row, get_logger
from rsna_knee.utils.resources import measure_resources, pick_device

logger = get_logger(__name__)


@dataclass
class BaselineConfig:
    experiment_id: str = "RSNA-BASELINE-001"
    seed: int = 42
    n_folds: int = 5
    pilot_folds: int | None = None
    image_size: int = 224
    prefer_fat_suppression: bool = True
    mid_frac: float = 0.5
    cache_dirname: str = "cache_midplane_224"
    num_workers: int = 0
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    epochs: int = 25
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    hflip_p: float = 0.5
    rotate_deg: float = 10.0
    brightness: float = 0.15
    contrast: float = 0.15
    scale_min: float = 0.90
    scale_max: float = 1.00
    device: str = "auto"
    # When True, never touch real DICOM roots (Mac safety).
    force_synthetic: bool = False


@dataclass
class BaselineResult:
    experiment_id: str
    oof_macro_auc: float
    per_label_auc: dict[str, float]
    fold_metrics: list[dict[str, Any]]
    train_runtime_sec: float
    infer_runtime_sec: float
    peak_rss_mb: float | None
    cuda_peak_mb: float | None
    output_dir: str
    conclusion: str
    extras: dict[str, Any] = field(default_factory=dict)


def load_baseline_config(path: str | Path | None = None, **overrides: Any) -> BaselineConfig:
    cfg = BaselineConfig()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        mapping = {
            "experiment_id": raw.get("experiment_id"),
            "seed": raw.get("seed"),
            "n_folds": raw.get("n_folds"),
            "pilot_folds": raw.get("pilot_folds"),
            "image_size": (raw.get("data") or {}).get("image_size"),
            "prefer_fat_suppression": (raw.get("data") or {}).get("prefer_fat_suppression"),
            "mid_frac": (raw.get("data") or {}).get("mid_slice_frac"),
            "cache_dirname": (raw.get("data") or {}).get("cache_dirname"),
            "num_workers": (raw.get("data") or {}).get("num_workers"),
            "backbone": (raw.get("model") or {}).get("backbone"),
            "pretrained": (raw.get("model") or {}).get("pretrained"),
            "dropout": (raw.get("model") or {}).get("dropout"),
            "epochs": (raw.get("train") or {}).get("epochs"),
            "batch_size": (raw.get("train") or {}).get("batch_size"),
            "lr": (raw.get("train") or {}).get("lr"),
            "weight_decay": (raw.get("train") or {}).get("weight_decay"),
            "amp": (raw.get("train") or {}).get("amp"),
            "grad_clip": (raw.get("train") or {}).get("grad_clip"),
            "early_stopping_patience": (raw.get("train") or {}).get("early_stopping_patience"),
            "hflip_p": (raw.get("augmentation") or {}).get("hflip_p"),
            "rotate_deg": (raw.get("augmentation") or {}).get("rotate_deg"),
            "brightness": (raw.get("augmentation") or {}).get("brightness"),
            "contrast": (raw.get("augmentation") or {}).get("contrast"),
            "scale_min": (raw.get("augmentation") or {}).get("scale_min"),
            "scale_max": (raw.get("augmentation") or {}).get("scale_max"),
            "device": (raw.get("runtime") or {}).get("device"),
        }
        for k, v in mapping.items():
            if v is not None:
                setattr(cfg, k, v)
    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg


def _make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_studies,
        pin_memory=torch.cuda.is_available(),
    )


def _train_one_fold(
    model,
    train_loader,
    val_loader,
    *,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    grad_clip: float,
    patience: int,
    ckpt_path: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    use_amp = bool(amp and device.startswith("cuda"))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except Exception:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler(enabled=use_amp)
        autocast_ctx = lambda: autocast(enabled=use_amp)

    best_auc = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    for epoch in range(epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            weights = batch.get("weights")
            if weights is None:
                weights = torch.ones_like(labels)
            else:
                weights = weights.to(device)
            # Skip NaN labels defensively (gold rows should be complete).
            mask = ~torch.isnan(labels)
            labels = torch.where(mask, labels, torch.zeros_like(labels))
            sample_w = weights * mask.float()

            optim.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(images)
                # Masked + confidence-weighted BCE.
                loss_mat = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss = (loss_mat * sample_w).sum() / sample_w.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()

        val_metrics, _ = _predict_loader(model, val_loader, device=device, use_amp=use_amp)
        macro = val_metrics.macro_auc
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "val_macro_auc": macro,
            }
        )
        logger.info(
            "epoch=%s train_loss=%.4f val_macro_auc=%s",
            epoch,
            history[-1]["train_loss"],
            macro,
        )
        improved = macro == macro and macro > best_auc
        if improved:
            best_auc = float(macro)
            best_epoch = epoch
            stale = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "auc": best_auc}, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stopping at epoch=%s best_epoch=%s", epoch, best_epoch)
                break

    # Reload best
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])

    return {
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "history": history,
        "criterion": "BCEWithLogitsLoss",
    }


def _predict_loader(model, loader, *, device: str, use_amp: bool):
    import torch

    model.eval()
    uids: list[str] = []
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    def _autocast():
        if use_amp and device.startswith("cuda"):
            try:
                return torch.amp.autocast("cuda", enabled=True)
            except Exception:
                from torch.cuda.amp import autocast

                return autocast(enabled=True)
        from contextlib import nullcontext

        return nullcontext()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            with _autocast():
                logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            uids.extend(batch["study_uid"])
            probs_list.append(probs)
            labels_list.append(batch["labels"].numpy())
    y_score = (
        np.concatenate(probs_list, axis=0)
        if probs_list
        else np.zeros((0, len(TARGET_COLUMNS)))
    )
    y_true = (
        np.concatenate(labels_list, axis=0)
        if labels_list
        else np.zeros((0, len(TARGET_COLUMNS)))
    )
    metrics = macro_roc_auc(y_true, y_score)
    return metrics, {"study_uid": uids, "y_score": y_score, "y_true": y_true}


def run_imaging_baseline(
    train_df: pd.DataFrame,
    train_series_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_series_df: pd.DataFrame,
    *,
    train_series_root: Path | None,
    test_series_root: Path | None,
    output_dir: str | Path,
    cfg: BaselineConfig | None = None,
    config_path: str | Path | None = None,
) -> BaselineResult:
    """Run leakage-safe study-level 5-fold imaging baseline.

    On Mac / metadata-only: pass ``train_series_root=None`` (or
    ``cfg.force_synthetic=True``) to exercise the loop on synthetic tensors —
    never downloads competition DICOMs.
    """
    cfg = cfg or load_baseline_config(config_path)
    seed_everything(cfg.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.force_synthetic:
        train_series_root = None
        test_series_root = None

    device = pick_device(cfg.device)
    # AMP only meaningful on CUDA; keep flag but disable elsewhere.
    amp = bool(cfg.amp and device.startswith("cuda"))

    gold = train_df.loc[gold_label_mask(train_df)].copy()
    if gold.empty:
        raise ValueError("No gold-labeled studies available for imaging baseline")

    n_splits = min(cfg.n_folds, len(gold))
    stratify_col = "ACL" if gold["ACL"].nunique(dropna=True) >= 2 else None
    folds = make_study_folds(
        gold, n_splits=n_splits, seed=cfg.seed, stratify_col=stratify_col
    )
    # Leakage audit across all folds
    n_overlaps = 0
    for f in folds:
        try:
            f.assert_no_leakage()
        except AssertionError:
            n_overlaps += 1
    if cfg.pilot_folds is not None:
        folds = folds[: max(1, int(cfg.pilot_folds))]

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

    fold_metrics: list[dict[str, Any]] = []
    fold_ckpts: list[Path] = []
    train_t0 = time.perf_counter()
    peak_rss = None
    cuda_peak = None

    import torch

    with measure_resources(device=device) as train_res:
        for fold_i, tr, va in iter_fold_frames(gold, folds):
            assert_no_group_overlap(tr[STUDY_ID_COL], va[STUDY_ID_COL], context=f"fold {fold_i}")
            logger.info(
                "fold=%s train=%s val=%s device=%s",
                fold_i,
                len(tr),
                len(va),
                device,
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
            )
            val_ds = StudyMultiPlaneDataset(
                va,
                train_series_df,
                series_root=train_series_root,
                cache_dir=cache_dir,
                image_size=cfg.image_size,
                prefer_fat_suppression=cfg.prefer_fat_suppression,
                mid_frac=cfg.mid_frac,
                train=False,
                seed=cfg.seed,
            )
            train_loader = _make_loader(
                train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
            )
            val_loader = _make_loader(
                val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
            )

            model = EfficientNetB0MultiPlane25D.build(
                BackboneConfig(
                    name=cfg.backbone,
                    pretrained=cfg.pretrained,
                    dropout=cfg.dropout,
                )
            )
            ckpt_path = out / "checkpoints" / f"fold{fold_i}.pt"
            fold_info = _train_one_fold(
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
                ckpt_path=ckpt_path,
            )
            metrics, pred_bundle = _predict_loader(
                model, val_loader, device=device, use_amp=amp
            )
            # Write OOF
            for uid, score in zip(pred_bundle["study_uid"], pred_bundle["y_score"]):
                idx = oof[STUDY_ID_COL].astype(str) == str(uid)
                oof.loc[idx, list(TARGET_COLUMNS)] = score

            fold_metrics.append(
                {
                    "fold": fold_i,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "best_epoch": fold_info["best_epoch"],
                    "best_auc": fold_info["best_auc"],
                    **metrics.to_dict(),
                }
            )
            fold_ckpts.append(ckpt_path)
            (out / "checkpoints" / f"fold{fold_i}_history.json").write_text(
                json.dumps(fold_info["history"], indent=2)
            )

    train_runtime = time.perf_counter() - train_t0
    peak_rss = train_res.get("peak_rss_mb")
    cuda_peak = train_res.get("cuda_peak_allocated_mb")

    # OOF metrics
    y_true = gold.set_index(STUDY_ID_COL).loc[oof[STUDY_ID_COL], list(TARGET_COLUMNS)].to_numpy(
        dtype=float
    )
    y_score = oof[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    oof_metrics = macro_roc_auc(y_true, y_score)
    oof_path = out / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    # Inference on test (mean of fold checkpoints)
    infer_t0 = time.perf_counter()
    with measure_resources(device=device) as infer_res:
        test_probs = _predict_test_mean_folds(
            test_df=test_df,
            test_series_meta=test_series_df,
            test_series_root=test_series_root,
            ckpts=fold_ckpts,
            cfg=cfg,
            cache_dir=out / f"{cfg.cache_dirname}_test",
            device=device,
            amp=amp,
        )
    infer_runtime = time.perf_counter() - infer_t0

    sub = test_df[[STUDY_ID_COL]].copy()
    for i, col in enumerate(TARGET_COLUMNS):
        sub[col] = test_probs[:, i]
    submission_path = out / "submission.csv"
    write_submission(sub, submission_path, expected_ids=test_df[STUDY_ID_COL].astype(str).tolist())

    # Analysis
    corr = label_correlation(y_score)
    corr.to_csv(out / "oof_label_correlation.csv")
    max_offdiag = float(
        np.nanmax(np.abs(corr.to_numpy() - np.eye(len(TARGET_COLUMNS))))
    ) if corr.shape[0] else float("nan")
    macros = [m["macro_auc"] for m in fold_metrics]
    stability = fold_stability(macros)
    sw = strongest_weakest_labels(oof_metrics.per_label_auc)
    bottleneck = infer_bottleneck(
        n_gold=len(gold),
        fold_std=stability.get("std", float("nan")),
        max_abs_pred_corr=max_offdiag if max_offdiag == max_offdiag else 0.0,
        oof_macro=oof_metrics.macro_auc,
    )
    leakage = leakage_checklist(
        n_fold_overlaps=n_overlaps,
        report_used_at_inference=False,
        patient_id_available=False,
    )

    conclusion = (
        f"OOF macro AUC={oof_metrics.macro_auc:.4f} on n_gold={len(gold)}. "
        f"Likely bottleneck: {bottleneck}. "
        "No pseudo-labels / ensembles / HP search in this run."
    )

    card = ExperimentCard(
        experiment_id=cfg.experiment_id,
        configuration=str(config_path) if config_path else "BaselineConfig",
        seed=cfg.seed,
        model=f"EfficientNetB0MultiPlane25D/{cfg.backbone}",
        input_geometry=f"multiplane_midslice_{cfg.image_size}",
        data_selection="gold_labels_only",
        labels="official_12_targets",
        folds=f"GroupKFold n_splits={len(folds)} by StudyInstanceUID",
        augmentation="hflip_with_compartment_swap+rotate+brightness_contrast",
        optimizer=f"AdamW lr={cfg.lr} wd={cfg.weight_decay}",
        scheduler="CosineAnnealingLR",
        metric=f"macro_roc_auc={oof_metrics.macro_auc:.6f}",
        runtime=(
            f"train_sec={train_runtime:.1f}; infer_sec={infer_runtime:.1f}; "
            f"device={device}; peak_rss_mb={peak_rss}; cuda_peak_mb={cuda_peak}"
        ),
        oof_predictions=str(oof_path),
        conclusion=conclusion,
        status="complete",
        notes=f"n_params≈{count_parameters(EfficientNetB0MultiPlane25D.build(BackboneConfig(pretrained=False)))}",
    )
    save_experiment_card(card, out / "experiment_card.json")

    summary = {
        "experiment_id": cfg.experiment_id,
        "n_gold": int(len(gold)),
        "n_folds_run": len(folds),
        "oof_macro_auc": oof_metrics.macro_auc,
        "per_label_auc": oof_metrics.per_label_auc,
        "fold_metrics": fold_metrics,
        "fold_stability": stability,
        "strongest_weakest": sw,
        "max_abs_offdiag_pred_corr": max_offdiag,
        "bottleneck": bottleneck,
        "leakage": leakage,
        "train_runtime_sec": train_runtime,
        "infer_runtime_sec": infer_runtime,
        "infer_resource": infer_res,
        "peak_rss_mb": peak_rss,
        "cuda_peak_mb": cuda_peak,
        "device": device,
        "submission_path": str(submission_path),
        "config": asdict(cfg),
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=True))

    results_path = out / "results_row.csv"
    repo_results = Path(__file__).resolve().parents[3] / "experiments" / "results" / "results_table.csv"
    if repo_results.parent.exists():
        results_path = repo_results
    append_results_row(
        {
            "experiment_id": cfg.experiment_id,
            "macro_auc_oof": oof_metrics.macro_auc,
            "n_folds": len(folds),
            "n_gold": len(gold),
            "seed": cfg.seed,
            "image_size": cfg.image_size,
            "backbone": cfg.backbone,
            "train_runtime_sec": round(train_runtime, 2),
            "infer_runtime_sec": round(infer_runtime, 2),
            "device": device,
        },
        path=results_path,
    )

    return BaselineResult(
        experiment_id=cfg.experiment_id,
        oof_macro_auc=float(oof_metrics.macro_auc),
        per_label_auc=dict(oof_metrics.per_label_auc),
        fold_metrics=fold_metrics,
        train_runtime_sec=float(train_runtime),
        infer_runtime_sec=float(infer_runtime),
        peak_rss_mb=peak_rss,
        cuda_peak_mb=cuda_peak,
        output_dir=str(out),
        conclusion=conclusion,
        extras=summary,
    )


def _predict_test_mean_folds(
    *,
    test_df: pd.DataFrame,
    test_series_meta: pd.DataFrame,
    test_series_root: Path | None,
    ckpts: list[Path],
    cfg: BaselineConfig,
    cache_dir: Path,
    device: str,
    amp: bool,
) -> np.ndarray:
    import torch

    ds = StudyMultiPlaneDataset(
        test_df,
        test_series_meta,
        series_root=test_series_root,
        cache_dir=cache_dir,
        image_size=cfg.image_size,
        prefer_fat_suppression=cfg.prefer_fat_suppression,
        mid_frac=cfg.mid_frac,
        train=False,
        seed=cfg.seed,
        inference=True,
    )
    loader = _make_loader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    acc = None
    n = 0
    for ckpt in ckpts:
        model = EfficientNetB0MultiPlane25D.build(
            BackboneConfig(name=cfg.backbone, pretrained=False, dropout=cfg.dropout)
        )
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
        _metrics, bundle = _predict_loader(model, loader, device=device, use_amp=amp)
        if acc is None:
            acc = bundle["y_score"].astype(np.float64)
        else:
            acc += bundle["y_score"].astype(np.float64)
        n += 1
    if acc is None or n == 0:
        return np.full((len(test_df), len(TARGET_COLUMNS)), 0.5, dtype=np.float32)
    return (acc / n).astype(np.float32)
