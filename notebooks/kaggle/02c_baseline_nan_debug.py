"""Kaggle entrypoint: RSNA-BASELINE-001-DEBUG-001 (NaN source diagnosis).

Replays the frozen pilot conditions (fold 0, 5 epochs, seed 42, same model /
preprocessing / split) with read-only probes at every stage.

ABSOLUTE RULES:
  - Do NOT mutate production imaging_loop / metrics / model / config.
  - Do NOT "fix" anything in this run.
  - Output is diagnosis only. Gate = BLOCKED_PENDING_FIX.

NEVER download competition DICOMs to a local Mac from this notebook.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _ensure_control_plane_on_path() -> Path | None:
    import zipfile

    print("=== bootstrap diagnostics ===")
    for root in (
        Path("/kaggle/input"),
        Path("/kaggle/input/datasets"),
        Path("/kaggle/working"),
        Path("."),
    ):
        print(f"exists {root}={root.exists()}")
        if root.exists():
            try:
                for p in sorted(root.iterdir())[:50]:
                    print(f"  {p}")
            except Exception as exc:
                print(f"  list_error={exc}")

    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        for init in datasets_root.rglob("rsna_knee/__init__.py"):
            src = init.parent.parent
            print("found_package_via_datasets_rglob", src)
            if (src / "rsna_knee").exists():
                sys.path.insert(0, str(src))
                return src
        for zpath in datasets_root.rglob("*.zip"):
            if zpath.name not in {"src.zip", "configs.zip", "control_plane.zip"} and "src" not in zpath.name:
                continue
            work = Path("/kaggle/working/rsna-knee-control")
            work.mkdir(parents=True, exist_ok=True)
            print("extracting dataset zip", zpath)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(work)
        for src in (
            Path("/kaggle/working/rsna-knee-control/src"),
            Path("/kaggle/working/rsna-knee-control"),
        ):
            if (src / "rsna_knee").exists():
                sys.path.insert(0, str(src))
                return src

    for zpath in (Path("control_plane.zip"), Path("/kaggle/working/control_plane.zip")):
        if zpath.exists():
            work = Path("/kaggle/working/rsna-knee-control")
            marker = work / "src" / "rsna_knee" / "__init__.py"
            if not marker.exists():
                print("extracting", zpath, "->", work)
                work.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(work)
            src = work / "src"
            if (src / "rsna_knee").exists():
                sys.path.insert(0, str(src))
                return src

    candidates = [
        Path("src"),
        Path("/kaggle/working/src"),
        Path("/kaggle/working/rsna-knee-control/src"),
        Path("/kaggle/input/rsna-knee-control/src"),
        Path("/kaggle/input/datasets/alikarimiansarasa/rsna-knee-control/src"),
        Path("/kaggle/input/rsna-knee/src"),
    ]
    try:
        candidates.append(Path(__file__).resolve().parents[2] / "src")
    except Exception:
        pass

    for p in candidates:
        print(f"check {p} -> {(p / 'rsna_knee').exists()}")
        if (p / "rsna_knee").exists():
            sys.path.insert(0, str(p))
            return p

    if Path("/kaggle/input/datasets").exists():
        for ds_root in sorted(Path("/kaggle/input/datasets").glob("*/*")):
            work = Path("/kaggle/working/rsna-knee-control")
            work.mkdir(parents=True, exist_ok=True)
            for zpath in sorted(ds_root.glob("*.zip")):
                print("extracting", zpath)
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(work)
            for src in (work / "src", ds_root / "src", ds_root):
                if (src / "rsna_knee").exists():
                    sys.path.insert(0, str(src))
                    return src
    return None


_SRC = _ensure_control_plane_on_path()
if _SRC is None:
    raise RuntimeError(
        "rsna_knee package not found under /kaggle/input/datasets or control_plane.zip. "
        "Attach dataset alikarimiansarasa/rsna-knee-control."
    )
print("using_src=", _SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from rsna_knee.data.loaders import (
    load_series_metadata,
    load_test_metadata,
    load_train_metadata,
)
from rsna_knee.datasets.imaging import StudyMultiPlaneDataset, collate_studies
from rsna_knee.evaluation.metrics import macro_roc_auc
from rsna_knee.evaluation.splits import (
    assert_no_group_overlap,
    gold_label_mask,
    iter_fold_frames,
    make_study_folds,
)
from rsna_knee.inference.predict import write_submission
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.models.backbone_25d import BackboneConfig, EfficientNetB0MultiPlane25D
from rsna_knee.runtime.adapter import resolve_data_paths
from rsna_knee.submission.validate import validate_submission
from rsna_knee.training.augment import AugmentConfig
from rsna_knee.training.imaging_loop import (
    BaselineConfig,
    _make_loader,
    _predict_test_mean_folds,
    load_baseline_config,
)
from rsna_knee.training.seed import seed_everything
from rsna_knee.utils.resources import pick_device

EXPERIMENT_ID = "RSNA-BASELINE-001-DEBUG-001"
PILOT_EPOCHS = 5
PILOT_FOLDS = 1
SEED = 42


def _find_config() -> Path | None:
    candidates = [
        Path("/kaggle/working/rsna-knee-control/configs/experiment/rsna_baseline_001.yaml"),
        Path("configs/experiment/rsna_baseline_001.yaml"),
        Path("/kaggle/working/configs/experiment/rsna_baseline_001.yaml"),
        Path("/kaggle/input/rsna-knee-control/configs/experiment/rsna_baseline_001.yaml"),
        Path(
            "/kaggle/input/datasets/alikarimiansarasa/rsna-knee-control/"
            "configs/experiment/rsna_baseline_001.yaml"
        ),
        Path("/kaggle/input/rsna-knee/configs/experiment/rsna_baseline_001.yaml"),
    ]
    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        candidates.extend(list(datasets_root.rglob("rsna_baseline_001.yaml"))[:5])
    work = Path("/kaggle/working")
    if work.exists():
        candidates.extend(list(work.rglob("rsna_baseline_001.yaml"))[:5])
    try:
        candidates.append(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "experiment"
            / "rsna_baseline_001.yaml"
        )
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, default=str))


def _tensor_health(t: torch.Tensor, name: str = "tensor") -> dict[str, Any]:
    x = t.detach().float().cpu()
    flat = x.reshape(-1)
    finite = torch.isfinite(flat)
    n = int(flat.numel())
    n_nan = int(torch.isnan(flat).sum().item())
    n_inf = int(torch.isinf(flat).sum().item())
    if finite.any():
        vals = flat[finite]
        stats = {
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        }
    else:
        stats = {"min": None, "max": None, "mean": None, "std": None}
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "n": n,
        "nan_count": n_nan,
        "inf_count": n_inf,
        "finite_count": int(finite.sum().item()),
        "all_finite": bool(torch.isfinite(t).all().item()),
        **stats,
    }


def _array_health(arr: np.ndarray, name: str = "array") -> dict[str, Any]:
    a = np.asarray(arr)
    flat = a.reshape(-1).astype(float, copy=False)
    n = int(flat.size)
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    finite = np.isfinite(flat)
    if finite.any():
        vals = flat[finite]
        stats = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()) if vals.size > 1 else 0.0,
        }
    else:
        stats = {"min": None, "max": None, "mean": None, "std": None}
    return {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "n": n,
        "nan_count": n_nan,
        "inf_count": n_inf,
        "finite_count": int(finite.sum()),
        "all_finite": bool(np.isfinite(a).all()),
        **stats,
    }


def _y_true_audit(y_true: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    unique = np.unique(yt[~np.isnan(yt)])
    pos = []
    neg = []
    for i, name in enumerate(TARGET_COLUMNS):
        col = yt[:, i]
        pos.append({"label": name, "pos": int(np.nansum(col == 1)), "neg": int(np.nansum(col == 0))})
        neg.append(int(np.nansum(col == 0)))
    return {
        **_array_health(yt, "y_true"),
        "unique_values": [float(u) for u in unique.tolist()],
        "positive_count_per_label": {p["label"]: p["pos"] for p in pos},
        "negative_count_per_label": {p["label"]: p["neg"] for p in pos},
        "assert_all_finite": bool(np.isfinite(yt).all()),
    }


def _label_status(y_true_col: np.ndarray, y_score_col: np.ndarray) -> str:
    if np.isnan(y_true_col).any():
        return "TARGET_NAN"
    if np.isnan(y_score_col).any():
        return "PREDICTION_NAN"
    if np.isinf(y_score_col).any():
        return "PREDICTION_INF"
    if np.unique(y_true_col).size < 2:
        return "ONE_CLASS"
    return "VALID"


def _per_label_metric_debug(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, Any]]:
    yt = np.asarray(y_true, dtype=float)
    ys = np.asarray(y_score, dtype=float)
    prod = macro_roc_auc(yt, ys)
    rows = []
    for i, name in enumerate(TARGET_COLUMNS):
        col_t = yt[:, i]
        col_s = ys[:, i]
        auc = prod.per_label_auc.get(name)
        rows.append(
            {
                "label": name,
                "positive_count": int(np.nansum(col_t == 1)),
                "negative_count": int(np.nansum(col_t == 0)),
                "nan_prediction_count": int(np.isnan(col_s).sum()),
                "inf_prediction_count": int(np.isinf(col_s).sum()),
                "auc": None if auc != auc else float(auc),
                "status": _label_status(col_t, col_s),
            }
        )
    return rows


def _param_buffer_health(model: nn.Module) -> dict[str, Any]:
    n_param_nan = 0
    n_param_inf = 0
    max_abs = 0.0
    n_buf_nan = 0
    n_buf_inf = 0
    max_abs_buf = 0.0
    nan_param_names: list[str] = []
    nan_buf_names: list[str] = []
    for name, p in model.named_parameters():
        if p is None:
            continue
        t = p.detach()
        if torch.isnan(t).any():
            n_param_nan += 1
            nan_param_names.append(name)
        if torch.isinf(t).any():
            n_param_inf += 1
        if t.numel():
            finite = t[torch.isfinite(t)]
            if finite.numel():
                max_abs = max(max_abs, float(finite.abs().max().item()))
    for name, b in model.named_buffers():
        if b is None or not torch.is_floating_point(b):
            continue
        t = b.detach()
        if torch.isnan(t).any():
            n_buf_nan += 1
            nan_buf_names.append(name)
        if torch.isinf(t).any():
            n_buf_inf += 1
        if t.numel():
            finite = t[torch.isfinite(t)]
            if finite.numel():
                max_abs_buf = max(max_abs_buf, float(finite.abs().max().item()))
    return {
        "n_param_tensors_with_nan": n_param_nan,
        "n_param_tensors_with_inf": n_param_inf,
        "max_abs_param": max_abs,
        "n_buffer_tensors_with_nan": n_buf_nan,
        "n_buffer_tensors_with_inf": n_buf_inf,
        "max_abs_buffer": max_abs_buf,
        "nan_param_names": nan_param_names[:20],
        "nan_buffer_names": nan_buf_names[:20],
    }


def _grad_health(model: nn.Module) -> dict[str, Any]:
    max_abs = 0.0
    sum_abs = 0.0
    n = 0
    n_nan = 0
    n_inf = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if torch.isnan(g).any():
            n_nan += 1
        if torch.isinf(g).any():
            n_inf += 1
        finite = g[torch.isfinite(g)]
        if finite.numel():
            max_abs = max(max_abs, float(finite.abs().max().item()))
            sum_abs += float(finite.abs().sum().item())
            n += int(finite.numel())
    return {
        "n_grad_tensors_with_nan": n_nan,
        "n_grad_tensors_with_inf": n_inf,
        "max_abs_grad": max_abs if n else None,
        "mean_abs_grad": (sum_abs / n) if n else None,
        "all_finite": n_nan == 0 and n_inf == 0,
    }


def _state_dict_health(state: dict[str, Any]) -> dict[str, Any]:
    n_nan = 0
    n_inf = 0
    max_abs = 0.0
    bad: list[str] = []
    for k, v in state.items():
        if not torch.is_tensor(v) or not torch.is_floating_point(v):
            continue
        t = v.detach()
        if torch.isnan(t).any():
            n_nan += 1
            bad.append(k)
        if torch.isinf(t).any():
            n_inf += 1
            if k not in bad:
                bad.append(k)
        finite = t[torch.isfinite(t)]
        if finite.numel():
            max_abs = max(max_abs, float(finite.abs().max().item()))
    return {
        "n_tensors_with_nan": n_nan,
        "n_tensors_with_inf": n_inf,
        "max_abs": max_abs,
        "bad_keys_sample": bad[:30],
        "healthy": n_nan == 0 and n_inf == 0,
    }


def _autocast_ctx(use_amp: bool, device: str):
    if use_amp and device.startswith("cuda"):
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except Exception:
            from torch.cuda.amp import autocast

            return autocast(enabled=True)
    from contextlib import nullcontext

    return nullcontext()


def predict_loader_instrumented(
    model,
    loader,
    *,
    device: str,
    use_amp: bool,
    collect_shadow_fp32: bool = True,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Mirror production _predict_loader with stage health probes.

    Returns (MetricResult, bundle, audit). The MetricResult uses the exact
    ndarray passed into production macro_roc_auc.
    """
    model.eval()
    uids: list[str] = []
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    logits_chunks: list[torch.Tensor] = []
    input_health: list[dict[str, Any]] = []
    logits_health_batches: list[dict[str, Any]] = []
    probs_health_batches: list[dict[str, Any]] = []
    shadow_probs: list[np.ndarray] = []
    dtypes: dict[str, Any] = {
        "amp_enabled": bool(use_amp and device.startswith("cuda")),
        "input_dtypes": [],
        "logit_dtypes": [],
        "prob_dtypes": [],
        "param_dtype_sample": None,
    }
    for name, p in model.named_parameters():
        dtypes["param_dtype_sample"] = str(p.dtype)
        break

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            dtypes["input_dtypes"].append(str(images.dtype))
            input_health.append(_tensor_health(images, "input"))
            with _autocast_ctx(use_amp, device):
                logits = model(images)
            dtypes["logit_dtypes"].append(str(logits.dtype))
            logits_health_batches.append(_tensor_health(logits, "logits"))
            logits_chunks.append(logits.detach().float().cpu())
            probs_t = torch.sigmoid(logits)
            dtypes["prob_dtypes"].append(str(probs_t.dtype))
            probs_health_batches.append(_tensor_health(probs_t, "probs"))
            probs = probs_t.detach().cpu().numpy()
            uids.extend(batch["study_uid"])
            probs_list.append(probs)
            labels_list.append(batch["labels"].numpy())

            if collect_shadow_fp32:
                # Shadow only — does not affect official AMP metric / OOF / checkpoint.
                with _autocast_ctx(False, device):
                    logits_fp32 = model(images.float())
                shadow_probs.append(torch.sigmoid(logits_fp32).detach().cpu().numpy())

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
    # Exact object passed into production metric.
    metrics = macro_roc_auc(y_true, y_score)

    logits_cat = (
        torch.cat(logits_chunks, dim=0)
        if logits_chunks
        else torch.zeros(0, len(TARGET_COLUMNS))
    )
    audit = {
        "y_true": _y_true_audit(y_true),
        "logits": _tensor_health(logits_cat, "logits_cat"),
        "probabilities": _array_health(y_score, "probs_passed_to_metric"),
        "prediction_array_passed_to_metric": _array_health(y_score, "y_score_exact"),
        "per_label": _per_label_metric_debug(y_true, y_score),
        "macro_auc": None if metrics.macro_auc != metrics.macro_auc else float(metrics.macro_auc),
        "evaluated_labels": list(metrics.evaluated_labels),
        "skipped_labels": list(metrics.skipped_labels),
        "dtypes": dtypes,
        "input_batch_health_first": input_health[0] if input_health else None,
        "logits_batch_health_first": logits_health_batches[0] if logits_health_batches else None,
        "probs_batch_health_first": probs_health_batches[0] if probs_health_batches else None,
        "assert_logits_finite": bool(torch.isfinite(logits_cat).all().item()) if logits_cat.numel() else True,
        "assert_probs_finite": bool(np.isfinite(y_score).all()),
        "assert_y_true_finite": bool(np.isfinite(y_true).all()),
    }
    if shadow_probs:
        ys_fp32 = np.concatenate(shadow_probs, axis=0)
        m_fp32 = macro_roc_auc(y_true, ys_fp32)
        audit["shadow_fp32"] = {
            "probabilities": _array_health(ys_fp32, "probs_fp32"),
            "macro_auc": None if m_fp32.macro_auc != m_fp32.macro_auc else float(m_fp32.macro_auc),
            "max_abs_diff_vs_amp": float(np.nanmax(np.abs(ys_fp32 - y_score)))
            if ys_fp32.size
            else None,
        }

    bundle = {"study_uid": uids, "y_score": y_score, "y_true": y_true, "logits": logits_cat.numpy()}
    return metrics, bundle, audit


def train_one_fold_debug(
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
    """Byte-for-byte same numeric steps as production _train_one_fold + probes."""
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    use_amp = bool(amp and device.startswith("cuda"))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_train = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except Exception:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler(enabled=use_amp)
        autocast_train = lambda: autocast(enabled=use_amp)

    best_auc = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    epoch_audits: list[dict[str, Any]] = []
    epoch0_arrays: dict[str, Any] | None = None
    epoch1_arrays: dict[str, Any] | None = None

    for epoch in range(epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_grad_rows: list[dict[str, Any]] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            weights = batch.get("weights")
            if weights is None:
                weights = torch.ones_like(labels)
            else:
                weights = weights.to(device)
            mask = ~torch.isnan(labels)
            labels = torch.where(mask, labels, torch.zeros_like(labels))
            sample_w = weights * mask.float()

            optim.zero_grad(set_to_none=True)
            with autocast_train():
                logits = model(images)
                loss_mat = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss = (loss_mat * sample_w).sum() / sample_w.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            gh = _grad_health(model)
            scale_before = float(scaler.get_scale()) if use_amp else 1.0
            found_inf = None
            if use_amp and hasattr(scaler, "_found_inf"):
                try:
                    found_inf = bool(scaler._found_inf.item())  # type: ignore[attr-defined]
                except Exception:
                    found_inf = None
            epoch_grad_rows.append(
                {
                    "loss": float(loss.item()),
                    "loss_finite": bool(np.isfinite(float(loss.item()))),
                    **gh,
                    "scaler_scale": scale_before,
                    "scaler_found_inf": found_inf,
                }
            )
            scaler.step(optim)
            scaler.update()
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()

        param_h = _param_buffer_health(model)
        metrics, bundle, val_audit = predict_loader_instrumented(
            model, val_loader, device=device, use_amp=use_amp, collect_shadow_fp32=True
        )
        macro = metrics.macro_auc
        train_loss = total_loss / max(n_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_auc": macro,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_macro_auc={macro} "
            f"logits_finite={val_audit['assert_logits_finite']} "
            f"probs_finite={val_audit['assert_probs_finite']}"
        )

        # Aggregate grad health across epoch batches
        grad_summary = {
            "n_batches": n_batches,
            "any_nan_grad": any(r["n_grad_tensors_with_nan"] > 0 for r in epoch_grad_rows),
            "any_inf_grad": any(r["n_grad_tensors_with_inf"] > 0 for r in epoch_grad_rows),
            "max_abs_grad": max(
                (r["max_abs_grad"] or 0.0 for r in epoch_grad_rows), default=None
            ),
            "mean_abs_grad_last": epoch_grad_rows[-1]["mean_abs_grad"] if epoch_grad_rows else None,
            "all_losses_finite": all(r["loss_finite"] for r in epoch_grad_rows),
            "scaler_scale_last": epoch_grad_rows[-1]["scaler_scale"] if epoch_grad_rows else None,
        }

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "logits_finite": val_audit["assert_logits_finite"],
            "probs_finite": val_audit["assert_probs_finite"],
            "y_true_finite": val_audit["assert_y_true_finite"],
            "pred_finite": val_audit["assert_probs_finite"],
            "valid_labels": len(val_audit["evaluated_labels"]),
            "skipped_labels": len(val_audit["skipped_labels"]),
            "macro_auc": None if macro != macro else float(macro),
            "logits_nan": val_audit["logits"]["nan_count"],
            "probs_nan": val_audit["probabilities"]["nan_count"],
            "param_nan_tensors": param_h["n_param_tensors_with_nan"],
            "buffer_nan_tensors": param_h["n_buffer_tensors_with_nan"],
            "max_abs_param": param_h["max_abs_param"],
            "grad_any_nan": grad_summary["any_nan_grad"],
            "grad_any_inf": grad_summary["any_inf_grad"],
            "max_abs_grad": grad_summary["max_abs_grad"],
        }
        epoch_audits.append(
            {
                "epoch_health": epoch_row,
                "validation_audit": val_audit,
                "param_buffer_health": param_h,
                "grad_summary": grad_summary,
                "study_uids": bundle["study_uid"],
            }
        )
        if epoch == 0:
            epoch0_arrays = {
                "y_true": bundle["y_true"].copy(),
                "y_score": bundle["y_score"].copy(),
                "logits": bundle["logits"].copy(),
                "study_uid": list(bundle["study_uid"]),
            }
        if epoch == 1:
            epoch1_arrays = {
                "y_true": bundle["y_true"].copy(),
                "y_score": bundle["y_score"].copy(),
                "logits": bundle["logits"].copy(),
                "study_uid": list(bundle["study_uid"]),
            }

        improved = macro == macro and macro > best_auc
        ckpt_decision = {
            "epoch": epoch,
            "macro": None if macro != macro else float(macro),
            "best_auc_before": best_auc,
            "macro_equals_macro": bool(macro == macro),
            "macro_gt_best": bool(macro == macro and macro > best_auc),
            "improved": bool(improved),
            "reason": (
                "saved_new_best"
                if improved
                else (
                    "nan_macro_rejected"
                    if macro != macro
                    else "macro_not_greater_than_best"
                )
            ),
        }
        epoch_audits[-1]["checkpoint_decision"] = ckpt_decision

        if improved:
            best_auc = float(macro)
            best_epoch = epoch
            stale = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "auc": best_auc}, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch={epoch} best_epoch={best_epoch}")
                break

    # Reload best — same as production
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
        "epoch_audits": epoch_audits,
        "epoch0_arrays": epoch0_arrays,
        "epoch1_arrays": epoch1_arrays,
        "criterion": "BCEWithLogitsLoss",
        "use_amp": use_amp,
    }


def compare_val_vs_oof(
    val_uids: list[str],
    val_scores: np.ndarray,
    oof: pd.DataFrame,
) -> dict[str, Any]:
    """Compare validation predictions to OOF rows for the same StudyInstanceUIDs."""
    cols = list(TARGET_COLUMNS)
    oof_by_uid = oof.set_index(oof[STUDY_ID_COL].astype(str))
    oof_rows = []
    missing = []
    for uid in val_uids:
        if uid not in oof_by_uid.index:
            missing.append(uid)
            oof_rows.append(np.full(len(cols), np.nan, dtype=float))
        else:
            oof_rows.append(oof_by_uid.loc[uid, cols].to_numpy(dtype=float))
    oof_mat = np.vstack(oof_rows) if oof_rows else np.zeros((0, len(cols)))
    val_mat = np.asarray(val_scores, dtype=float)

    same_shape = val_mat.shape == oof_mat.shape
    same_uid_order = missing == [] and list(val_uids) == [
        str(u) for u in val_uids
    ]  # order preserved by construction
    # UID order: validation order vs OOF subset reconstructed in validation order.
    same_uid_order = len(missing) == 0
    same_column_order = cols == list(TARGET_COLUMNS)

    finite_val = int(np.isfinite(val_mat).sum())
    finite_oof = int(np.isfinite(oof_mat).sum())
    nan_val = int(np.isnan(val_mat).sum())
    nan_oof = int(np.isnan(oof_mat).sum())

    if same_shape and finite_val and finite_oof:
        both = np.isfinite(val_mat) & np.isfinite(oof_mat)
        max_abs = float(np.max(np.abs(val_mat[both] - oof_mat[both]))) if both.any() else None
        identical = bool(both.all() and np.allclose(val_mat[both], oof_mat[both], equal_nan=False))
        if not both.all():
            identical = False
    else:
        max_abs = None
        identical = False
        if same_shape and nan_val == nan_oof == val_mat.size:
            identical = True  # both all-NaN
            max_abs = 0.0

    invariant_ok = finite_val == finite_oof
    return {
        "same_shape": same_shape,
        "same_uid_order": same_uid_order,
        "same_column_order": same_column_order,
        "finite_count_validation": finite_val,
        "finite_count_oof": finite_oof,
        "nan_count_validation": nan_val,
        "nan_count_oof": nan_oof,
        "max_abs_diff": max_abs,
        "identical": identical,
        "finite_count_invariant_holds": invariant_ok,
        "missing_uids_in_oof": missing,
        "n_val_rows": len(val_uids),
        "val_shape": list(val_mat.shape),
        "oof_subset_shape": list(oof_mat.shape),
    }


def independent_sklearn_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    ys = np.asarray(y_score, dtype=float)
    out: dict[str, Any] = {}
    for i, name in enumerate(TARGET_COLUMNS):
        mask = ~np.isnan(yt[:, i]) & ~np.isnan(ys[:, i])
        if mask.sum() < 2 or np.unique(yt[mask, i]).size < 2:
            out[name] = None
            continue
        try:
            out[name] = float(roc_auc_score(yt[mask, i], ys[mask, i]))
        except Exception as exc:
            out[name] = f"error:{exc}"
    vals = [v for v in out.values() if isinstance(v, float)]
    return {
        "per_label": out,
        "macro": float(np.mean(vals)) if vals else None,
        "n_evaluated": len(vals),
    }


def assign_oof_like_production(
    oof: pd.DataFrame,
    pred_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Exact production OOF assignment path with tracing."""
    matched = 0
    unmatched = []
    assignment_log = []
    before_finite = int(np.isfinite(oof[list(TARGET_COLUMNS)].to_numpy(dtype=float)).sum())
    for uid, score in zip(pred_bundle["study_uid"], pred_bundle["y_score"]):
        idx = oof[STUDY_ID_COL].astype(str) == str(uid)
        n_match = int(idx.sum())
        assignment_log.append(
            {
                "uid": str(uid),
                "n_match": n_match,
                "score_finite": int(np.isfinite(score).sum()),
                "score_nan": int(np.isnan(score).sum()),
                "score_shape": list(np.asarray(score).shape),
            }
        )
        if n_match == 0:
            unmatched.append(str(uid))
            continue
        # Exact production statement:
        oof.loc[idx, list(TARGET_COLUMNS)] = score
        matched += 1
    after = oof[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    after_finite = int(np.isfinite(after).sum())
    rows_populated = int(np.isfinite(after).all(axis=1).sum())
    return {
        "matched_uids": matched,
        "unmatched_uids": unmatched,
        "finite_cells_before": before_finite,
        "finite_cells_after": after_finite,
        "nan_cells_after": int(np.isnan(after).sum()),
        "rows_populated_all_labels_finite": rows_populated,
        "rows_expected": len(pred_bundle["study_uid"]),
        "oof_shape": list(oof.shape),
        "assignment_log_sample": assignment_log[:5],
        "assignment_log_all": assignment_log,
        # Inspect post-assignment values for first matched uid
        "post_assign_sample": None,
    }


def diagnose_root_cause(
    *,
    epoch_health_rows: list[dict[str, Any]],
    oof_cmp: dict[str, Any],
    oof_assign: dict[str, Any],
    ckpt_health: dict[str, Any],
    test_finite: int,
    val_best_finite: int,
    epoch_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    ep0 = epoch_health_rows[0] if epoch_health_rows else {}
    ep1 = epoch_health_rows[1] if len(epoch_health_rows) > 1 else {}

    # Determine first failure stage chronologically using evidence.
    first_failure = "UNKNOWN"
    root = "UNKNOWN"
    secondary = None
    confidence = "MEDIUM"

    # Stage order: inputs → logits → probs → val preds → metric → params/grads → oof → test
    if ep0.get("y_true_finite") is False:
        first_failure = "y_true"
        root = "DATA/LABEL_PROBLEM"
        confidence = "HIGH"
    elif not ep0.get("logits_finite", True):
        first_failure = "logits"
        root = "MODEL_NUMERICAL_INSTABILITY"
        confidence = "HIGH"
    elif not ep0.get("probs_finite", True):
        first_failure = "sigmoid_probabilities"
        root = "MODEL_NUMERICAL_INSTABILITY"
        confidence = "HIGH"
    elif ep1 and ep0.get("macro_auc") is not None and ep1.get("macro_auc") is None:
        # Epoch transition NaN
        if ep1.get("probs_finite") is False or (ep1.get("probs_nan") or 0) > 0:
            first_failure = "validation_probabilities_epoch1"
            if (ep1.get("param_nan_tensors") or 0) > 0 or (ep1.get("buffer_nan_tensors") or 0) > 0:
                root = "MODEL_NUMERICAL_INSTABILITY"
            elif ep1.get("grad_any_nan"):
                root = "MODEL_NUMERICAL_INSTABILITY"
            else:
                root = "VALIDATION_PIPELINE_BUG"
            confidence = "HIGH"
        elif ep1.get("logits_finite") is False:
            first_failure = "logits_epoch1"
            root = "MODEL_NUMERICAL_INSTABILITY"
            confidence = "HIGH"
        elif ep1.get("valid_labels", 1) == 0 and ep0.get("valid_labels", 0) > 0:
            # Class counts don't change — if only valid_labels changed with finite preds,
            # something odd; still check one-class carefully.
            if ep1.get("pred_finite") and ep0.get("pred_finite"):
                # Preds finite both epochs but all labels skipped → one-class? Unlikely to change.
                root = "METRIC_BUG"
                first_failure = "metric_epoch1"
                confidence = "MEDIUM"
            else:
                first_failure = "metric_epoch1"
                root = "ONE_CLASS_METRIC_LIMITATION"
                confidence = "LOW"  # must prove with evidence; default low
        else:
            first_failure = "metric_epoch1"
            root = "METRIC_BUG"
            confidence = "MEDIUM"

    # OOF vs val discrepancy (can be secondary or primary if val was fine)
    oof_broken = (oof_assign.get("finite_cells_after", 0) == 0) or (
        not oof_cmp.get("finite_count_invariant_holds", True)
    )
    val_ok_at_best = val_best_finite > 0

    causes = []
    if root != "UNKNOWN":
        causes.append(root)
    if oof_broken and val_ok_at_best:
        causes.append("OOF_ASSIGNMENT_BUG")
        if first_failure == "UNKNOWN":
            first_failure = "oof_assignment"
            root = "OOF_ASSIGNMENT_BUG"
            confidence = "HIGH"
        else:
            secondary = "OOF_ASSIGNMENT_BUG"
            confidence = "HIGH"
    if not ckpt_health.get("healthy", True):
        causes.append("CHECKPOINT_BUG")
        secondary = secondary or "CHECKPOINT_BUG"

    if len(causes) > 1:
        final_root = "MULTIPLE"
    else:
        final_root = root if root != "UNKNOWN" else (causes[0] if causes else "UNKNOWN")

    # Minimal proposed fix (NOT applied)
    proposed = "Await evidence-complete review before changing production."
    if "OOF_ASSIGNMENT_BUG" in causes or final_root == "OOF_ASSIGNMENT_BUG":
        proposed = (
            "Replace row-wise `oof.loc[idx, cols] = score` with an explicit "
            "column-aligned write (e.g. assign each TARGET_COLUMNS[i] from "
            "score[i], or build a DataFrame with columns=TARGET_COLUMNS). "
            "Do not change model/loss/AMP until OOF finite-count invariant holds."
        )
    elif final_root == "MODEL_NUMERICAL_INSTABILITY":
        proposed = (
            "Keep architecture frozen; first isolate whether NaNs are in parameters "
            "or BatchNorm buffers. Separate follow-up: try eval without AMP or "
            "reset/inspect BN stats — do not apply in this debug experiment."
        )
    elif final_root == "METRIC_BUG":
        proposed = (
            "Inspect production macro_roc_auc inputs at epoch 1; do not alter "
            "undefined→NaN contract. Fix only if inputs are finite and metric errs."
        )
    elif final_root == "VALIDATION_PIPELINE_BUG":
        proposed = (
            "Trace _predict_loader dtype/AMP path at epoch 1; keep training config "
            "frozen until the exact NaN injection line is confirmed."
        )

    # Epoch transition explanation
    transition = "insufficient_epochs"
    if ep0 and ep1:
        transition = (
            f"epoch0: logits_finite={ep0.get('logits_finite')} probs_finite={ep0.get('probs_finite')} "
            f"macro={ep0.get('macro_auc')} valid_labels={ep0.get('valid_labels')}; "
            f"epoch1: logits_finite={ep1.get('logits_finite')} probs_finite={ep1.get('probs_finite')} "
            f"macro={ep1.get('macro_auc')} valid_labels={ep1.get('valid_labels')} "
            f"probs_nan={ep1.get('probs_nan')} param_nan={ep1.get('param_nan_tensors')} "
            f"buffer_nan={ep1.get('buffer_nan_tensors')}"
        )

    return {
        "ROOT_CAUSE": final_root,
        "PRIMARY_CAUSE": root,
        "SECONDARY_CAUSE": secondary,
        "FIRST_FAILURE": first_failure,
        "EPOCH_TRANSITION": transition,
        "CONFIDENCE": confidence,
        "PROPOSED_MINIMAL_FIX": proposed,
        "test_finite": test_finite,
        "val_best_finite": val_best_finite,
        "oof_finite_after": oof_assign.get("finite_cells_after"),
        "finite_count_invariant_holds": oof_cmp.get("finite_count_invariant_holds"),
    }


def main() -> None:
    seed_everything(SEED)
    working = Path(os.environ.get("RSNA_WORKING_DIR", "/kaggle/working"))
    out = working / EXPERIMENT_ID
    out.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.perf_counter()

    config_path = _find_config()
    print("config_path=", config_path)

    # --- Paths / GPU ---
    paths = resolve_data_paths(mode="kaggle", allow_missing_dicom=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable — DEBUG requires Kaggle GPU")
    gpu_name = torch.cuda.get_device_properties(0).name
    print("GPU", gpu_name, "torch", torch.__version__)

    cfg = load_baseline_config(config_path) if config_path else BaselineConfig()
    cfg.experiment_id = EXPERIMENT_ID
    cfg.pilot_folds = PILOT_FOLDS
    cfg.epochs = PILOT_EPOCHS
    overrides = {
        "experiment_id": cfg.experiment_id,
        "pilot_folds": cfg.pilot_folds,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "note": "DEBUG overrides only; official rsna_baseline_001.yaml unchanged",
    }
    _write_json(out / "debug_overrides.json", overrides)

    train_df = load_train_metadata(paths.train_csv)
    train_series = load_series_metadata(paths.train_series_csv)
    test_df = load_test_metadata(paths.test_csv)
    test_series = load_series_metadata(paths.test_series_csv)

    gold = train_df.loc[gold_label_mask(train_df)].copy()
    print("n_gold=", len(gold))

    device = pick_device(cfg.device)
    amp = bool(cfg.amp and str(device).startswith("cuda"))
    print("device=", device, "amp=", amp)

    n_splits = min(cfg.n_folds, len(gold))
    stratify_col = "ACL" if gold["ACL"].nunique(dropna=True) >= 2 else None
    folds = make_study_folds(
        gold, n_splits=n_splits, seed=cfg.seed, stratify_col=stratify_col
    )
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

    # OOF init — exact production
    oof = gold[[STUDY_ID_COL]].copy()
    for col in TARGET_COLUMNS:
        oof[col] = np.nan
    oof_init_finite = int(np.isfinite(oof[list(TARGET_COLUMNS)].to_numpy(dtype=float)).sum())

    fold_i, tr, va = next(iter(iter_fold_frames(gold, folds)))
    assert_no_group_overlap(tr[STUDY_ID_COL], va[STUDY_ID_COL], context=f"fold {fold_i}")
    print(f"fold={fold_i} train={len(tr)} val={len(va)}")

    train_ds = StudyMultiPlaneDataset(
        tr,
        train_series,
        series_root=paths.train_series_dir,
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
        train_series,
        series_root=paths.train_series_dir,
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
        BackboneConfig(name=cfg.backbone, pretrained=cfg.pretrained, dropout=cfg.dropout)
    )
    ckpt_path = out / "checkpoints" / f"fold{fold_i}.pt"

    t_train0 = time.perf_counter()
    fold_info = train_one_fold_debug(
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
    train_sec = time.perf_counter() - t_train0

    # Post-reload validation (production OOF source)
    metrics, pred_bundle, post_audit = predict_loader_instrumented(
        model, val_loader, device=device, use_amp=amp, collect_shadow_fp32=True
    )

    # Fresh checkpoint load compare
    model_fresh = EfficientNetB0MultiPlane25D.build(
        BackboneConfig(name=cfg.backbone, pretrained=False, dropout=cfg.dropout)
    )
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    ckpt_health = _state_dict_health(state["model"])
    ckpt_health["saved_epoch"] = state.get("epoch")
    ckpt_health["saved_auc"] = state.get("auc")
    model_fresh.load_state_dict(state["model"])
    model_fresh.to(device)
    metrics_fresh, bundle_fresh, audit_fresh = predict_loader_instrumented(
        model_fresh, val_loader, device=device, use_amp=amp, collect_shadow_fp32=False
    )
    logits_diff = float(
        np.nanmax(np.abs(pred_bundle["logits"] - bundle_fresh["logits"]))
    )
    probs_diff = float(
        np.nanmax(np.abs(pred_bundle["y_score"] - bundle_fresh["y_score"]))
    )

    # OOF assignment — exact production loop
    oof_assign = assign_oof_like_production(oof, pred_bundle)
    # Sample post-assign values
    if pred_bundle["study_uid"]:
        uid0 = str(pred_bundle["study_uid"][0])
        row = oof.loc[oof[STUDY_ID_COL].astype(str) == uid0, list(TARGET_COLUMNS)]
        oof_assign["post_assign_sample"] = {
            "uid": uid0,
            "oof_row_values": row.to_numpy(dtype=float).tolist() if len(row) else None,
            "val_score": np.asarray(pred_bundle["y_score"][0], dtype=float).tolist(),
        }

    oof_cmp = compare_val_vs_oof(
        pred_bundle["study_uid"], pred_bundle["y_score"], oof
    )

    # OOF metrics (production)
    y_true_oof = gold.set_index(STUDY_ID_COL).loc[
        oof[STUDY_ID_COL], list(TARGET_COLUMNS)
    ].to_numpy(dtype=float)
    y_score_oof = oof[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    oof_metrics = macro_roc_auc(y_true_oof, y_score_oof)
    oof.to_csv(out / "oof_predictions.csv", index=False)

    # Test path — production mean-folds
    t_inf0 = time.perf_counter()
    test_probs = _predict_test_mean_folds(
        test_df=test_df,
        test_series_meta=test_series,
        test_series_root=paths.test_series_dir,
        ckpts=[ckpt_path],
        cfg=cfg,
        cache_dir=out / f"{cfg.cache_dirname}_test",
        device=device,
        amp=amp,
    )
    infer_sec = time.perf_counter() - t_inf0
    test_health = _array_health(test_probs, "test_probs")
    sub = test_df[[STUDY_ID_COL]].copy()
    for i, col in enumerate(TARGET_COLUMNS):
        sub[col] = test_probs[:, i]
    sub_path = out / "submission.csv"
    write_submission(
        sub, sub_path, expected_ids=test_df[STUDY_ID_COL].astype(str).tolist()
    )
    sub_val = validate_submission(
        sub_path,
        expected_ids=test_df[STUDY_ID_COL].astype(str).tolist(),
        expected_row_count=len(test_df),
    )

    # Metric diagnostic on epoch 0 / 1 arrays
    metric_diag: dict[str, Any] = {}
    for tag, arrays in (
        ("epoch0", fold_info.get("epoch0_arrays")),
        ("epoch1", fold_info.get("epoch1_arrays")),
    ):
        if arrays is None:
            metric_diag[tag] = None
            continue
        prod = macro_roc_auc(arrays["y_true"], arrays["y_score"])
        indep = independent_sklearn_auc(arrays["y_true"], arrays["y_score"])
        metric_diag[tag] = {
            "production": prod.to_dict(),
            "independent_sklearn": indep,
            "y_true_health": _array_health(arrays["y_true"], "y_true"),
            "y_score_health": _array_health(arrays["y_score"], "y_score"),
            "logits_health": _array_health(arrays["logits"], "logits"),
            "per_label_debug": _per_label_metric_debug(arrays["y_true"], arrays["y_score"]),
        }

    epoch_health_rows = [a["epoch_health"] for a in fold_info["epoch_audits"]]
    # Write epoch_health.csv
    eh = pd.DataFrame(epoch_health_rows)
    eh.to_csv(out / "epoch_health.csv", index=False)

    diagnosis = diagnose_root_cause(
        epoch_health_rows=epoch_health_rows,
        oof_cmp=oof_cmp,
        oof_assign=oof_assign,
        ckpt_health=ckpt_health,
        test_finite=test_health["finite_count"],
        val_best_finite=int(np.isfinite(pred_bundle["y_score"]).sum()),
        epoch_audits=fold_info["epoch_audits"],
    )

    # --- Artifacts ---
    validation_prediction_audit = {
        "post_reload_best_checkpoint": post_audit,
        "fresh_checkpoint_load": audit_fresh,
        "logits_max_abs_diff_reload_vs_fresh": logits_diff,
        "probs_max_abs_diff_reload_vs_fresh": probs_diff,
        "macro_auc_reload": None
        if metrics.macro_auc != metrics.macro_auc
        else float(metrics.macro_auc),
        "macro_auc_fresh": None
        if metrics_fresh.macro_auc != metrics_fresh.macro_auc
        else float(metrics_fresh.macro_auc),
        "per_epoch": [
            {
                "epoch": a["epoch_health"]["epoch"],
                "validation_audit": a["validation_audit"],
                "checkpoint_decision": a.get("checkpoint_decision"),
            }
            for a in fold_info["epoch_audits"]
        ],
    }
    _write_json(out / "validation_prediction_audit.json", validation_prediction_audit)

    oof_assignment_audit = {
        "initialization": {
            "shape": [len(gold), 1 + len(TARGET_COLUMNS)],
            "finite_cells": oof_init_finite,
            "expected_rows": len(gold),
            "init_value": "nan",
        },
        "assignment": oof_assign,
        "val_vs_oof_comparison": oof_cmp,
        "oof_metrics": oof_metrics.to_dict(),
        "validation_uids": pred_bundle["study_uid"],
        "fold": fold_i,
        "note": (
            "OOF filled AFTER best-checkpoint reload, same as production "
            "run_imaging_baseline. Per-epoch val AUC never writes OOF."
        ),
    }
    _write_json(out / "oof_assignment_audit.json", oof_assignment_audit)
    _write_json(out / "metric_diagnostic.json", metric_diag)

    checkpoint_health = {
        "path": str(ckpt_path),
        "state_dict": ckpt_health,
        "best_epoch": fold_info["best_epoch"],
        "best_auc": fold_info["best_auc"],
        "selection_rule": "macro == macro and macro > best_auc (NaN rejected)",
        "param_buffer_by_epoch": [
            {"epoch": a["epoch_health"]["epoch"], **a["param_buffer_health"]}
            for a in fold_info["epoch_audits"]
        ],
        "grad_by_epoch": [
            {"epoch": a["epoch_health"]["epoch"], **a["grad_summary"]}
            for a in fold_info["epoch_audits"]
        ],
        "reload_vs_fresh": {
            "logits_max_abs_diff": logits_diff,
            "probs_max_abs_diff": probs_diff,
            "macro_auc_reload": validation_prediction_audit["macro_auc_reload"],
            "macro_auc_fresh": validation_prediction_audit["macro_auc_fresh"],
        },
    }
    _write_json(out / "checkpoint_health.json", checkpoint_health)

    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "device": device,
        "gpu_name": gpu_name,
        "amp": amp,
        "train_runtime_sec": train_sec,
        "infer_runtime_sec": infer_sec,
        "wall_sec": time.perf_counter() - t_wall0,
        "n_train": len(tr),
        "n_val": len(va),
        "epochs": cfg.epochs,
        "seed": SEED,
        "fold": fold_i,
        "history": fold_info["history"],
        "cuda_peak_mb": float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available()
        else None,
        "pytorch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path) if config_path else None,
        "submission_ok": bool(sub_val.ok),
        "submission_errors": sub_val.errors,
    }
    _write_json(out / "runtime.json", runtime)

    # Summarize tensor health for nan_diagnosis
    def _last(field: str, default=None):
        if not epoch_health_rows:
            return default
        return epoch_health_rows[-1].get(field, default)

    nan_diagnosis = {
        **diagnosis,
        "LOGITS": "finite"
        if all(r.get("logits_finite") for r in epoch_health_rows)
        else "NaN_at_some_epoch",
        "PROBABILITIES": "finite"
        if all(r.get("probs_finite") for r in epoch_health_rows)
        else "NaN_at_some_epoch",
        "PARAMETERS": "finite"
        if all((r.get("param_nan_tensors") or 0) == 0 for r in epoch_health_rows)
        else "NaN_at_some_epoch",
        "BUFFERS": "finite"
        if all((r.get("buffer_nan_tensors") or 0) == 0 for r in epoch_health_rows)
        else "NaN_at_some_epoch",
        "GRADIENTS": "finite"
        if all(not r.get("grad_any_nan") for r in epoch_health_rows)
        else "NaN_at_some_epoch",
        "Y_TRUE": "finite" if all(r.get("y_true_finite") for r in epoch_health_rows) else "NaN",
        "VALIDATION_PREDICTIONS": "finite"
        if int(np.isfinite(pred_bundle["y_score"]).sum()) == pred_bundle["y_score"].size
        else "NaN",
        "OOF": "finite"
        if oof_assign["finite_cells_after"] == oof_assign["rows_expected"] * len(TARGET_COLUMNS)
        else "NaN",
        "METRIC_OOF": "valid"
        if oof_metrics.macro_auc == oof_metrics.macro_auc
        else "NaN",
        "TEST_PREDICTIONS": "finite" if test_health["all_finite"] else "NaN",
        "CHECKPOINT": "healthy" if ckpt_health["healthy"] else "corrupt",
        "GATE": "BLOCKED_PENDING_FIX",
        "NEXT_ACTION": "STOP — await review",
        "epoch_health_table": epoch_health_rows,
        "test_vs_oof": {
            "test_finite": test_health["finite_count"],
            "test_nan": test_health["nan_count"],
            "oof_finite": oof_assign["finite_cells_after"],
            "oof_nan": oof_assign["nan_cells_after"],
            "explanation_hint": (
                "Test uses column-wise sub[col]=probs[:,i]; OOF uses "
                "oof.loc[idx, cols]=score row assignment. Compare finite counts."
            ),
        },
        "amp_dtypes_epoch0": (
            fold_info["epoch_audits"][0]["validation_audit"]["dtypes"]
            if fold_info["epoch_audits"]
            else None
        ),
    }
    _write_json(out / "nan_diagnosis.json", nan_diagnosis)

    # Terminal summary (exact format)
    print("\n" + "=" * 60)
    print("RSNA-BASELINE-001-DEBUG-001")
    print()
    print(f"ROOT_CAUSE:\n{diagnosis['ROOT_CAUSE']}")
    print()
    print(f"FIRST_FAILURE:\n{diagnosis['FIRST_FAILURE']}")
    print()
    print(f"EPOCH_TRANSITION:\n{diagnosis['EPOCH_TRANSITION']}")
    print()
    print(f"LOGITS:\n{nan_diagnosis['LOGITS']}")
    print()
    print(f"PROBABILITIES:\n{nan_diagnosis['PROBABILITIES']}")
    print()
    print(f"PARAMETERS:\n{nan_diagnosis['PARAMETERS']}")
    print()
    print(f"GRADIENTS:\n{nan_diagnosis['GRADIENTS']}")
    print()
    print(f"Y_TRUE:\n{nan_diagnosis['Y_TRUE']}")
    print()
    print(f"VALIDATION_PREDICTIONS:\n{nan_diagnosis['VALIDATION_PREDICTIONS']}")
    print()
    print(f"OOF:\n{nan_diagnosis['OOF']}")
    print()
    print(f"METRIC:\n{nan_diagnosis['METRIC_OOF']}")
    print()
    print(f"TEST_PREDICTIONS:\n{nan_diagnosis['TEST_PREDICTIONS']}")
    print()
    print(f"CHECKPOINT:\n{nan_diagnosis['CHECKPOINT']}")
    print()
    print(f"CONFIDENCE:\n{diagnosis['CONFIDENCE']}")
    print()
    print(f"PROPOSED_MINIMAL_FIX:\n{diagnosis['PROPOSED_MINIMAL_FIX']}")
    print()
    print("GATE:\nBLOCKED_PENDING_FIX")
    print()
    print("NEXT_ACTION:\nSTOP — await review")
    print("=" * 60)

    # Persist terminal summary
    (out / "terminal_summary.txt").write_text(
        "\n".join(
            [
                "RSNA-BASELINE-001-DEBUG-001",
                "",
                f"ROOT_CAUSE:\n{diagnosis['ROOT_CAUSE']}",
                "",
                f"FIRST_FAILURE:\n{diagnosis['FIRST_FAILURE']}",
                "",
                f"EPOCH_TRANSITION:\n{diagnosis['EPOCH_TRANSITION']}",
                "",
                f"LOGITS:\n{nan_diagnosis['LOGITS']}",
                "",
                f"PROBABILITIES:\n{nan_diagnosis['PROBABILITIES']}",
                "",
                f"PARAMETERS:\n{nan_diagnosis['PARAMETERS']}",
                "",
                f"GRADIENTS:\n{nan_diagnosis['GRADIENTS']}",
                "",
                f"Y_TRUE:\n{nan_diagnosis['Y_TRUE']}",
                "",
                f"VALIDATION_PREDICTIONS:\n{nan_diagnosis['VALIDATION_PREDICTIONS']}",
                "",
                f"OOF:\n{nan_diagnosis['OOF']}",
                "",
                f"METRIC:\n{nan_diagnosis['METRIC_OOF']}",
                "",
                f"TEST_PREDICTIONS:\n{nan_diagnosis['TEST_PREDICTIONS']}",
                "",
                f"CHECKPOINT:\n{nan_diagnosis['CHECKPOINT']}",
                "",
                f"CONFIDENCE:\n{diagnosis['CONFIDENCE']}",
                "",
                f"PROPOSED_MINIMAL_FIX:\n{diagnosis['PROPOSED_MINIMAL_FIX']}",
                "",
                "GATE:\nBLOCKED_PENDING_FIX",
                "",
                "NEXT_ACTION:\nSTOP — await review",
            ]
        )
    )
    print("artifacts_written=", out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
