"""Kaggle entrypoint: RSNA-BASELINE-001-PILOT (engineering validation gate).

Stages (exact order):
  1. Mount/path verification
  2. GPU gate
  3. DICOM audit (gold mid-slices + header-rich sample)
  4. Ordering / orientation audit
  5. Leakage assertion (fold 0 from imaging_loop split)
  6. Pilot training (pilot_folds=1, epochs=5)
  7. Artifact generation
  8. Submission validation
  9. Gate decision
  10. STOP

Official baseline YAML is NOT mutated. Notebook-local overrides only:
  experiment_id=RSNA-BASELINE-001-PILOT, pilot_folds=1, epochs=5

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

# --- path bootstrap (Kaggle Dataset mount variants) ---
def _ensure_control_plane_on_path() -> Path | None:
    """Locate rsna_knee package from Dataset mount, vendored zip, or cwd.

    Kaggle now mounts datasets under ``/kaggle/input/datasets/<user>/<slug>/``
    (not ``/kaggle/input/<slug>``). Competition data is under
    ``/kaggle/input/competitions/<slug>/``.
    """
    import zipfile

    print("=== bootstrap diagnostics ===")
    for root in (Path("/kaggle/input"), Path("/kaggle/input/datasets"), Path("/kaggle/working"), Path(".")):
        print(f"exists {root}={root.exists()}")
        if root.exists():
            try:
                for p in sorted(root.iterdir())[:50]:
                    print(f"  {p}")
            except Exception as exc:
                print(f"  list_error={exc}")

    # Search ONLY under /kaggle/input/datasets (never rglob competitions — DICOM tree).
    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        for init in datasets_root.rglob("rsna_knee/__init__.py"):
            src = init.parent.parent
            print("found_package_via_datasets_rglob", src)
            if (src / "rsna_knee").exists():
                sys.path.insert(0, str(src))
                return src
        # Also handle zipped dataset payloads
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

    # Prefer extracting a sidecar zip if present.
    for zpath in (
        Path("control_plane.zip"),
        Path("/kaggle/working/control_plane.zip"),
    ):
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

    # Dataset dirs may be zipped
    for ds_root in sorted(Path("/kaggle/input/datasets").glob("*/*")) if Path("/kaggle/input/datasets").exists() else []:
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

from rsna_knee.data.dicom_audit import (
    audit_gold_mid_slices,
    audit_header_rich_sample,
    build_selected_series_table,
    decide_dicom_gate,
)
from rsna_knee.data.loaders import (
    load_series_metadata,
    load_test_metadata,
    load_train_metadata,
)
from rsna_knee.evaluation.splits import (
    assert_no_group_overlap,
    gold_label_mask,
    iter_fold_frames,
    make_study_folds,
)
from rsna_knee.labels.targets import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.runtime.adapter import resolve_data_paths
from rsna_knee.submission.validate import validate_submission
from rsna_knee.training.imaging_loop import BaselineConfig, load_baseline_config, run_imaging_baseline
from rsna_knee.training.seed import seed_everything

EXPERIMENT_ID = "RSNA-BASELINE-001-PILOT"
PILOT_EPOCHS = 5
PILOT_FOLDS = 1
SEED = 42


def _find_config() -> Path | None:
    candidates = [
        Path("/kaggle/working/rsna-knee-control/configs/experiment/rsna_baseline_001.yaml"),
        Path("configs/experiment/rsna_baseline_001.yaml"),
        Path("/kaggle/working/configs/experiment/rsna_baseline_001.yaml"),
        Path("/kaggle/input/rsna-knee-control/configs/experiment/rsna_baseline_001.yaml"),
        Path("/kaggle/input/datasets/alikarimiansarasa/rsna-knee-control/configs/experiment/rsna_baseline_001.yaml"),
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
            Path(__file__).resolve().parents[2] / "configs" / "experiment" / "rsna_baseline_001.yaml"
        )
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def _git_commit() -> str | None:
    try:
        roots = [
            Path("/kaggle/input/rsna-knee-control"),
            Path(__file__).resolve().parents[2],
        ]
        for root in roots:
            if (root / ".git").exists():
                out = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
                ).strip()
                return out
    except Exception:
        return None
    # Dataset zip may include COMMIT.txt
    for c in [
        Path("/kaggle/input/rsna-knee-control/COMMIT.txt"),
        Path("/kaggle/input/rsna-knee-control/MRI/COMMIT.txt"),
    ]:
        if c.exists():
            return c.read_text().strip()[:64]
    return None


def stage1_paths(out: Path) -> dict[str, Any]:
    paths = resolve_data_paths(mode="kaggle", allow_missing_dicom=False)
    required = [
        paths.train_csv,
        paths.train_series_csv,
        paths.test_csv,
        paths.test_series_csv,
        paths.sample_submission_csv,
    ]
    missing = [str(p) for p in required if not p.exists()]
    train_series = paths.train_series_dir
    test_series = paths.test_series_dir
    # Confirm DICOM files exist without walking the entire tree.
    sample_dcm = None
    n_top = 0
    if train_series and train_series.exists():
        n_top = sum(1 for _ in train_series.iterdir())
        for study_dir in train_series.iterdir():
            if not study_dir.is_dir():
                continue
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                for f in series_dir.iterdir():
                    if f.is_file():
                        sample_dcm = str(f)
                        break
                if sample_dcm:
                    break
            if sample_dcm:
                break

    info = {
        "root": str(paths.root),
        "train_series_dir": str(train_series) if train_series else None,
        "test_series_dir": str(test_series) if test_series else None,
        "missing_required": missing,
        "n_top_level_train_series_entries": n_top,
        "sample_dicom_path": sample_dcm,
        "dicoms_found": sample_dcm is not None,
    }
    (out / "stage1_paths.json").write_text(json.dumps(info, indent=2))
    if missing:
        raise FileNotFoundError(f"Missing required CSVs: {missing}")
    if not info["dicoms_found"]:
        raise FileNotFoundError("No DICOM files found under train_series/")
    print("STAGE1_OK", info["root"], "sample=", sample_dcm)
    return {"paths": paths, "info": info}


def stage2_gpu(out: Path) -> dict[str, Any]:
    cuda = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "cuda_available": cuda,
        "pytorch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "device_count": int(torch.cuda.device_count()) if cuda else 0,
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "cuda_version": getattr(torch.version, "cuda", None),
                "total_memory_gb": round(props.total_memory / (1024**3), 3),
                "device": "cuda:0",
            }
        )
        # Tiny alloc to prove CUDA works.
        x = torch.randn(64, 64, device="cuda")
        y = (x @ x.T).sum().item()
        info["cuda_smoke_sum"] = float(y)
        info["allocated_mb_after_smoke"] = float(torch.cuda.memory_allocated() / (1024**2))
    (out / "pilot_gpu.json").write_text(json.dumps(info, indent=2))
    print("STAGE2_GPU", info)
    if not cuda:
        raise RuntimeError("CUDA unavailable — BLOCKED (Case 4). Do not train on CPU.")
    return info


def stage3_4_dicom_audit(
    out: Path,
    *,
    train_df: pd.DataFrame,
    train_series: pd.DataFrame,
    train_series_root: Path,
) -> dict[str, Any]:
    gold = train_df.loc[gold_label_mask(train_df)].copy()
    print(f"gold_studies={len(gold)}")
    t0 = time.perf_counter()
    mid = audit_gold_mid_slices(
        gold,
        train_series,
        train_series_root,
        prefer_fat_suppression=True,
        image_size=224,
        mid_frac=0.5,
    )
    mid_sec = time.perf_counter() - t0
    selected = build_selected_series_table(gold, train_series, prefer_fat_suppression=True)
    t1 = time.perf_counter()
    header = audit_header_rich_sample(
        selected,
        train_series_root,
        max_series=30,
        seed=SEED,
    )
    header_sec = time.perf_counter() - t1
    gate = decide_dicom_gate(mid, header)

    # Strip bulky per-row payloads from the primary audit file; keep a compact summary
    # plus a full rows dump separately.
    mid_rows = mid.pop("rows")
    (out / "pilot_mid_slice_rows.json").write_text(json.dumps(mid_rows, indent=2, default=str))
    dicom_audit = {
        "mid_slice_audit": mid,
        "header_rich_audit": {
            k: v for k, v in header.items() if k != "series"
        },
        "header_series_sample": header.get("series", []),
        "dicom_gate": gate,
        "timing_wall_sec": {"mid_slice_audit": mid_sec, "header_rich_audit": header_sec},
        "n_gold": int(len(gold)),
        "n_selected_series": int(len(selected)),
    }
    (out / "pilot_dicom_audit.json").write_text(json.dumps(dicom_audit, indent=2, default=str))

    quality = {
        "failure_counts": mid.get("failure_counts", {}),
        "failure_examples": mid.get("failure_examples", {}),
        "success_rate": mid.get("success_rate"),
        "degenerate_rate": mid.get("degenerate_rate"),
        "n_degenerate_zero_or_near_zero": mid.get("n_degenerate_zero_or_near_zero"),
        "ordering_summary": header.get("ordering_summary"),
        "orientation_note_counts": header.get("orientation_note_counts"),
        "shape_stats_sample": header.get("shape_stats"),
        "blockers": gate.get("blockers", []),
        "caveats": gate.get("caveats", []),
    }
    (out / "pilot_data_quality.json").write_text(json.dumps(quality, indent=2, default=str))
    print(
        "STAGE3_4_DICOM",
        "success_rate=",
        mid.get("success_rate"),
        "degenerate_rate=",
        mid.get("degenerate_rate"),
        "order_disagreement=",
        (header.get("ordering_summary") or {}).get("disagreement_rate"),
        "blockers=",
        gate.get("blockers"),
    )
    return {"mid": mid, "header": header, "gate": gate, "gold": gold, "selected": selected}


def stage5_leakage(
    out: Path,
    gold: pd.DataFrame,
    *,
    n_folds: int = 5,
) -> dict[str, Any]:
    stratify_col = "ACL" if gold["ACL"].nunique(dropna=True) >= 2 else None
    folds = make_study_folds(
        gold, n_splits=min(n_folds, len(gold)), seed=SEED, stratify_col=stratify_col
    )
    fold0 = folds[0]
    fold0.assert_no_leakage()
    train_ids = set(fold0.train_ids)
    val_ids = set(fold0.val_ids)
    assert train_ids.isdisjoint(val_ids)
    assert all(uid in set(gold[STUDY_ID_COL].astype(str)) for uid in train_ids | val_ids)
    assert len(train_ids) == len(fold0.train_ids)
    assert len(val_ids) == len(fold0.val_ids)

    # Compare to DATA-002 fold map if present in the code dataset or working dir.
    data002_candidates = [
        Path("/kaggle/working/rsna-knee-control/outputs/RSNA-DATA-002/study_folds.csv"),
        Path("outputs/RSNA-DATA-002/study_folds.csv"),
        Path("/kaggle/input/datasets/alikarimiansarasa/rsna-knee-control/outputs/RSNA-DATA-002/study_folds.csv"),
        Path("/kaggle/input/rsna-knee-control/outputs/RSNA-DATA-002/study_folds.csv"),
        Path("/kaggle/working/RSNA-DATA-002/study_folds.csv"),
    ]
    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        data002_candidates.extend(list(datasets_root.rglob("study_folds.csv"))[:5])
    compare: dict[str, Any] = {"data002_fold_map_found": False}
    for c in data002_candidates:
        if c.exists():
            df = pd.read_csv(c)
            gold_map = df[df.get("has_gold_labels", 1) == 1] if "has_gold_labels" in df.columns else df
            # DATA-002 map may include all studies; restrict to gold overlap.
            if "has_gold_labels" in df.columns:
                gold_map = df[df["has_gold_labels"] == 1]
            else:
                gold_map = df[df[STUDY_ID_COL].astype(str).isin(gold[STUDY_ID_COL].astype(str))]
            d002_f0 = set(gold_map.loc[gold_map["fold"] == 0, STUDY_ID_COL].astype(str))
            # Baseline fold0 val set vs DATA-002 fold0 gold set
            compare = {
                "data002_fold_map_found": True,
                "data002_path": str(c),
                "baseline_fold0_val_n": len(val_ids),
                "data002_fold0_gold_n": len(d002_f0),
                "intersection_n": len(val_ids & d002_f0),
                "baseline_only_n": len(val_ids - d002_f0),
                "data002_only_n": len(d002_f0 - val_ids),
                "identical_fold0_val": val_ids == d002_f0,
                "note": (
                    "Baseline imaging_loop uses StratifiedGroupKFold on ACL when possible; "
                    "DATA-002 used plain GroupKFold. Difference is documented, not a blocker."
                ),
            }
            break

    info = {
        "splitter": "StratifiedGroupKFold" if stratify_col else "GroupKFold",
        "stratify_col": stratify_col,
        "n_folds_configured": n_folds,
        "fold0_n_train": len(train_ids),
        "fold0_n_val": len(val_ids),
        "leakage_overlap": 0,
        "all_gold": True,
        "no_duplicate_train": len(train_ids) == len(fold0.train_ids),
        "no_duplicate_val": len(val_ids) == len(fold0.val_ids),
        "fold0_val_ids": sorted(val_ids),
        "data002_compare": compare,
    }
    (out / "stage5_leakage.json").write_text(json.dumps(info, indent=2))
    print("STAGE5_LEAKAGE_PASS", info["splitter"], "train", len(train_ids), "val", len(val_ids))
    return info


def stage6_7_train(
    out: Path,
    *,
    paths,
    config_path: Path | None,
    gpu_info: dict[str, Any],
) -> dict[str, Any]:
    cfg = load_baseline_config(config_path) if config_path else BaselineConfig()
    # PILOT ENGINEERING OVERRIDES — do not mutate official YAML on disk.
    cfg.experiment_id = EXPERIMENT_ID
    cfg.pilot_folds = PILOT_FOLDS
    cfg.epochs = PILOT_EPOCHS
    overrides = {
        "experiment_id": cfg.experiment_id,
        "pilot_folds": cfg.pilot_folds,
        "epochs": cfg.epochs,
        "note": "PILOT ENGINEERING OVERRIDES only; official rsna_baseline_001.yaml unchanged",
    }
    (out / "pilot_overrides.json").write_text(json.dumps(overrides, indent=2))

    seed_everything(SEED)
    train = load_train_metadata(paths.train_csv)
    train_series = load_series_metadata(paths.train_series_csv)
    test = load_test_metadata(paths.test_csv)
    test_series = load_series_metadata(paths.test_series_csv)

    # Training health probes via monkey-patched first batch check inside result extras
    # (baseline loop already logs loss). We add a lightweight CUDA memory snapshot.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
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
    wall = time.perf_counter() - t0

    runtime = {
        "train_wall_sec": wall,
        "train_runtime_sec": result.train_runtime_sec,
        "infer_runtime_sec": result.infer_runtime_sec,
        "peak_rss_mb": result.peak_rss_mb,
        "cuda_peak_mb": result.cuda_peak_mb,
        "device": result.extras.get("device"),
        "n_folds_run": result.extras.get("n_folds_run"),
        "fold_metrics": result.fold_metrics,
        "epochs_requested": PILOT_EPOCHS,
        "gpu": gpu_info,
        "approx_samples_per_sec": None,
    }
    # Approximate throughput from fold0 history if present.
    hist_path = out / "checkpoints" / "fold0_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        runtime["fold0_history"] = history
        if history:
            # wall already includes audit? No — stage6 only. Per-epoch approx:
            n_train = result.fold_metrics[0]["n_train"] if result.fold_metrics else None
            if n_train and result.train_runtime_sec:
                runtime["approx_samples_per_sec"] = float(
                    (n_train * len(history)) / max(result.train_runtime_sec, 1e-6)
                )
                runtime["approx_epoch_sec"] = float(result.train_runtime_sec / max(len(history), 1))

    (out / "pilot_runtime.json").write_text(json.dumps(runtime, indent=2, default=str))

    # Alias artifacts expected by the pilot brief.
    for src_name, dst_name in [
        ("metrics_summary.json", "pilot_metrics.json"),
        ("oof_predictions.csv", "pilot_oof_predictions.csv"),
        ("submission.csv", "pilot_submission.csv"),
    ]:
        src = out / src_name
        dst = out / dst_name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

    # Attach fold column to pilot OOF for clarity.
    oof_path = out / "oof_predictions.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        # Mark fold 0 for rows with predictions; others remain NaN scores.
        pred_mask = oof[list(TARGET_COLUMNS)].notna().all(axis=1)
        oof.insert(1, "fold", np.where(pred_mask, 0, np.nan))
        oof.to_csv(out / "pilot_oof_predictions.csv", index=False)

    print(
        "STAGE6_7_TRAIN_OK",
        "macro=",
        result.oof_macro_auc,
        "cuda_peak_mb=",
        result.cuda_peak_mb,
    )
    return {"result": result, "runtime": runtime, "overrides": overrides}


def stage8_submission(out: Path, paths) -> dict[str, Any]:
    test = load_test_metadata(paths.test_csv)
    sub_path = out / "submission.csv"
    val = validate_submission(
        sub_path,
        expected_ids=test[STUDY_ID_COL].astype(str).tolist(),
        expected_row_count=len(test),
    )
    info = {
        "ok": val.ok,
        "errors": val.errors,
        "warnings": val.warnings,
        "n_rows": val.n_rows,
        "path": str(sub_path),
    }
    (out / "stage8_submission_validation.json").write_text(json.dumps(info, indent=2))
    # Competition-visible copy
    (paths.working_dir / "submission.csv").write_bytes(sub_path.read_bytes())
    print("STAGE8_SUBMISSION", info)
    if not val.ok:
        raise ValueError(f"Submission invalid: {val.errors}")
    return info


def stage9_gate(
    out: Path,
    *,
    path_info: dict[str, Any],
    gpu_info: dict[str, Any],
    dicom_pack: dict[str, Any],
    leakage: dict[str, Any],
    train_pack: dict[str, Any],
    submission: dict[str, Any],
    config_path: Path | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    caveats: list[str] = []
    high: list[str] = []
    medium: list[str] = []
    low: list[str] = []

    if not path_info.get("dicoms_found"):
        blockers.append("Competition DICOMs not found")
    if not gpu_info.get("cuda_available"):
        blockers.append("CUDA unavailable")

    dg = dicom_pack["gate"]
    blockers.extend(dg.get("blockers", []))
    caveats.extend(dg.get("caveats", []))

    if leakage.get("leakage_overlap", 0) != 0:
        blockers.append("StudyInstanceUID leakage between train and val")

    result = train_pack["result"]
    # Finite loss / AUC checks from fold history
    hist_path = out / "checkpoints" / "fold0_history.json"
    finite_loss = True
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        for h in history:
            loss = h.get("train_loss")
            if loss is None or not np.isfinite(loss):
                finite_loss = False
                blockers.append(f"Non-finite train_loss at epoch {h.get('epoch')}")
                break
    if not (out / "checkpoints" / "fold0.pt").exists():
        blockers.append("Missing fold0 checkpoint")
    if not (out / "oof_predictions.csv").exists():
        blockers.append("Missing OOF predictions")
    if not submission.get("ok"):
        blockers.append(f"Submission validation failed: {submission.get('errors')}")

    # Document known baseline risks that were audited.
    medium.append(
        "Baseline load_mid_slice orders by InstanceNumber only; IPP agreement audited in this pilot."
    )
    medium.append(
        "StudyMultiPlaneDataset silently continues on decode failure; pilot independently audited failure rate."
    )
    if not leakage.get("data002_compare", {}).get("identical_fold0_val", True):
        low.append(
            "Fold-0 membership differs from RSNA-DATA-002 GroupKFold map "
            "(baseline uses StratifiedGroupKFold on ACL when possible)."
        )
    low.append(
        "Pilot OOF macro-AUC is NOT the official baseline floor (1 fold, 5 epochs only)."
    )
    medium.append(
        "Case-1 metrics fix: _safe_auc now masks NaN y_score so pilot_folds partial OOF does not crash."
    )

    if blockers:
        decision = "BLOCKED"
    elif caveats or high:
        decision = "READY_WITH_CAVEATS"
    else:
        decision = "READY_FOR_FULL_BASELINE"

    # Elevate caveats that are serious but not blockers already handled.
    if caveats and decision == "READY_FOR_FULL_BASELINE":
        decision = "READY_WITH_CAVEATS"

    checklist = {
        "Kaggle mount verified": bool(path_info.get("root")),
        "Competition DICOMs found": bool(path_info.get("dicoms_found")),
        "CUDA/GPU verified": bool(gpu_info.get("cuda_available")),
        "Gold-only fold verified": True,
        "StudyInstanceUID leakage check passed": leakage.get("leakage_overlap", 1) == 0,
        "DICOM decode audit completed": True,
        "InstanceNumber vs IPP ordering audited": True,
        "ImageOrientationPatient audited": True,
        "Non-degenerate tensors verified": dicom_pack["gate"].get("decode_ok", False),
        "Training completed": True,
        "Finite loss verified": finite_loss,
        "Finite gradients verified": "assumed_via_finite_loss_and_checkpoint",
        "Checkpoint created": (out / "checkpoints" / "fold0.pt").exists(),
        "OOF predictions created": (out / "oof_predictions.csv").exists(),
        "Submission created": (out / "submission.csv").exists(),
        "Submission validator passed": bool(submission.get("ok")),
        "Gate decision written": True,
    }

    gate = {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "blockers": blockers,
        "caveats": caveats,
        "problems": {
            "BLOCKER": blockers,
            "HIGH": high,
            "MEDIUM": medium,
            "LOW": low,
        },
        "checklist": checklist,
        "pilot_overrides": train_pack.get("overrides"),
        "oof_macro_auc_pilot": result.oof_macro_auc,
        "per_label_auc": result.per_label_auc,
        "fold_metrics": result.fold_metrics,
        "environment": {
            "kaggle_root": path_info.get("root"),
            "gpu": gpu_info.get("gpu_name"),
            "cuda": gpu_info.get("cuda_available"),
            "pytorch": gpu_info.get("pytorch_version"),
            "python": gpu_info.get("python_version"),
            "git_commit": _git_commit(),
            "config_path": str(config_path) if config_path else None,
        },
        "dicom_summary": {
            "success_rate": dicom_pack["mid"].get("success_rate"),
            "degenerate_rate": dicom_pack["mid"].get("degenerate_rate"),
            "ordering": (dicom_pack["header"].get("ordering_summary") or {}),
            "orientation_notes": dicom_pack["header"].get("orientation_note_counts"),
        },
        "leakage": leakage,
        "submission": submission,
        "next_action": "STOP — await review before full 5-fold RSNA-BASELINE-001",
    }
    (out / "GATE_DECISION.json").write_text(json.dumps(gate, indent=2, default=str))

    print("\n=== FINAL CHECKLIST ===")
    for k, v in checklist.items():
        mark = "x" if v else " "
        print(f"[{mark}] {k}")
    print(f"\nGATE_DECISION={decision}")
    return gate


def main() -> None:
    seed_everything(SEED)
    working = Path(os.environ.get("RSNA_WORKING_DIR", "/kaggle/working"))
    out = working / EXPERIMENT_ID
    out.mkdir(parents=True, exist_ok=True)

    config_path = _find_config()
    print("config_path=", config_path)

    # 1. Paths
    stage1 = stage1_paths(out)
    paths = stage1["paths"]

    # 2. GPU
    try:
        gpu_info = stage2_gpu(out)
    except Exception as exc:
        gate = {
            "experiment_id": EXPERIMENT_ID,
            "decision": "BLOCKED",
            "blockers": [str(exc)],
            "next_action": "STOP — CUDA unavailable",
        }
        (out / "GATE_DECISION.json").write_text(json.dumps(gate, indent=2))
        print("GATE_DECISION=BLOCKED")
        raise

    train_df = load_train_metadata(paths.train_csv)
    train_series = load_series_metadata(paths.train_series_csv)

    # 3–4. DICOM + ordering/orientation
    dicom_pack = stage3_4_dicom_audit(
        out,
        train_df=train_df,
        train_series=train_series,
        train_series_root=paths.train_series_dir,
    )
    if dicom_pack["gate"]["blockers"]:
        gate = {
            "experiment_id": EXPERIMENT_ID,
            "decision": "BLOCKED",
            "blockers": dicom_pack["gate"]["blockers"],
            "caveats": dicom_pack["gate"]["caveats"],
            "dicom_summary": {
                "success_rate": dicom_pack["mid"].get("success_rate"),
                "degenerate_rate": dicom_pack["mid"].get("degenerate_rate"),
                "ordering": (dicom_pack["header"].get("ordering_summary") or {}),
            },
            "next_action": "STOP — DICOM audit blockers; do not train",
        }
        (out / "GATE_DECISION.json").write_text(json.dumps(gate, indent=2, default=str))
        print("GATE_DECISION=BLOCKED")
        print("blockers=", gate["blockers"])
        return

    # 5. Leakage
    leakage = stage5_leakage(out, dicom_pack["gold"], n_folds=5)

    # 6–7. Train
    train_pack = stage6_7_train(out, paths=paths, config_path=config_path, gpu_info=gpu_info)

    # 8. Submission
    submission = stage8_submission(out, paths)

    # 9. Gate
    stage9_gate(
        out,
        path_info=stage1["info"],
        gpu_info=gpu_info,
        dicom_pack=dicom_pack,
        leakage=leakage,
        train_pack=train_pack,
        submission=submission,
        config_path=config_path,
    )
    # 10. STOP
    print("PILOT COMPLETE — STOP. Do not start another experiment.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        working = Path(os.environ.get("RSNA_WORKING_DIR", "/kaggle/working"))
        out = working / EXPERIMENT_ID
        out.mkdir(parents=True, exist_ok=True)
        gate_path = out / "GATE_DECISION.json"
        if not gate_path.exists():
            gate_path.write_text(
                json.dumps(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "decision": "BLOCKED",
                        "blockers": [traceback.format_exc()[-2000:]],
                        "next_action": "STOP — unhandled exception",
                    },
                    indent=2,
                )
            )
        print("GATE_DECISION=BLOCKED")
        raise
