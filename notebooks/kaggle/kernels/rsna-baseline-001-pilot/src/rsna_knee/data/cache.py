"""Disk cache for preprocessed mid-plane study tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def cache_path(cache_dir: Path, study_uid: str) -> Path:
    safe = str(study_uid).replace("/", "_")
    return Path(cache_dir) / f"{safe}.npz"


def load_cached(cache_dir: Path, study_uid: str) -> np.ndarray | None:
    path = cache_path(cache_dir, study_uid)
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["image"].astype(np.float32)


def save_cached(cache_dir: Path, study_uid: str, image: np.ndarray) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, study_uid)
    # Quantize to uint8 for compact cache; reload scales back to [0,1].
    u8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    np.savez_compressed(path, image=u8)
    return path


def load_cached_float(cache_dir: Path, study_uid: str) -> np.ndarray | None:
    raw = load_cached(cache_dir, study_uid)
    if raw is None:
        return None
    if raw.dtype == np.uint8:
        return (raw.astype(np.float32) / 255.0)
    return raw.astype(np.float32)
