"""DICOM slice I/O — intended for Kaggle runtimes only.

The Mac control plane must not materialize competition DICOMs. Local unit
tests use synthetic arrays via ``synthetic_mid_slice``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from rsna_knee.labels.targets import ANATOMICAL_PLANES


def _try_import_pydicom():
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional local dep
        raise ImportError(
            "pydicom is required for DICOM decode on Kaggle. "
            "Install pydicom (+ pylibjpeg/gdcm for compressed transfer syntaxes)."
        ) from exc
    return pydicom


def list_dicom_paths(series_dir: Path) -> list[Path]:
    if not series_dir.exists():
        return []
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        paths = sorted(p for p in series_dir.iterdir() if p.is_file())
    return paths


def _read_dicom_array(path: Path) -> tuple[np.ndarray, int]:
    """Return (float32 H×W, InstanceNumber)."""
    pydicom = _try_import_pydicom()
    ds = pydicom.dcmread(str(path), force=True)
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        # Compressed transfer syntax fallback chain.
        for plugin in ("pylibjpeg", "gdcm"):
            try:
                import importlib

                importlib.import_module(plugin)
                arr = ds.pixel_array.astype(np.float32)
                break
            except Exception:
                continue
        else:
            raise

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric.upper() == "MONOCHROME1":
        arr = arr.max() - arr

    instance = int(getattr(ds, "InstanceNumber", 0) or 0)
    if arr.ndim == 3:
        # Rare multi-frame / RGB — take first channel / frame.
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    return arr, instance


def percentile_window(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip to percentiles and scale to [0, 1]."""
    lo, hi = np.percentile(arr, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip(arr, lo, hi)
    out = (out - lo) / (hi - lo)
    return out.astype(np.float32)


def resize_square(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize HxW float array to size×size."""
    try:
        import cv2

        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    except Exception:
        from PIL import Image

        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        return (np.asarray(img).astype(np.float32) / 255.0)


def load_mid_slice(
    series_dir: Path,
    *,
    image_size: int = 224,
    mid_frac: float = 0.5,
) -> np.ndarray:
    """Load the mid-InstanceNumber slice from a series directory → (H, W) float32 [0,1]."""
    paths = list_dicom_paths(series_dir)
    if not paths:
        raise FileNotFoundError(f"No DICOM files under {series_dir}")

    slices: list[tuple[int, np.ndarray]] = []
    errors: list[str] = []
    for path in paths:
        try:
            arr, instance = _read_dicom_array(path)
            slices.append((instance, arr))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")
    if not slices:
        raise RuntimeError(
            f"Failed to decode any DICOM in {series_dir}. "
            f"Examples: {errors[:3]}"
        )

    slices.sort(key=lambda x: x[0])
    idx = int(np.clip(round((len(slices) - 1) * mid_frac), 0, len(slices) - 1))
    mid = slices[idx][1]
    mid = percentile_window(mid)
    mid = resize_square(mid, image_size)
    return mid


def synthetic_mid_slice(seed: int, size: int = 224) -> np.ndarray:
    """Deterministic synthetic mid-slice for local tests (no DICOM)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.15, size=(size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    circle = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.3) ** 2
    base[circle] += 0.2
    return np.clip(base, 0.0, 1.0)


def build_multiplane_tensor(
    plane_slices: dict[str, np.ndarray],
    *,
    planes: Sequence[str] = ANATOMICAL_PLANES,
    image_size: int = 224,
) -> np.ndarray:
    """Stack plane mid-slices into (C, H, W) with C=len(planes).

    Missing planes are filled by replicating the mean of available planes
    (or zeros if none exist — caller should avoid empty studies).
    """
    available = [plane_slices[p] for p in planes if p in plane_slices]
    if not available:
        fill = np.zeros((image_size, image_size), dtype=np.float32)
    else:
        fill = np.mean(np.stack(available, axis=0), axis=0).astype(np.float32)

    channels = []
    for plane in planes:
        if plane in plane_slices:
            channels.append(plane_slices[plane].astype(np.float32))
        else:
            channels.append(fill.copy())
    return np.stack(channels, axis=0)
