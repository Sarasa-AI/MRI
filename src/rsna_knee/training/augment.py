"""Compartment-safe imaging augmentations for study-level tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rsna_knee.labels.laterality import swap_compartment_label_matrix


@dataclass
class AugmentConfig:
    hflip_p: float = 0.5
    rotate_deg: float = 10.0
    brightness: float = 0.15
    contrast: float = 0.15
    scale_min: float = 0.90
    scale_max: float = 1.00
    # D-007: horizontal flips allowed only with medial↔lateral swaps.
    horizontal_flip_policy: str = "allow_with_compartment_swap"


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image

        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(img).astype(np.float32) / 255.0


def _rotate(chw: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-6:
        return chw
    try:
        import cv2

        c, h, w = chw.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        out = np.stack(
            [cv2.warpAffine(chw[i], matrix, (w, h), flags=cv2.INTER_LINEAR) for i in range(c)],
            axis=0,
        )
        return out.astype(np.float32)
    except Exception:
        # Soft fallback: skip rotation if cv2 unavailable.
        return chw


def _scale_crop(chw: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return chw
    c, h, w = chw.shape
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = np.stack([_resize(chw[i], max(nh, nw)) for i in range(c)], axis=0)
    # Center crop or pad back to h×w
    _, rh, rw = resized.shape
    if rh >= h and rw >= w:
        y0 = (rh - h) // 2
        x0 = (rw - w) // 2
        return resized[:, y0 : y0 + h, x0 : x0 + w].astype(np.float32)
    out = np.zeros_like(chw)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    out[:, y0 : y0 + rh, x0 : x0 + rw] = resized
    return out


def augment_study(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    cfg: AugmentConfig,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply controlled aug to (C,H,W) image and (12,) labels.

    Returns (image, labels, meta) where meta records whether H-flip occurred.
    When ``weights`` is provided it is compartment-swapped with labels on H-flip
    and returned in ``meta['weights']``.
    """
    assert image.ndim == 3
    y = labels.astype(np.float32).copy()
    x = image.astype(np.float32).copy()
    w = None if weights is None else weights.astype(np.float32).copy()
    meta: dict = {"hflip": False}

    # Scale
    scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
    x = _scale_crop(x, scale)

    # Rotate
    deg = float(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    x = _rotate(x, deg)

    # Brightness / contrast
    if cfg.brightness > 0:
        x = x + float(rng.uniform(-cfg.brightness, cfg.brightness))
    if cfg.contrast > 0:
        factor = 1.0 + float(rng.uniform(-cfg.contrast, cfg.contrast))
        mean = float(x.mean())
        x = (x - mean) * factor + mean
    x = np.clip(x, 0.0, 1.0)

    # Horizontal flip with compartment label swap (D-007)
    if (
        cfg.horizontal_flip_policy == "allow_with_compartment_swap"
        and rng.random() < cfg.hflip_p
    ):
        x = np.ascontiguousarray(x[:, :, ::-1])
        y = swap_compartment_label_matrix(y.reshape(1, -1))[0]
        if w is not None:
            w = swap_compartment_label_matrix(w.reshape(1, -1))[0]
        meta["hflip"] = True

    if w is not None:
        meta["weights"] = w.astype(np.float32)
    return x.astype(np.float32), y.astype(np.float32), meta
