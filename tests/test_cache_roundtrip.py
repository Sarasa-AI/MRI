"""Cache round-trip: uint8 on disk must reload as float32 in [0, 1]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rsna_knee.data.cache import (
    cache_path,
    load_cached,
    load_cached_float,
    save_cached,
)


TOL = 0.5 / 255.0


def _sample_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((3, 32, 32), dtype=np.float32)


def test_save_stores_uint8(tmp_path: Path):
    image = _sample_image()
    path = save_cached(tmp_path, "study-a", image)
    with np.load(path) as data:
        on_disk = data["image"]
    assert on_disk.dtype == np.uint8
    assert int(on_disk.min()) >= 0
    assert int(on_disk.max()) <= 255


def test_load_cached_preserves_uint8_dtype(tmp_path: Path):
    image = _sample_image(1)
    save_cached(tmp_path, "study-b", image)
    raw = load_cached(tmp_path, "study-b")
    assert raw is not None
    assert raw.dtype == np.uint8


def test_roundtrip_float01(tmp_path: Path):
    image = _sample_image(2)
    save_cached(tmp_path, "study-c", image)
    reloaded = load_cached_float(tmp_path, "study-c")
    assert reloaded is not None
    assert reloaded.dtype == np.float32
    assert bool(np.isfinite(reloaded).all())
    assert float(reloaded.min()) >= 0.0 - 1e-6
    assert float(reloaded.max()) <= 1.0 + 1e-6
    assert reloaded.shape == image.shape
    np.testing.assert_allclose(reloaded, image, atol=TOL, rtol=0.0)


def test_case_a_cache_miss(tmp_path: Path):
    assert load_cached_float(tmp_path, "missing-uid") is None
    assert not cache_path(tmp_path, "missing-uid").exists()


def test_case_b_cache_hit_normalized(tmp_path: Path):
    image = _sample_image(3)
    save_cached(tmp_path, "study-d", image)
    hit = load_cached_float(tmp_path, "study-d")
    assert hit is not None
    assert hit.dtype == np.float32
    assert float(hit.max()) <= 1.0 + 1e-6
    assert float(hit.min()) >= 0.0 - 1e-6
    assert bool(np.isfinite(hit).all())


def test_case_c_repeated_hit_stable(tmp_path: Path):
    image = _sample_image(4)
    save_cached(tmp_path, "study-e", image)
    hit1 = load_cached_float(tmp_path, "study-e")
    hit2 = load_cached_float(tmp_path, "study-e")
    assert hit1 is not None and hit2 is not None
    np.testing.assert_allclose(hit1, hit2, atol=0.0, rtol=0.0)


def test_cache_miss_approx_cache_hit(tmp_path: Path):
    """Producer [0,1] after save/load must match within uint8 quantization."""
    image = _sample_image(5)
    # Simulate miss path return value (producer float in [0,1]).
    miss = image.copy()
    save_cached(tmp_path, "study-f", miss)
    hit = load_cached_float(tmp_path, "study-f")
    assert hit is not None
    np.testing.assert_allclose(miss, hit, atol=TOL, rtol=0.0)
    assert float(hit.max()) <= 1.0 + 1e-6


def test_regression_no_0255_reload(tmp_path: Path):
    """The DEBUG-001 failure mode: reload max must not be ~255."""
    image = np.full((3, 8, 8), 0.5, dtype=np.float32)
    save_cached(tmp_path, "study-g", image)
    hit = load_cached_float(tmp_path, "study-g")
    assert hit is not None
    assert float(hit.max()) < 1.5
    assert float(hit.max()) != pytest.approx(255.0)
