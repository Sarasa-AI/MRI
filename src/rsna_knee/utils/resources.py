"""Runtime / memory measurement helpers."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator


@dataclass
class ResourceSnapshot:
    wall_seconds: float
    peak_rss_mb: float | None
    cuda_peak_allocated_mb: float | None = None
    device: str = "cpu"


def _rss_mb() -> float | None:
    try:
        import resource

        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0
    except Exception:
        return None


def _cuda_peak_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    except Exception:
        return None
    return None


@contextmanager
def measure_resources(device: str = "cpu") -> Iterator[dict]:
    """Context manager that records wall time + peak RSS / CUDA memory."""
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    box: dict = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        elapsed = time.perf_counter() - t0
        snap = ResourceSnapshot(
            wall_seconds=float(elapsed),
            peak_rss_mb=_rss_mb(),
            cuda_peak_allocated_mb=_cuda_peak_mb(),
            device=device,
        )
        box.update(asdict(snap))


def pick_device(prefer: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if prefer == "mps":
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
