"""Training exports."""

from rsna_knee.training.loop import SmokeTrainResult, run_metadata_smoke
from rsna_knee.training.seed import seed_everything

try:
    from rsna_knee.training.imaging_loop import (
        BaselineConfig,
        BaselineResult,
        load_baseline_config,
        run_imaging_baseline,
    )
except ImportError:  # torch optional at import time for metadata-only envs
    BaselineConfig = None  # type: ignore
    BaselineResult = None  # type: ignore
    load_baseline_config = None  # type: ignore
    run_imaging_baseline = None  # type: ignore

__all__ = [
    "SmokeTrainResult",
    "run_metadata_smoke",
    "seed_everything",
    "BaselineConfig",
    "BaselineResult",
    "load_baseline_config",
    "run_imaging_baseline",
]
