"""Study-level multi-plane mid-slice imaging dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from rsna_knee.data.cache import load_cached_float, save_cached
from rsna_knee.data.dicom import (
    build_multiplane_tensor,
    load_mid_slice,
    synthetic_mid_slice,
)
from rsna_knee.data.series_selection import select_series_for_study
from rsna_knee.labels.laterality import assert_vision_only_inference_inputs
from rsna_knee.labels.targets import (
    ANATOMICAL_PLANES,
    REPORT_COL,
    SERIES_ID_COL,
    STUDY_ID_COL,
    TARGET_COLUMNS,
)
from rsna_knee.training.augment import AugmentConfig, augment_study


class StudyMultiPlaneDataset:
    """PyTorch-compatible dataset returning study tensors + labels.

    When ``series_root`` is None, synthetic mid-slices are generated so the
    training loop can be smoke-tested on the Mac without competition DICOMs.
    """

    def __init__(
        self,
        studies: pd.DataFrame,
        series_meta: pd.DataFrame,
        *,
        series_root: Path | None,
        cache_dir: Path | None = None,
        image_size: int = 224,
        planes: Sequence[str] = ANATOMICAL_PLANES,
        prefer_fat_suppression: bool = True,
        mid_frac: float = 0.5,
        train: bool = False,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 42,
        inference: bool = False,
        weights: pd.DataFrame | None = None,
    ) -> None:
        if inference:
            assert_vision_only_inference_inputs(
                [c for c in studies.columns if c != REPORT_COL]
            )
        self.studies = studies.reset_index(drop=True)
        self.series_meta = series_meta
        self.series_root = Path(series_root) if series_root is not None else None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.image_size = image_size
        self.planes = tuple(planes)
        self.prefer_fat_suppression = prefer_fat_suppression
        self.mid_frac = mid_frac
        self.train = train
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.seed = seed
        self._epoch = 0
        self.weights = None
        if weights is not None:
            w = weights.copy()
            w[STUDY_ID_COL] = w[STUDY_ID_COL].astype(str)
            self.weights = w.set_index(STUDY_ID_COL)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.studies)

    def _labels_for_row(self, row: pd.Series) -> np.ndarray:
        vals = []
        for col in TARGET_COLUMNS:
            if col in row.index and pd.notna(row[col]):
                vals.append(float(row[col]))
            else:
                vals.append(float("nan"))
        return np.asarray(vals, dtype=np.float32)

    def _weights_for_uid(self, study_uid: str) -> np.ndarray:
        if self.weights is None or study_uid not in self.weights.index:
            return np.ones(len(TARGET_COLUMNS), dtype=np.float32)
        row = self.weights.loc[study_uid]
        vals = []
        for col in TARGET_COLUMNS:
            v = row[col] if col in row.index else 1.0
            vals.append(0.0 if pd.isna(v) else float(v))
        return np.asarray(vals, dtype=np.float32)

    def _load_image(self, study_uid: str) -> np.ndarray:
        if self.cache_dir is not None:
            cached = load_cached_float(self.cache_dir, study_uid)
            if cached is not None:
                return cached

        if self.series_root is None:
            # Synthetic path for local smoke tests.
            plane_slices = {
                plane: synthetic_mid_slice(
                    seed=abs(hash((study_uid, plane, self.seed))) % (2**32),
                    size=self.image_size,
                )
                for plane in self.planes
            }
            image = build_multiplane_tensor(
                plane_slices, planes=self.planes, image_size=self.image_size
            )
        else:
            picks = select_series_for_study(
                self.series_meta,
                study_uid,
                planes=self.planes,
                prefer_fat_suppression=self.prefer_fat_suppression,
            )
            plane_slices: dict[str, np.ndarray] = {}
            for pick in picks:
                series_dir = (
                    self.series_root / study_uid / pick.series_uid
                )
                try:
                    plane_slices[pick.plane] = load_mid_slice(
                        series_dir,
                        image_size=self.image_size,
                        mid_frac=self.mid_frac,
                    )
                except Exception:
                    continue
            if not plane_slices:
                # Last-resort zeros — training loop should log these.
                image = np.zeros(
                    (len(self.planes), self.image_size, self.image_size),
                    dtype=np.float32,
                )
            else:
                image = build_multiplane_tensor(
                    plane_slices, planes=self.planes, image_size=self.image_size
                )

        if self.cache_dir is not None:
            save_cached(self.cache_dir, study_uid, image)
        return image

    def __getitem__(self, idx: int) -> dict:
        row = self.studies.iloc[idx]
        study_uid = str(row[STUDY_ID_COL])
        image = self._load_image(study_uid)
        labels = self._labels_for_row(row)
        weights = self._weights_for_uid(study_uid)

        if self.train:
            rng = np.random.default_rng(self.seed + self._epoch * 100_003 + idx)
            image, labels, meta = augment_study(
                image, labels, cfg=self.augment_cfg, rng=rng, weights=weights
            )
            weights = meta.get("weights", weights)

        return {
            "study_uid": study_uid,
            "image": image,
            "labels": labels,
            "weights": weights,
        }


def collate_studies(batch: list[dict]) -> dict:
    """Collate to torch tensors when torch is available; else numpy."""
    uids = [b["study_uid"] for b in batch]
    images = np.stack([b["image"] for b in batch], axis=0)
    labels = np.stack([b["labels"] for b in batch], axis=0)
    weights = np.stack(
        [b.get("weights", np.ones(labels.shape[1], dtype=np.float32)) for b in batch],
        axis=0,
    )
    try:
        import torch

        return {
            "study_uid": uids,
            "image": torch.from_numpy(images),
            "labels": torch.from_numpy(labels),
            "weights": torch.from_numpy(weights),
        }
    except ImportError:  # pragma: no cover
        return {
            "study_uid": uids,
            "image": images,
            "labels": labels,
            "weights": weights,
        }
