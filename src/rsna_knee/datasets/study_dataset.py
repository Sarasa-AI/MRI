"""Dataset interfaces — metadata stub only (no DICOM decode yet)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rsna_knee.labels.laterality import assert_vision_only_inference_inputs
from rsna_knee.labels.targets import REPORT_COL, STUDY_ID_COL, TARGET_COLUMNS


@dataclass
class StudyRecord:
    study_uid: str
    labels: dict[str, float | None]
    has_report: bool


class StudyMetadataDataset:
    """Study-level metadata dataset for local / unit-test workflows.

    Imaging tensors are intentionally absent. A future Kaggle imaging
    dataset should implement the same study-level interface without
    requiring Report at inference.
    """

    def __init__(self, frame: pd.DataFrame, *, inference: bool = False) -> None:
        if inference:
            assert_vision_only_inference_inputs(
                [c for c in frame.columns if c != REPORT_COL]
            )
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> StudyRecord:
        row = self.frame.iloc[idx]
        labels = {
            name: (None if pd.isna(row[name]) else float(row[name]))
            for name in TARGET_COLUMNS
            if name in self.frame.columns
        }
        return StudyRecord(
            study_uid=str(row[STUDY_ID_COL]),
            labels=labels,
            has_report=bool(REPORT_COL in self.frame.columns and pd.notna(row.get(REPORT_COL))),
        )
