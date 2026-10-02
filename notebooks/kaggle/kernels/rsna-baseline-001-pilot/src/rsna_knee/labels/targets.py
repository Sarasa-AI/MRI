"""Official competition target labels and schema constants."""

from __future__ import annotations

from typing import Final

# Official submission / evaluation targets (order matters for CSV columns).
TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

N_TARGETS: Final[int] = len(TARGET_COLUMNS)

STUDY_ID_COL: Final[str] = "StudyInstanceUID"
SERIES_ID_COL: Final[str] = "SeriesInstanceUID"
REPORT_COL: Final[str] = "Report"

# Series metadata columns (train_series.csv / test_series.csv).
SERIES_META_COLUMNS: Final[tuple[str, ...]] = (
    STUDY_ID_COL,
    SERIES_ID_COL,
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)

ANATOMICAL_PLANES: Final[tuple[str, ...]] = ("Sagittal", "Coronal", "Axial")

# Medial/lateral compartment pairs. Horizontal flips must swap these labels
# (or be disabled). These are anatomical compartments within a knee study,
# not left/right knee PatientLaterality.
COMPARTMENT_FLIP_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("Medial Meniscus", "Lateral Meniscus"),
    ("Medial OA", "Lateral OA"),
)

# Reports exist in train only. Must never be required at inference.
REPORT_ALLOWED_AT_INFERENCE: Final[bool] = False

SUBMISSION_ID_COL: Final[str] = STUDY_ID_COL
SUBMISSION_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (SUBMISSION_ID_COL, *TARGET_COLUMNS)

COMPETITION_SLUG: Final[str] = "rsna-knee-abnormality-detection"
