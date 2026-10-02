"""Series selection policy for multi-plane mid-slice 2.5D inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from rsna_knee.labels.targets import (
    ANATOMICAL_PLANES,
    SERIES_ID_COL,
    STUDY_ID_COL,
)


@dataclass(frozen=True)
class SelectedSeries:
    study_uid: str
    plane: str
    series_uid: str
    fluid_sensitive: int
    fat_suppression: int


def _score_series(row: pd.Series, *, prefer_fat_suppression: bool) -> tuple:
    """Higher is better. Prefer fluid-sensitive, then fat-suppression."""
    fluid = int(row.get("Fluid_Sensitive", 0) or 0)
    fat = int(row.get("Fat_Suppression", 0) or 0)
    if prefer_fat_suppression:
        return (fluid, fat)
    return (fluid, 0)


def select_series_for_study(
    series_df: pd.DataFrame,
    study_uid: str,
    *,
    planes: Sequence[str] = ANATOMICAL_PLANES,
    prefer_fat_suppression: bool = True,
) -> list[SelectedSeries]:
    """Pick one series per anatomical plane for a study.

    Preference within a plane:
      1. Fluid_Sensitive == 1
      2. Fat_Suppression == 1 (if enabled)
      3. Stable SeriesInstanceUID tie-break
    """
    study = series_df[series_df[STUDY_ID_COL].astype(str) == str(study_uid)]
    selected: list[SelectedSeries] = []
    for plane in planes:
        candidates = study[study["Anatomical_Plane"].astype(str) == plane]
        if candidates.empty:
            continue
        ranked = candidates.copy()
        ranked["_score"] = ranked.apply(
            lambda r: _score_series(r, prefer_fat_suppression=prefer_fat_suppression),
            axis=1,
        )
        ranked = ranked.sort_values(
            by=["_score", SERIES_ID_COL], ascending=[False, True]
        )
        best = ranked.iloc[0]
        selected.append(
            SelectedSeries(
                study_uid=str(study_uid),
                plane=plane,
                series_uid=str(best[SERIES_ID_COL]),
                fluid_sensitive=int(best.get("Fluid_Sensitive", 0) or 0),
                fat_suppression=int(best.get("Fat_Suppression", 0) or 0),
            )
        )
    return selected


def select_series_table(
    series_df: pd.DataFrame,
    study_uids: Iterable[str],
    *,
    planes: Sequence[str] = ANATOMICAL_PLANES,
    prefer_fat_suppression: bool = True,
) -> pd.DataFrame:
    """Build a long table of selected series for many studies."""
    rows: list[dict] = []
    for sid in study_uids:
        picks = select_series_for_study(
            series_df,
            sid,
            planes=planes,
            prefer_fat_suppression=prefer_fat_suppression,
        )
        for p in picks:
            rows.append(
                {
                    STUDY_ID_COL: p.study_uid,
                    "plane": p.plane,
                    SERIES_ID_COL: p.series_uid,
                    "Fluid_Sensitive": p.fluid_sensitive,
                    "Fat_Suppression": p.fat_suppression,
                }
            )
    return pd.DataFrame(rows)
