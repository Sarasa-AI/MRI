"""Read-only DICOM audit helpers for RSNA-BASELINE-001-PILOT.

Does NOT alter baseline loading. Reuses list_dicom_paths / series selection /
_read_dicom_array to measure real-data assumptions (ordering, orientation,
decode success, degenerate tensors, timing).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rsna_knee.data.dicom import (
    _read_dicom_array,
    list_dicom_paths,
    load_mid_slice,
    percentile_window,
    resize_square,
)
from rsna_knee.data.series_selection import select_series_for_study
from rsna_knee.labels.targets import ANATOMICAL_PLANES, SERIES_ID_COL, STUDY_ID_COL

HEADER_TAGS = (
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "InstanceNumber",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "SamplesPerPixel",
    "Modality",
)


@dataclass
class TensorQuality:
    status: str  # valid | zero | near_zero | nan | inf | failed | missing
    shape: list[int] | None = None
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    reason: str | None = None


@dataclass
class MidSliceAuditRow:
    study_uid: str
    plane: str
    series_uid: str | None
    fluid_sensitive: int | None = None
    fat_suppression: int | None = None
    success: bool = False
    quality: TensorQuality = field(default_factory=lambda: TensorQuality(status="missing"))
    discovery_sec: float | None = None
    decode_preprocess_sec: float | None = None


def _try_import_pydicom():
    import pydicom

    return pydicom


def _tag_value(ds: Any, name: str) -> Any:
    if not hasattr(ds, name):
        return None
    val = getattr(ds, name)
    if val is None:
        return None
    try:
        # Convert MultiValue / Dataset types to JSON-friendly forms.
        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
            return [float(x) if _is_number(x) else str(x) for x in val]
        if _is_number(val):
            return float(val) if isinstance(val, float) else int(val) if float(val).is_integer() else float(val)
        return str(val)
    except Exception:
        return str(val)


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return not isinstance(x, bool)
    except Exception:
        return False


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def classify_array(arr: np.ndarray | None, *, reason: str | None = None) -> TensorQuality:
    if arr is None:
        return TensorQuality(status="missing", reason=reason)
    if not np.isfinite(arr).all():
        if np.isnan(arr).any():
            return TensorQuality(status="nan", shape=list(arr.shape), reason=reason)
        return TensorQuality(status="inf", shape=list(arr.shape), reason=reason)
    mean = float(arr.mean())
    std = float(arr.std())
    amin = float(arr.min())
    amax = float(arr.max())
    shape = list(arr.shape)
    if amax == 0.0 and amin == 0.0:
        return TensorQuality("zero", shape, mean, std, amin, amax, reason)
    if std < 1e-6 and abs(mean) < 1e-6:
        return TensorQuality("near_zero", shape, mean, std, amin, amax, reason)
    if std < 1e-6:
        return TensorQuality("near_zero", shape, mean, std, amin, amax, reason or "near-constant")
    return TensorQuality("valid", shape, mean, std, amin, amax, reason)


def series_dir_for(study_uid: str, series_uid: str, series_root: Path) -> Path:
    return Path(series_root) / str(study_uid) / str(series_uid)


def audit_gold_mid_slices(
    gold_studies: pd.DataFrame,
    series_meta: pd.DataFrame,
    series_root: Path,
    *,
    planes: Sequence[str] = ANATOMICAL_PLANES,
    prefer_fat_suppression: bool = True,
    image_size: int = 224,
    mid_frac: float = 0.5,
) -> dict[str, Any]:
    """Decode baseline mid-slices for every gold study × plane."""
    rows: list[MidSliceAuditRow] = []
    failure_examples: dict[str, list[dict[str, str]]] = {}
    failure_counts: Counter[str] = Counter()

    for _, study_row in gold_studies.iterrows():
        study_uid = str(study_row[STUDY_ID_COL])
        picks = select_series_for_study(
            series_meta,
            study_uid,
            planes=planes,
            prefer_fat_suppression=prefer_fat_suppression,
        )
        by_plane = {p.plane: p for p in picks}
        for plane in planes:
            if plane not in by_plane:
                row = MidSliceAuditRow(
                    study_uid=study_uid,
                    plane=plane,
                    series_uid=None,
                    success=False,
                    quality=TensorQuality(status="missing", reason="no_series_for_plane"),
                )
                rows.append(row)
                failure_counts["no_series_for_plane"] += 1
                failure_examples.setdefault("no_series_for_plane", []).append(
                    {"study": study_uid, "plane": plane}
                )
                continue

            pick = by_plane[plane]
            sdir = series_dir_for(study_uid, pick.series_uid, series_root)
            t0 = time.perf_counter()
            paths = list_dicom_paths(sdir)
            discovery_sec = time.perf_counter() - t0
            if not paths:
                reason = "missing_dicom_files" if not sdir.exists() else "empty_series_dir"
                if not sdir.exists():
                    reason = "missing_series_dir"
                row = MidSliceAuditRow(
                    study_uid=study_uid,
                    plane=plane,
                    series_uid=pick.series_uid,
                    fluid_sensitive=pick.fluid_sensitive,
                    fat_suppression=pick.fat_suppression,
                    success=False,
                    quality=TensorQuality(status="missing", reason=reason),
                    discovery_sec=discovery_sec,
                )
                rows.append(row)
                failure_counts[reason] += 1
                failure_examples.setdefault(reason, []).append(
                    {
                        "study": study_uid,
                        "plane": plane,
                        "series": pick.series_uid,
                    }
                )
                continue

            t1 = time.perf_counter()
            try:
                arr = load_mid_slice(sdir, image_size=image_size, mid_frac=mid_frac)
                decode_sec = time.perf_counter() - t1
                quality = classify_array(arr)
                success = quality.status == "valid"
                if not success:
                    failure_counts[quality.status] += 1
                    failure_examples.setdefault(quality.status, []).append(
                        {
                            "study": study_uid,
                            "plane": plane,
                            "series": pick.series_uid,
                        }
                    )
                rows.append(
                    MidSliceAuditRow(
                        study_uid=study_uid,
                        plane=plane,
                        series_uid=pick.series_uid,
                        fluid_sensitive=pick.fluid_sensitive,
                        fat_suppression=pick.fat_suppression,
                        success=success,
                        quality=quality,
                        discovery_sec=discovery_sec,
                        decode_preprocess_sec=decode_sec,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                decode_sec = time.perf_counter() - t1
                reason = _classify_exception(exc)
                failure_counts[reason] += 1
                failure_examples.setdefault(reason, []).append(
                    {
                        "study": study_uid,
                        "plane": plane,
                        "series": pick.series_uid,
                        "error": str(exc)[:240],
                    }
                )
                rows.append(
                    MidSliceAuditRow(
                        study_uid=study_uid,
                        plane=plane,
                        series_uid=pick.series_uid,
                        fluid_sensitive=pick.fluid_sensitive,
                        fat_suppression=pick.fat_suppression,
                        success=False,
                        quality=TensorQuality(status="failed", reason=f"{reason}: {exc}"[:300]),
                        discovery_sec=discovery_sec,
                        decode_preprocess_sec=decode_sec,
                    )
                )

    n = len(rows)
    n_success = sum(1 for r in rows if r.success)
    n_zero = sum(1 for r in rows if r.quality.status in {"zero", "near_zero"})
    decode_times = [r.decode_preprocess_sec for r in rows if r.decode_preprocess_sec is not None]
    discovery_times = [r.discovery_sec for r in rows if r.discovery_sec is not None]

    return {
        "n_study_plane_attempts": n,
        "n_success": n_success,
        "success_rate": float(n_success / n) if n else 0.0,
        "n_degenerate_zero_or_near_zero": n_zero,
        "degenerate_rate": float(n_zero / n) if n else 0.0,
        "failure_counts": dict(failure_counts),
        "failure_examples": {k: v[:5] for k, v in failure_examples.items()},
        "timing": {
            "discovery_sec": _summarize_numeric(discovery_times),
            "decode_preprocess_sec": _summarize_numeric(decode_times),
        },
        "rows": [
            {
                "study_uid": r.study_uid,
                "plane": r.plane,
                "series_uid": r.series_uid,
                "fluid_sensitive": r.fluid_sensitive,
                "fat_suppression": r.fat_suppression,
                "success": r.success,
                "quality": asdict(r.quality),
                "discovery_sec": r.discovery_sec,
                "decode_preprocess_sec": r.decode_preprocess_sec,
            }
            for r in rows
        ],
    }


def _classify_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    name = type(exc).__name__
    if isinstance(exc, FileNotFoundError) or "no dicom" in msg:
        return "missing_dicom"
    if "pixel" in msg or "transfer syntax" in msg or "compress" in msg:
        return "unsupported_pixel_data"
    if "pydicom" in msg or name.startswith("InvalidDicom"):
        return "pydicom_failure"
    if "shape" in msg or "dimension" in msg:
        return "unexpected_dimensions"
    return f"other:{name}"


def _summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def read_series_headers(
    series_dir: Path,
    *,
    max_instances: int | None = None,
) -> dict[str, Any]:
    """Read allowlisted DICOM tags for instances in a series (no pixel decode preferred)."""
    pydicom = _try_import_pydicom()
    paths = list_dicom_paths(series_dir)
    if max_instances is not None:
        paths = paths[: max_instances]
    instances: list[dict[str, Any]] = []
    missing_tag_counts: Counter[str] = Counter()
    errors: list[str] = []

    for path in paths:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )
            record = {"file": path.name}
            for tag in HEADER_TAGS:
                val = _tag_value(ds, tag)
                record[tag] = val
                if val is None:
                    missing_tag_counts[tag] += 1
            instances.append(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")

    rows = [inst.get("Rows") for inst in instances if inst.get("Rows") is not None]
    cols = [inst.get("Columns") for inst in instances if inst.get("Columns") is not None]
    return {
        "series_dir": str(series_dir),
        "n_files": len(list_dicom_paths(series_dir)),
        "n_headers_read": len(instances),
        "n_header_errors": len(errors),
        "header_errors": errors[:5],
        "missing_tag_counts": dict(missing_tag_counts),
        "rows_stats": _summarize_numeric([float(x) for x in rows]),
        "cols_stats": _summarize_numeric([float(x) for x in cols]),
        "instances": instances,
    }


def physical_slice_positions(instances: list[dict[str, Any]]) -> tuple[np.ndarray | None, str | None]:
    """Project ImagePositionPatient onto slice normal from ImageOrientationPatient."""
    usable = []
    for inst in instances:
        ipp = inst.get("ImagePositionPatient")
        iop = inst.get("ImageOrientationPatient")
        inst_n = inst.get("InstanceNumber")
        if ipp is None or iop is None or inst_n is None:
            continue
        if len(ipp) != 3 or len(iop) != 6:
            continue
        usable.append((float(inst_n), np.asarray(ipp, dtype=float), np.asarray(iop, dtype=float)))
    if len(usable) < 2:
        return None, "insufficient_ipp_iop_instance_triplets"

    # Use first IOP as reference normal (row × col).
    iop0 = usable[0][2]
    row = iop0[:3]
    col = iop0[3:]
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None, "degenerate_iop_normal"
    normal = normal / norm

    instance_nums = np.asarray([u[0] for u in usable], dtype=float)
    proj = np.asarray([float(np.dot(u[1], normal)) for u in usable], dtype=float)
    # Sort by InstanceNumber (baseline order) and return projections in that order.
    order = np.argsort(instance_nums)
    return np.stack([instance_nums[order], proj[order]], axis=1), None


def compare_instance_vs_ipp_order(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare InstanceNumber order vs physical IPP projection order."""
    paired, err = physical_slice_positions(instances)
    if paired is None:
        return {
            "comparable": False,
            "reason": err,
            "spearman": None,
            "direction_agree": None,
            "pairwise_inversion_rate": None,
            "n": 0,
        }

    inst = paired[:, 0]
    proj = paired[:, 1]
    # Rank correlation between InstanceNumber and physical position.
    if np.unique(inst).size < 2 or np.unique(proj).size < 2:
        return {
            "comparable": False,
            "reason": "constant_instance_or_projection",
            "spearman": None,
            "direction_agree": None,
            "pairwise_inversion_rate": None,
            "n": int(len(inst)),
        }

    rho = float(_spearman(inst, proj))
    # Direction: does increasing InstanceNumber monotonically increase or decrease proj?
    diffs_inst = np.diff(inst)
    diffs_proj = np.diff(proj)
    # Only adjacent pairs where InstanceNumber increases (should always after sort).
    sign = np.sign(diffs_proj[diffs_inst > 0])
    if sign.size == 0:
        direction_agree = None
        inversion_rate = None
    else:
        # Agree if mostly one direction (allow reverse-sorted series as consistent).
        pos = float((sign > 0).mean())
        neg = float((sign < 0).mean())
        majority = max(pos, neg)
        direction_agree = majority >= 0.95
        inversion_rate = float(1.0 - majority)

    return {
        "comparable": True,
        "reason": None,
        "spearman": float(rho) if rho == rho else None,
        "abs_spearman": float(abs(rho)) if rho == rho else None,
        "direction_agree": direction_agree,
        "pairwise_inversion_rate": inversion_rate,
        "n": int(len(inst)),
        "proj_span": float(proj.max() - proj.min()),
    }


def audit_orientation_sample(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize IOP patterns for a series sample (evidence only, no correction)."""
    iops = []
    for inst in instances:
        iop = inst.get("ImageOrientationPatient")
        if iop is not None and len(iop) == 6:
            iops.append([float(x) for x in iop])
    if not iops:
        return {"n_with_iop": 0, "unique_iop_count": 0, "notes": ["ImageOrientationPatient missing"]}

    arr = np.asarray(iops, dtype=float)
    # Round for uniqueness of near-identical orientations.
    rounded = np.round(arr, decimals=3)
    uniq = np.unique(rounded, axis=0)
    # Plane normal from first IOP.
    row = arr[0, :3]
    col = arr[0, 3:]
    normal = np.cross(row, col)
    nrm = np.linalg.norm(normal)
    if nrm > 1e-8:
        normal = normal / nrm
    axis_alignment = {
        "abs_normal": [float(abs(x)) for x in normal],
        "dominant_axis": int(np.argmax(np.abs(normal))) if nrm > 1e-8 else None,
    }
    notes = []
    if len(uniq) > 1:
        notes.append("multiple_distinct_IOP_within_series")
    # If row/col not approximately orthonormal, flag.
    if abs(np.dot(row, col)) > 0.05:
        notes.append("row_col_not_orthogonal")
    return {
        "n_with_iop": int(len(iops)),
        "unique_iop_count": int(len(uniq)),
        "example_iop": iops[0],
        "axis_alignment": axis_alignment,
        "notes": notes,
    }


def audit_header_rich_sample(
    selected_series: pd.DataFrame,
    series_root: Path,
    *,
    max_series: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Header + ordering + orientation audit on a capped series sample."""
    if selected_series.empty:
        return {"n_series": 0, "series": []}

    rng = np.random.default_rng(seed)
    df = selected_series.copy().reset_index(drop=True)
    if len(df) > max_series:
        idx = rng.choice(len(df), size=max_series, replace=False)
        df = df.iloc[sorted(idx)].reset_index(drop=True)

    series_reports = []
    order_results = []
    orientation_notes: Counter[str] = Counter()
    rows_all: list[float] = []
    cols_all: list[float] = []
    slice_counts: list[float] = []

    for _, row in df.iterrows():
        study_uid = str(row[STUDY_ID_COL])
        series_uid = str(row[SERIES_ID_COL])
        plane = str(row.get("plane", row.get("Anatomical_Plane", "")))
        sdir = series_dir_for(study_uid, series_uid, series_root)
        header = read_series_headers(sdir)
        order = compare_instance_vs_ipp_order(header["instances"])
        orient = audit_orientation_sample(header["instances"])
        for note in orient.get("notes", []):
            orientation_notes[note] += 1
        if header["rows_stats"]["n"]:
            # Collect per-series representative row/col from first instance.
            inst0 = header["instances"][0] if header["instances"] else {}
            if inst0.get("Rows") is not None:
                rows_all.append(float(inst0["Rows"]))
            if inst0.get("Columns") is not None:
                cols_all.append(float(inst0["Columns"]))
        slice_counts.append(float(header["n_files"]))
        order_results.append(order)
        series_reports.append(
            {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "plane": plane,
                "n_files": header["n_files"],
                "missing_tag_counts": header["missing_tag_counts"],
                "ordering": order,
                "orientation": orient,
                "rows": header["instances"][0].get("Rows") if header["instances"] else None,
                "columns": header["instances"][0].get("Columns") if header["instances"] else None,
                "modality": header["instances"][0].get("Modality") if header["instances"] else None,
                "photometric": (
                    header["instances"][0].get("PhotometricInterpretation")
                    if header["instances"]
                    else None
                ),
            }
        )

    comparable = [o for o in order_results if o.get("comparable")]
    disagree = [
        o
        for o in comparable
        if (o.get("abs_spearman") is not None and o["abs_spearman"] < 0.95)
        or (o.get("pairwise_inversion_rate") is not None and o["pairwise_inversion_rate"] > 0.05)
    ]
    aspect = []
    for r, c in zip(rows_all, cols_all):
        if c:
            aspect.append(r / c)

    return {
        "n_series_sampled": len(series_reports),
        "sample_note": "sample statistics only — not whole-dataset",
        "shape_stats": {
            "rows": _percentile_block(rows_all),
            "columns": _percentile_block(cols_all),
            "aspect_ratio": _percentile_block(aspect),
            "slice_count": _percentile_block(slice_counts),
        },
        "ordering_summary": {
            "n_comparable": len(comparable),
            "n_disagreement": len(disagree),
            "disagreement_rate": float(len(disagree) / len(comparable)) if comparable else None,
            "mean_abs_spearman": float(
                np.mean([o["abs_spearman"] for o in comparable if o.get("abs_spearman") is not None])
            )
            if comparable
            else None,
            "examples_disagreement": [
                s
                for s, o in zip(series_reports, order_results)
                if o.get("comparable")
                and (
                    (o.get("abs_spearman") is not None and o["abs_spearman"] < 0.95)
                    or (
                        o.get("pairwise_inversion_rate") is not None
                        and o["pairwise_inversion_rate"] > 0.05
                    )
                )
            ][:5],
        },
        "orientation_note_counts": dict(orientation_notes),
        "series": series_reports,
    }


def _percentile_block(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def build_selected_series_table(
    studies: pd.DataFrame,
    series_meta: pd.DataFrame,
    *,
    prefer_fat_suppression: bool = True,
) -> pd.DataFrame:
    from rsna_knee.data.series_selection import select_series_table

    return select_series_table(
        series_meta,
        studies[STUDY_ID_COL].astype(str).tolist(),
        prefer_fat_suppression=prefer_fat_suppression,
    )


def decide_dicom_gate(
    mid_audit: dict[str, Any],
    header_audit: dict[str, Any],
    *,
    max_failure_rate: float = 0.10,
    max_degenerate_rate: float = 0.05,
    max_order_disagreement_rate: float = 0.10,
) -> dict[str, Any]:
    """Return blocker flags from DICOM audits (does not mutate baseline)."""
    blockers: list[str] = []
    caveats: list[str] = []

    fail_rate = 1.0 - float(mid_audit.get("success_rate", 0.0))
    deg_rate = float(mid_audit.get("degenerate_rate", 0.0))
    if fail_rate > max_failure_rate:
        blockers.append(
            f"DICOM mid-slice failure rate {fail_rate:.3f} > {max_failure_rate} "
            f"(failures={mid_audit.get('failure_counts')})"
        )
    elif fail_rate > 0:
        caveats.append(f"Non-zero mid-slice failure rate {fail_rate:.3f}")

    if deg_rate > max_degenerate_rate:
        blockers.append(
            f"Degenerate (zero/near-zero) tensor rate {deg_rate:.3f} > {max_degenerate_rate}"
        )
    elif deg_rate > 0:
        caveats.append(f"Degenerate tensor rate {deg_rate:.3f}")

    order = header_audit.get("ordering_summary") or {}
    n_comp = int(order.get("n_comparable") or 0)
    dis_rate = order.get("disagreement_rate")
    if n_comp == 0:
        caveats.append("No series comparable for InstanceNumber vs IPP ordering")
    elif dis_rate is not None and dis_rate > max_order_disagreement_rate:
        blockers.append(
            f"InstanceNumber vs IPP disagreement rate {dis_rate:.3f} > {max_order_disagreement_rate} "
            f"(n_comparable={n_comp})"
        )
    elif dis_rate is not None and dis_rate > 0:
        caveats.append(f"Ordering disagreement rate {dis_rate:.3f} on sample")

    orient_notes = header_audit.get("orientation_note_counts") or {}
    if orient_notes.get("row_col_not_orthogonal"):
        blockers.append("ImageOrientationPatient row/col not orthogonal in sampled series")
    if orient_notes.get("ImageOrientationPatient missing") or orient_notes.get(
        "ImageOrientationPatient missing"
    ):
        caveats.append("Some orientation metadata missing")
    # Missing IOP entirely across sample
    series = header_audit.get("series") or []
    if series:
        n_missing_iop = sum(1 for s in series if (s.get("orientation") or {}).get("n_with_iop", 0) == 0)
        if n_missing_iop == len(series):
            caveats.append("ImageOrientationPatient absent on all sampled series")
        elif n_missing_iop:
            caveats.append(f"ImageOrientationPatient missing on {n_missing_iop}/{len(series)} sampled series")

    return {
        "blockers": blockers,
        "caveats": caveats,
        "ordering_ok": not any("ordering" in b.lower() or "InstanceNumber" in b for b in blockers),
        "orientation_ok": not any("Orientation" in b or "orthogonal" in b for b in blockers),
        "decode_ok": not any("failure rate" in b or "Degenerate" in b for b in blockers),
    }
