"""Label-quality audit: gold vs report-derived labels (training-time only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.evaluation.bootstrap import bootstrap_macro_auc, bootstrap_per_label_auc
from rsna_knee.evaluation.metrics import macro_roc_auc
from rsna_knee.evaluation.splits import gold_label_mask
from rsna_knee.labels.report_parser import (
    STATE_MISSING,
    STATE_NEGATIVE,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    extract_reports_frame,
    soft_matrix,
    state_matrix,
)
from rsna_knee.labels.strategies import (
    STRATEGY_NAMES,
    StrategyName,
    build_all_strategies,
    hard_report_predictions,
    report_labels_as_predictions,
)
from rsna_knee.labels.targets import REPORT_COL, STUDY_ID_COL, TARGET_COLUMNS


def _gold_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    return train_df.loc[gold_label_mask(train_df)].copy()


def label_prevalence(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gold = gold_label_mask(train_df)
    for col in TARGET_COLUMNS:
        s_all = train_df[col]
        s_gold = train_df.loc[gold, col]
        rows.append(
            {
                "label": col,
                "n_non_null_all": int(s_all.notna().sum()),
                "n_pos_gold": int((s_gold == 1).sum()),
                "n_neg_gold": int((s_gold == 0).sum()),
                "prevalence_gold": float(s_gold.mean()) if len(s_gold) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def label_cooccurrence(train_df: pd.DataFrame) -> pd.DataFrame:
    gold = _gold_frame(train_df)
    y = gold[list(TARGET_COLUMNS)].astype(float)
    return y.corr(method="pearson")


def study_consistency(train_df: pd.DataFrame, extracted: pd.DataFrame) -> dict[str, Any]:
    """Duplicate-report groups and within-group soft-label variance."""
    reports = train_df[[STUDY_ID_COL, REPORT_COL]].copy()
    reports[STUDY_ID_COL] = reports[STUDY_ID_COL].astype(str)
    merged = reports.merge(extracted, on=STUDY_ID_COL, how="left")
    vc = merged["report_group"].value_counts()
    dup_groups = int((vc > 1).sum())
    dup_studies = int(vc[vc > 1].sum())
    # Soft-label std within duplicate groups
    soft_cols = [f"soft__{c}" for c in TARGET_COLUMNS]
    group_std = (
        merged.groupby("report_group")[soft_cols].std().mean(axis=1).dropna()
        if dup_groups
        else pd.Series(dtype=float)
    )
    return {
        "n_unique_report_groups": int(vc.shape[0]),
        "n_duplicate_report_groups": dup_groups,
        "n_studies_in_duplicate_groups": dup_studies,
        "max_group_size": int(vc.max()) if len(vc) else 0,
        "mean_within_dup_soft_std": float(group_std.mean()) if len(group_std) else 0.0,
        "note": (
            "Duplicate reports must stay within one CV fold; shared text is not "
            "independent validation evidence."
        ),
    }


def state_summary(extracted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    states = state_matrix(extracted)
    for j, name in enumerate(TARGET_COLUMNS):
        col = states[:, j]
        rows.append(
            {
                "label": name,
                "n_positive": int((col == STATE_POSITIVE).sum()),
                "n_negative": int((col == STATE_NEGATIVE).sum()),
                "n_uncertain": int((col == STATE_UNCERTAIN).sum()),
                "n_missing": int((col == STATE_MISSING).sum()),
                "mention_rate": float(((col == STATE_POSITIVE) | (col == STATE_NEGATIVE) | (col == STATE_UNCERTAIN)).mean()),
                "silence_rate": float((col == STATE_MISSING).mean()),
                "pos_among_mentioned": float(
                    (col == STATE_POSITIVE).sum()
                    / max(((col == STATE_POSITIVE) | (col == STATE_NEGATIVE) | (col == STATE_UNCERTAIN)).sum(), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def contradiction_audit(extracted: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Flag gold studies where report state contradicts the expert label."""
    gold = _gold_frame(train_df)
    gold_ids = set(gold[STUDY_ID_COL].astype(str))
    ex = extracted[extracted[STUDY_ID_COL].astype(str).isin(gold_ids)].copy()
    g = gold.set_index(STUDY_ID_COL)
    rows = []
    for _, r in ex.iterrows():
        sid = str(r[STUDY_ID_COL])
        for col in TARGET_COLUMNS:
            gold_v = float(g.loc[sid, col])
            st = r[f"state__{col}"]
            if st == STATE_POSITIVE and gold_v == 0.0:
                kind = "false_positive_vs_gold"
            elif st == STATE_NEGATIVE and gold_v == 1.0:
                kind = "false_negative_vs_gold"
            elif st == STATE_MISSING and gold_v == 1.0:
                kind = "silent_miss_positive"
            else:
                continue
            rows.append(
                {
                    STUDY_ID_COL: sid,
                    "label": col,
                    "gold": gold_v,
                    "state": st,
                    "soft": r[f"soft__{col}"],
                    "conf": r[f"conf__{col}"],
                    "kind": kind,
                    "evidence": r.get(f"evidence__{col}", ""),
                }
            )
    return pd.DataFrame(rows)


def severity_uncertainty_tables(extracted: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sev_rows = []
    unc_rows = []
    for name in TARGET_COLUMNS:
        sev = extracted[f"sev__{name}"]
        st = extracted[f"state__{name}"]
        sev_rows.append(
            {
                "label": name,
                "mean_severity_if_positive": float(sev[st == STATE_POSITIVE].mean())
                if (st == STATE_POSITIVE).any()
                else float("nan"),
                "frac_high_sev": float(((st == STATE_POSITIVE) & (sev >= 0.99)).mean()),
                "frac_low_sev": float(((st == STATE_POSITIVE) & (sev <= 0.34)).mean()),
            }
        )
        unc_rows.append(
            {
                "label": name,
                "n_uncertain": int((st == STATE_UNCERTAIN).sum()),
                "uncertain_rate": float((st == STATE_UNCERTAIN).mean()),
            }
        )
    return {"severity": pd.DataFrame(sev_rows), "uncertainty": pd.DataFrame(unc_rows)}


def score_against_gold(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Score a prediction frame on the gold subset only."""
    gold = _gold_frame(train_df)
    pred = pred_df.copy()
    pred[STUDY_ID_COL] = pred[STUDY_ID_COL].astype(str)
    gold_ids = gold[[STUDY_ID_COL]].copy()
    gold_ids[STUDY_ID_COL] = gold_ids[STUDY_ID_COL].astype(str)
    y_true_df = gold.set_index(gold[STUDY_ID_COL].astype(str))[list(TARGET_COLUMNS)]
    pred_idx = pred.set_index(STUDY_ID_COL)
    common = [i for i in gold_ids[STUDY_ID_COL] if i in pred_idx.index]
    y_true = y_true_df.loc[common].to_numpy(dtype=float)
    y_score = pred_idx.loc[common, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    nan_frac = float(np.isnan(y_score).mean())
    y_score = np.where(np.isnan(y_score), 0.28, y_score)

    metrics = macro_roc_auc(y_true, y_score)
    boot = bootstrap_macro_auc(y_true, y_score, n_boot=n_boot, seed=seed)
    per_boot = bootstrap_per_label_auc(y_true, y_score, n_boot=n_boot, seed=seed)
    return {
        "n_gold": int(len(common)),
        "nan_frac_before_fill": nan_frac,
        "macro_auc": metrics.macro_auc,
        "per_label_auc": metrics.per_label_auc,
        "bootstrap_macro": boot.to_dict(),
        "bootstrap_per_label": {k: v.to_dict() for k, v in per_boot.items()},
        "evaluated_labels": list(metrics.evaluated_labels),
        "skipped_labels": list(metrics.skipped_labels),
    }


def prediction_correlation(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """Per-label Pearson correlation between two prediction frames on shared IDs."""
    aa = a.copy()
    bb = b.copy()
    aa[STUDY_ID_COL] = aa[STUDY_ID_COL].astype(str)
    bb[STUDY_ID_COL] = bb[STUDY_ID_COL].astype(str)
    m = aa.merge(bb, on=STUDY_ID_COL, suffixes=("_a", "_b"))
    cors = []
    per = {}
    for col in TARGET_COLUMNS:
        ca = pd.to_numeric(m[f"{col}_a"], errors="coerce")
        cb = pd.to_numeric(m[f"{col}_b"], errors="coerce")
        mask = ca.notna() & cb.notna()
        if mask.sum() < 3 or ca[mask].nunique() < 2 or cb[mask].nunique() < 2:
            per[col] = float("nan")
            continue
        r = float(ca[mask].corr(cb[mask]))
        per[col] = r
        if r == r:
            cors.append(r)
    return {
        "mean_pearson": float(np.mean(cors)) if cors else float("nan"),
        "per_label": per,
    }


def rank_correlation_macro(per_a: dict[str, float], per_b: dict[str, float]) -> float:
    """Spearman correlation of per-label AUC vectors (label ranking agreement)."""
    names = [n for n in TARGET_COLUMNS if per_a.get(n) == per_a.get(n) and per_b.get(n) == per_b.get(n)]
    if len(names) < 3:
        return float("nan")
    s1 = pd.Series({n: per_a[n] for n in names})
    s2 = pd.Series({n: per_b[n] for n in names})
    return float(s1.corr(s2, method="spearman"))


def lift_breadth(per_base: dict[str, float], per_new: dict[str, float], *, eps: float = 0.005) -> dict[str, Any]:
    deltas = {}
    for n in TARGET_COLUMNS:
        a, b = per_base.get(n), per_new.get(n)
        if a == a and b == b:
            deltas[n] = float(b - a)
    if not deltas:
        return {"n_improved": 0, "n_regressed": 0, "max_label": None, "max_delta": float("nan"), "broad": False}
    improved = [k for k, v in deltas.items() if v > eps]
    regressed = [k for k, v in deltas.items() if v < -eps]
    max_label = max(deltas, key=deltas.get)
    return {
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "improved_labels": improved,
        "regressed_labels": regressed,
        "max_label": max_label,
        "max_delta": deltas[max_label],
        "deltas": deltas,
        "broad": len(improved) >= 3 and deltas[max_label] < 0.5 * sum(max(0.0, d) for d in deltas.values()) + 1e-9,
    }


def decide_strategy(
    *,
    name: StrategyName,
    macro_auc: float,
    baseline_macro: float,
    bootstrap: dict,
    breadth: dict,
    imaging_oof_available: bool,
) -> dict[str, Any]:
    """KEEP / REJECT / NEEDS MORE EVIDENCE for a label strategy.

    Label-agreement gate (report→gold) is necessary but not sufficient for
    imaging KEEP. Imaging OOF lift remains required for final KEEP.
    """
    delta = float(macro_auc - baseline_macro) if baseline_macro == baseline_macro else float("nan")
    ci_low = bootstrap.get("ci95_low", float("nan"))
    ci_high = bootstrap.get("ci95_high", float("nan"))

    if name == "A_gold":
        return {
            "decision": "KEEP",
            "reason": (
                "Gold-only is the honest imaging floor and evaluation anchor. "
                "Always retained; not a silver-label strategy."
            ),
            "delta_vs_chance": float(macro_auc - 0.5),
        }

    # Chance baseline for report-as-predictor is 0.5; imaging baseline separate.
    if macro_auc < 0.55:
        return {
            "decision": "REJECT",
            "reason": f"Report-label agreement vs gold is near chance (macro AUC={macro_auc:.3f}).",
            "delta_vs_chance": float(macro_auc - 0.5),
        }

    if not imaging_oof_available:
        # Provisional decision from label-quality evidence only.
        if macro_auc >= 0.80:
            decision = "NEEDS MORE EVIDENCE"
            reason = (
                f"Strong report→gold macro AUC={macro_auc:.3f}. Usable for silver "
                "supervision pending imaging OOF vs RSNA-BASELINE-001 "
                "(gold-held-out ↑ >0.005, broad lift, bootstrap CI)."
            )
        elif macro_auc >= 0.70:
            decision = "NEEDS MORE EVIDENCE"
            reason = (
                f"Moderate report→gold agreement (macro AUC={macro_auc:.3f}). "
                "Prefer confidence-weighted / high-conf silver; confirm imaging OOF. "
                "Do not ship hard labels for findings with per-label AUC≲0.55."
            )
        else:
            decision = "REJECT"
            reason = (
                f"Weak report→gold agreement for this extractor (macro AUC={macro_auc:.3f}). "
                "Reject silver supervision from THIS label source; do not abandon G3 "
                "conceptually — upgrade the offline extractor first."
            )
        return {
            "decision": decision,
            "reason": reason,
            "delta_vs_chance": float(macro_auc - 0.5),
            "label_agreement_macro_auc": macro_auc,
            "bootstrap_ci95": [ci_low, ci_high],
            "breadth": breadth,
        }

    # Imaging path (when provided): require delta > 0.005 and CI helpful.
    if delta > 0.005 and (ci_low == ci_low and ci_low > 0):
        if breadth.get("broad"):
            decision = "KEEP"
            reason = f"Imaging OOF lift {delta:+.4f} is broad."
        else:
            decision = "NEEDS MORE EVIDENCE"
            reason = f"Imaging OOF lift {delta:+.4f} appears single-label driven ({breadth.get('max_label')})."
    elif delta > 0.005:
        decision = "NEEDS MORE EVIDENCE"
        reason = f"Point lift {delta:+.4f} but bootstrap CI overlaps zero / unstable on n=58."
    else:
        decision = "REJECT"
        reason = f"No material imaging OOF lift ({delta:+.4f})."
    return {
        "decision": decision,
        "reason": reason,
        "delta_vs_baseline": delta,
        "bootstrap_ci95": [ci_low, ci_high],
        "breadth": breadth,
    }


def run_label_audit(
    train_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    n_boot: int = 1000,
    seed: int = 42,
    imaging_oof: dict[StrategyName, dict] | None = None,
) -> dict[str, Any]:
    """Full label audit + strategy comparison. Writes artifacts under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert REPORT_COL in train_df.columns, "Report required for audit (train only)"
    gold = _gold_frame(train_df)
    assert len(gold) > 0, "No gold-labeled studies found"

    extracted = extract_reports_frame(train_df)
    extracted.to_csv(output_dir / "report_extractions.csv", index=False)

    prevalence = label_prevalence(train_df)
    prevalence.to_csv(output_dir / "label_prevalence.csv", index=False)
    cooc = label_cooccurrence(train_df)
    cooc.to_csv(output_dir / "label_cooccurrence_gold.csv")
    states = state_summary(extracted)
    states.to_csv(output_dir / "report_state_summary.csv", index=False)
    cons = study_consistency(train_df, extracted)
    (output_dir / "study_consistency.json").write_text(json.dumps(cons, indent=2))
    contradictions = contradiction_audit(extracted, train_df)
    contradictions.to_csv(output_dir / "gold_report_contradictions.csv", index=False)
    sev_unc = severity_uncertainty_tables(extracted)
    sev_unc["severity"].to_csv(output_dir / "severity_summary.csv", index=False)
    sev_unc["uncertainty"].to_csv(output_dir / "uncertainty_summary.csv", index=False)

    strategies = build_all_strategies(train_df, extracted=extracted)
    for name, bundle in strategies.items():
        bundle.studies.to_csv(output_dir / f"supervision_{name}_labels.csv", index=False)
        bundle.weights.to_csv(output_dir / f"supervision_{name}_weights.csv", index=False)
        (output_dir / f"supervision_{name}_meta.json").write_text(json.dumps(bundle.meta, indent=2))

    # Strategy scoring: report labels as predictors vs gold (label-quality OOF)
    soft_pred = report_labels_as_predictions(extracted)
    hard_pred = hard_report_predictions(extracted)
    soft_pred.to_csv(output_dir / "report_soft_predictions.csv", index=False)
    hard_pred.to_csv(output_dir / "report_hard_predictions.csv", index=False)

    soft_scores = score_against_gold(train_df, soft_pred, n_boot=n_boot, seed=seed)
    hard_scores = score_against_gold(train_df, hard_pred, n_boot=n_boot, seed=seed)
    # Chance / majority baseline on gold
    chance_pred = gold[[STUDY_ID_COL]].copy()
    for col in TARGET_COLUMNS:
        chance_pred[col] = 0.5
    chance_scores = score_against_gold(train_df, chance_pred, n_boot=n_boot, seed=seed)

    # Per-strategy: use supervision soft/hard values on gold IDs as "predictions"
    strategy_scores: dict[str, Any] = {}
    baseline_per = soft_scores["per_label_auc"]  # soft report as reference ranking
    for name in STRATEGY_NAMES:
        bundle = strategies[name]
        if name == "A_gold":
            # Gold labels predicting themselves → AUC=1; instead report chance floor
            # and mark as evaluation anchor, not a report strategy.
            scored = {
                "macro_auc": float("nan"),
                "per_label_auc": {c: float("nan") for c in TARGET_COLUMNS},
                "bootstrap_macro": {"point": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")},
                "note": "A_gold is the imaging-train / eval anchor, not a report predictor.",
            }
            breadth = {"n_improved": 0, "n_regressed": 0, "broad": False, "max_label": None, "max_delta": 0.0}
            decision = decide_strategy(
                name=name,
                macro_auc=0.5,
                baseline_macro=0.5,
                bootstrap=scored["bootstrap_macro"],
                breadth=breadth,
                imaging_oof_available=bool(imaging_oof and name in imaging_oof),
            )
        elif name == "B_hard":
            scored = hard_scores
            breadth = lift_breadth(chance_scores["per_label_auc"], scored["per_label_auc"])
            decision = decide_strategy(
                name=name,
                macro_auc=scored["macro_auc"],
                baseline_macro=0.5,
                bootstrap=scored["bootstrap_macro"],
                breadth=breadth,
                imaging_oof_available=bool(imaging_oof and name in imaging_oof),
            )
        elif name in ("C_soft", "D_conf_weighted"):
            scored = soft_scores
            breadth = lift_breadth(chance_scores["per_label_auc"], scored["per_label_auc"])
            decision = decide_strategy(
                name=name,
                macro_auc=scored["macro_auc"],
                baseline_macro=0.5,
                bootstrap=scored["bootstrap_macro"],
                breadth=breadth,
                imaging_oof_available=bool(imaging_oof and name in imaging_oof),
            )
        else:  # E
            # High-conf hard on gold subset: mask low-conf to prior
            pred = hard_pred.copy()
            # For E, unmentioned stay prior — already in soft; blend:
            soft_m = soft_matrix(extracted)
            conf = extracted[[STUDY_ID_COL] + [f"conf__{c}" for c in TARGET_COLUMNS]].copy()
            conf[STUDY_ID_COL] = conf[STUDY_ID_COL].astype(str)
            pred = pred.copy()
            pred[STUDY_ID_COL] = pred[STUDY_ID_COL].astype(str)
            # Use soft where conf high else prior
            ex_ids = extracted[STUDY_ID_COL].astype(str).tolist()
            for j, col in enumerate(TARGET_COLUMNS):
                c = extracted[f"conf__{col}"].to_numpy()
                s = soft_m[:, j]
                st = extracted[f"state__{col}"].to_numpy()
                vals = np.where(
                    (c >= 0.60) & ((st == STATE_POSITIVE) | (st == STATE_NEGATIVE)),
                    np.where(st == STATE_POSITIVE, 1.0, 0.0),
                    s,
                )
                pred[col] = vals
            scored = score_against_gold(train_df, pred, n_boot=n_boot, seed=seed)
            breadth = lift_breadth(chance_scores["per_label_auc"], scored["per_label_auc"])
            decision = decide_strategy(
                name=name,
                macro_auc=scored["macro_auc"],
                baseline_macro=0.5,
                bootstrap=scored["bootstrap_macro"],
                breadth=breadth,
                imaging_oof_available=bool(imaging_oof and name in imaging_oof),
            )

        # Attach imaging OOF if provided
        img = (imaging_oof or {}).get(name)
        if img:
            decision = decide_strategy(
                name=name,
                macro_auc=float(img["macro_auc"]),
                baseline_macro=float(imaging_oof["A_gold"]["macro_auc"]),  # type: ignore[index]
                bootstrap=img.get("bootstrap_macro", {}),
                breadth=lift_breadth(
                    imaging_oof["A_gold"]["per_label_auc"],  # type: ignore[index]
                    img["per_label_auc"],
                ),
                imaging_oof_available=True,
            )

        pred_corr = prediction_correlation(soft_pred, hard_pred)
        strategy_scores[name] = {
            "label_agreement": scored,
            "decision": decision,
            "supervision_meta": strategies[name].meta,
            "rank_corr_vs_soft_report": rank_correlation_macro(
                baseline_per, scored.get("per_label_auc", {})
            ),
            "prediction_corr_soft_vs_hard": pred_corr,
            "lift_breadth_vs_chance": breadth,
            "imaging_oof": img,
            "reproducible": True,
        }

    summary = {
        "n_train_studies": int(len(train_df)),
        "n_gold_studies": int(len(gold)),
        "n_report_studies": int(train_df[REPORT_COL].notna().sum()),
        "partial_labels": int(
            (train_df[list(TARGET_COLUMNS)].notna().any(axis=1) & ~gold_label_mask(train_df)).sum()
        ),
        "study_consistency": cons,
        "report_soft_vs_gold": soft_scores,
        "report_hard_vs_gold": hard_scores,
        "chance_vs_gold": chance_scores,
        "strategies": strategy_scores,
        "constraints": {
            "reports_at_inference": False,
            "eval_on_gold_only": True,
            "test_reports_used": False,
            "n_gold_warning": (
                "Do not declare imaging success from n=58 alone; require bootstrap "
                "CI and breadth, then confirm on public LB."
            ),
        },
        "next_experiment": None,  # filled by caller / report writer
    }

    # Missing-label analysis
    summary["missing_labels"] = {
        "all_or_nothing": bool(summary["partial_labels"] == 0),
        "n_silver_report_only": int(len(train_df) - len(gold)),
        "gold_definition": "all 12 targets non-null",
    }

    (output_dir / "label_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
