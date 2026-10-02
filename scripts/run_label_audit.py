#!/usr/bin/env python3
"""RSNA-LABELS-001 — label audit + strategy comparison (CSV / report only).

Hypothesis: report-derived training labels (B–E) are high-quality enough that,
when consumed offline, they can lift gold-held-out imaging OOF macro AUC by
>0.005 vs gold-only (A) without any report dependence at inference.

This script runs the *label-quality* gate on train.csv (no DICOMs):
  1. Gold / missing / prevalence / co-occurrence / study consistency
  2. Report extraction (pos/neg/uncertain/missing, severity, negation)
  3. Strategies A–E supervision artifacts
  4. Report→gold agreement (macro/per-label AUC + bootstrap CI)
  5. KEEP / REJECT / NEEDS MORE EVIDENCE per strategy

Imaging OOF for A–E requires Kaggle GPU + DICOMs — see
notebooks/kaggle/05_label_strategies_oof.py.

Usage (Mac control plane, metadata only):
  python scripts/run_label_audit.py --train-csv data/raw/train.csv \\
      --output outputs/RSNA-LABELS-001-audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.data.loaders import load_train_metadata
from rsna_knee.labels.audit import run_label_audit
from rsna_knee.labels.laterality import assert_vision_only_inference_inputs
from rsna_knee.labels.targets import REPORT_COL


def _write_report(summary: dict, path: Path) -> None:
    lines = [
        "# RSNA-LABELS-001 — Label Audit Report",
        "",
        "## Hypothesis",
        "Report-derived silver labels (offline only) can improve gold-held-out",
        "imaging OOF macro AUC by >0.005 vs gold-only without inference report use.",
        "",
        "## Corpus",
        f"- Train studies: **{summary['n_train_studies']}**",
        f"- Gold studies (all 12 non-null): **{summary['n_gold_studies']}**",
        f"- Partial label rows: **{summary['missing_labels']['n_silver_report_only']}** report-only; "
        f"partial={summary['missing_labels']['all_or_nothing']} all-or-nothing",
        f"- Duplicate report groups: **{summary['study_consistency']['n_duplicate_report_groups']}** "
        f"({summary['study_consistency']['n_studies_in_duplicate_groups']} studies)",
        "",
        "## Report → gold agreement (label-quality OOF on n=58)",
        "",
        "| Predictor | Macro AUC | 95% CI |",
        "|---|---:|---:|",
    ]
    for key, title in [
        ("chance_vs_gold", "Chance (0.5)"),
        ("report_hard_vs_gold", "B Hard report"),
        ("report_soft_vs_gold", "C/D Soft report"),
    ]:
        s = summary[key]
        b = s["bootstrap_macro"]
        lines.append(
            f"| {title} | {s['macro_auc']:.4f} | "
            f"[{b['ci95_low']:.4f}, {b['ci95_high']:.4f}] |"
        )
    lines += ["", "### Per-label soft-report AUC vs gold", ""]
    lines.append("| Label | AUC | 95% CI |")
    lines.append("|---|---:|---:|")
    soft = summary["report_soft_vs_gold"]
    for lab, auc in soft["per_label_auc"].items():
        bb = soft["bootstrap_per_label"][lab]
        lines.append(
            f"| {lab} | {auc:.4f} | [{bb['ci95_low']:.4f}, {bb['ci95_high']:.4f}] |"
        )

    lines += ["", "## Strategy decisions", ""]
    lines.append("| Strategy | Decision | Report→gold macro | Reason |")
    lines.append("|---|---|---:|---|")
    for name, block in summary["strategies"].items():
        d = block["decision"]
        la = block["label_agreement"]
        macro = la.get("macro_auc", float("nan"))
        macro_s = "n/a" if macro != macro else f"{macro:.4f}"
        lines.append(
            f"| {name} | **{d['decision']}** | {macro_s} | {d['reason']} |"
        )

    lines += [
        "",
        "## Constraints checked",
        f"- Reports at inference: `{summary['constraints']['reports_at_inference']}`",
        f"- Eval on gold only: `{summary['constraints']['eval_on_gold_only']}`",
        f"- Test reports used: `{summary['constraints']['test_reports_used']}`",
        "",
        "## Imaging OOF status",
        "Not measured in this CSV-only audit (no local DICOMs by policy).",
        "Run `notebooks/kaggle/05_label_strategies_oof.py` on Kaggle GPU.",
        "",
        f"## Next experiment",
        summary.get("next_experiment") or "TBD",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/RSNA-LABELS-001-audit"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train = load_train_metadata(args.train_csv)
    if REPORT_COL not in train.columns:
        raise SystemExit("train.csv must include Report (train split only)")

    # Guardrail: this script never builds an inference feature set from Report.
    assert_vision_only_inference_inputs(["StudyInstanceUID", "ACL"])

    summary = run_label_audit(
        train,
        output_dir=args.output,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # Recommend next experiment from decisions
    decisions = {
        k: v["decision"]["decision"] for k, v in summary["strategies"].items()
    }
    soft_macro = summary["report_soft_vs_gold"]["macro_auc"]
    weak_labels = [
        k
        for k, v in summary["report_soft_vs_gold"]["per_label_auc"].items()
        if v == v and v < 0.55
    ]
    if soft_macro < 0.70:
        next_exp = (
            "RSNA-LABELS-002 (highest information): replace/augment the regex extractor "
            "with an offline train-only LLM (or stronger multilingual) report→label model; "
            f"pass if report→gold macro AUC >0.80 on the 58 gold studies "
            f"(current regex soft={soft_macro:.3f}; weak labels={weak_labels}). "
            "Only then run imaging OOF for D/E vs A. Do NOT abandon G3 — this REJECT "
            "applies to the current extractor, not to report supervision as a strategy class."
        )
    elif decisions.get("D_conf_weighted") == "NEEDS MORE EVIDENCE":
        next_exp = (
            "RSNA-LABELS-001b imaging OOF on Kaggle: train EfficientNet-B0 2.5D with "
            "strategy D (confidence-weighted soft silver) vs A (gold-only), same folds; "
            "pass if gold OOF macro AUC ↑ >0.005 with broad per-label lift."
        )
    else:
        next_exp = (
            "RSNA-LABELS-001b imaging OOF: E (gold + high-conf silver) vs A on Kaggle."
        )
    summary["next_experiment"] = next_exp
    (args.output / "label_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    _write_report(summary, args.output / "LABEL_AUDIT_REPORT.md")
    print(json.dumps({"output": str(args.output), "decisions": decisions, "next": next_exp}, indent=2))


if __name__ == "__main__":
    main()
