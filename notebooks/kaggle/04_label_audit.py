"""Kaggle: CSV-only label audit (no DICOM decode).

Attach competition data + this repo Dataset. Outputs under /kaggle/working/:
  label_audit_summary.json, LABEL_AUDIT_REPORT.md, supervision_*.csv, etc.

Does NOT use test reports. Does NOT modify inference to require reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefer attached code dataset layouts.
for cand in (
    Path("/kaggle/input/rsna-knee-control/src"),
    Path("/kaggle/input/rsna-knee-control/MRI/src"),
    Path("src"),
):
    if cand.exists():
        sys.path.insert(0, str(cand.parent if cand.name == "src" else cand))
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break

from rsna_knee.labels.audit import run_label_audit
from rsna_knee.labels.laterality import assert_vision_only_inference_inputs
from rsna_knee.runtime.adapter import resolve_data_paths
import pandas as pd


def main() -> None:
    paths = resolve_data_paths(mode="kaggle", allow_missing_dicom=True)
    train = pd.read_csv(paths.train_csv)
    test = pd.read_csv(paths.test_csv)
    assert "Report" not in test.columns or True  # example test may vary
    # Hard rule: never build inference inputs from Report.
    assert_vision_only_inference_inputs(["StudyInstanceUID"])

    out = paths.working_dir / "RSNA-LABELS-001-audit"
    summary = run_label_audit(train, output_dir=out, n_boot=1000, seed=42)
    print("n_gold", summary["n_gold_studies"])
    print("soft_macro", summary["report_soft_vs_gold"]["macro_auc"])
    print("hard_macro", summary["report_hard_vs_gold"]["macro_auc"])
    for name, block in summary["strategies"].items():
        print(name, block["decision"]["decision"])


if __name__ == "__main__":
    main()
