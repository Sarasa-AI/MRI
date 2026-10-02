#!/usr/bin/env python3
"""Validate a submission.csv against RSNA Knee 2026 schema rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.submission.validate import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to submission.csv")
    parser.add_argument(
        "--expected-ids-csv",
        type=Path,
        default=None,
        help="Optional CSV with StudyInstanceUID column (e.g. test.csv)",
    )
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()

    expected_ids = None
    if args.expected_ids_csv is not None:
        import pandas as pd

        expected_ids = (
            pd.read_csv(args.expected_ids_csv)["StudyInstanceUID"].astype(str).tolist()
        )

    result = validate_submission(
        args.submission,
        expected_ids=expected_ids,
        expected_row_count=args.expected_rows,
    )
    for w in result.warnings:
        print(f"WARNING: {w}")
    if not result.ok:
        for e in result.errors:
            print(f"ERROR: {e}")
        return 1
    print(f"OK rows={result.n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
