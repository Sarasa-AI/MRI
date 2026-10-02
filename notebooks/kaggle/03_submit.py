"""Kaggle template: submission notebook (internet OFF).

Attach:
  - competition dataset
  - optional code Dataset containing this repo's src/
  - optional model weights Dataset (future)

Writes: /kaggle/working/submission.csv
"""

from __future__ import annotations

import pandas as pd

from rsna_knee.inference.predict import build_uniform_submission, write_submission
from rsna_knee.labels.targets import STUDY_ID_COL
from rsna_knee.runtime.adapter import resolve_data_paths
from rsna_knee.submission.validate import validate_submission


def main() -> None:
    paths = resolve_data_paths(mode="kaggle", allow_missing_dicom=True)
    test = pd.read_csv(paths.test_csv)
    # Replace with real model inference for competitive submissions.
    sub = build_uniform_submission(test)
    out = paths.working_dir / "submission.csv"
    write_submission(sub, out, expected_ids=test[STUDY_ID_COL].astype(str).tolist())
    result = validate_submission(out, expected_ids=test[STUDY_ID_COL].astype(str).tolist())
    print(result)
    assert result.ok, result.errors


if __name__ == "__main__":
    main()
