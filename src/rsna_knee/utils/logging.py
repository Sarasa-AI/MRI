"""Simple local logging (CSV/JSON results table)."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Mapping


def get_logger(name: str = "rsna_knee", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def append_results_row(row: Mapping[str, Any], path: str | Path) -> Path:
    """Append one experiment summary row to a CSV results table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))
    return path
