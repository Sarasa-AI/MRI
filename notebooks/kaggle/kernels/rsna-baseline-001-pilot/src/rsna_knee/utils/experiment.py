"""Experiment card schema — required fields for every experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


REQUIRED_EXPERIMENT_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "configuration",
    "seed",
    "model",
    "input_geometry",
    "data_selection",
    "labels",
    "folds",
    "augmentation",
    "optimizer",
    "scheduler",
    "metric",
    "runtime",
    "oof_predictions",
    "conclusion",
)


@dataclass
class ExperimentCard:
    experiment_id: str
    configuration: str
    seed: int
    model: str
    input_geometry: str
    data_selection: str
    labels: str
    folds: str
    augmentation: str
    optimizer: str
    scheduler: str
    metric: str
    runtime: str
    oof_predictions: str
    conclusion: str
    notes: str = ""
    status: str = "draft"  # draft | running | complete | abandoned

    def validate(self) -> None:
        data = asdict(self)
        missing = [k for k in REQUIRED_EXPERIMENT_FIELDS if data.get(k) in (None, "")]
        if missing:
            raise ValueError(f"ExperimentCard missing required fields: {missing}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def save_experiment_card(card: ExperimentCard, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_experiment_card(path: str | Path) -> ExperimentCard:
    data = json.loads(Path(path).read_text())
    known = {f.name for f in fields(ExperimentCard)}
    filtered = {k: v for k, v in data.items() if k in known}
    card = ExperimentCard(**filtered)
    card.validate()
    return card
