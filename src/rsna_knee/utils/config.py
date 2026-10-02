"""YAML config loading (simple, Hydra-compatible field layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment_config(
    experiment_yaml: str | Path,
    *,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load an experiment YAML and merge referenced defaults if present.

    Experiment files may list `defaults:` as a list of relative YAML paths
    under config_root (mirrors a lightweight Hydra defaults list).
    """
    exp_path = Path(experiment_yaml)
    cfg = load_yaml(exp_path)
    defaults = cfg.pop("defaults", []) or []
    root = Path(config_root) if config_root else exp_path.parent.parent
    merged: dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, dict):
            # e.g. {runtime: local} → configs/runtime/local.yaml
            for group, name in item.items():
                merged = deep_merge(merged, load_yaml(root / str(group) / f"{name}.yaml"))
        elif isinstance(item, str):
            merged = deep_merge(merged, load_yaml(root / f"{item}.yaml"))
    return deep_merge(merged, cfg)
