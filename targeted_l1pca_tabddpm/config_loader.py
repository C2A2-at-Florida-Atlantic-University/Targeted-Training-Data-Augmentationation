#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "num_runs": 1,
        "base_random_state": 42,
        "generators": ["ddpm"],
        "models": ["LR", "SVM", "RF", "DT"],
    },
    "data": {
        "data_path": "",
        "label_col": "",
        "drop_cols": [],
        "dataset_tag": "",
        "results_root": "",
    },
    "outlier_detection": {
        "outlier_test_frac": 0.30,
        "min_outliers_per_class": 3,
    },
    "augmentation": {
        "max_aug_mode": "core_equal",
        "max_non_core_aug_limit": 50,
    },
    "scaling": {
        "classifier": "none",
        "ddpm": "minmax",
    },
    "ddpm": {
        "repo_dir": "/mnt/beegfs/home/sshukla2020/Training_data_aug/tab-ddpm",
        "device": "auto",
        "num_timesteps": 500,
        "train_steps": 800,
        "lr": 0.001,
        "batch_size": 16,
        "seed": 12345,
        "round_decimals": 3,
    },
    "ctgan": {
        "epochs": 500,
        "batch_size": 16,
        "pac": 2,
        "round_decimals": None,
    },
    "models": {
        "LR": {"max_iter": 3000, "solver": "lbfgs", "C": 1.0, "random_state": 42},
        "SVM": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "SVM_lin": {"kernel": "linear", "C": 1.0},
        "RF": {"n_estimators": 200, "bootstrap": False, "random_state": 42},
        "DT": {"max_depth": 6, "random_state": 42},
    },
    "datasets": {},
}

MERGE_BLOCKS = {"experiment", "data", "outlier_detection", "augmentation", "scaling", "ddpm", "ctgan", "models"}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _split_global_and_datasets(raw: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    datasets = raw.get("datasets", {}) or {}
    global_cfg = {k: v for k, v in raw.items() if k != "datasets"}
    return global_cfg, datasets


def load_config(config_path: Optional[str] = None, dataset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load config.yaml and optionally merge one dataset-specific section.

    Merge order:
      DEFAULT_CONFIG < global config blocks < datasets.<dataset>

    Dataset sections may override any of these blocks:
      experiment, data, outlier_detection, augmentation, scaling, ddpm, ctgan, models
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if config_path:
        cpath = Path(config_path).expanduser()
        if not cpath.is_absolute():
            cpath = (Path.cwd() / cpath).resolve()
        if not cpath.exists():
            raise FileNotFoundError(f"Config file not found: {cpath}")
        raw = _load_yaml(cpath)
    else:
        raw = {}

    global_cfg, datasets = _split_global_and_datasets(raw)
    cfg = _deep_update(cfg, global_cfg)
    cfg["datasets"] = datasets

    if dataset:
        if dataset not in datasets:
            available = sorted(datasets.keys())
            raise KeyError(f"Dataset section '{dataset}' not found. Available datasets: {available}")
        ds_cfg = datasets[dataset] or {}
        allowed = {k: v for k, v in ds_cfg.items() if k in MERGE_BLOCKS}
        cfg = _deep_update(cfg, allowed)
        cfg["active_dataset"] = dataset
        if not cfg["data"].get("dataset_tag"):
            cfg["data"]["dataset_tag"] = dataset
    else:
        cfg["active_dataset"] = ""

    return cfg
