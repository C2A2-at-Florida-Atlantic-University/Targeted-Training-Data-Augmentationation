#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# This package distributes supporting modules as Python 3.10 bytecode (.pyc).

if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    raise RuntimeError(
        f"This package requires Python 3.10.x because supporting modules are .pyc-only. "
        f"Current Python: {sys.version.split()[0]}"
    )

from config_loader import load_config


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _csv_or_config(cli_value: str | None, cfg_value: Any) -> list[str]:
    return _as_list(cli_value) if cli_value else _as_list(cfg_value)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run TARGETED L1-PCA + sparse-outlier augmentation with DDPM/CTGAN."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", default="", help="Dataset section name under config.yaml -> datasets.")
    p.add_argument("--data-path", default="", help="Override CSV path from config dataset section.")
    p.add_argument("--label-col", default="", help="Override label column from config dataset section.")

    p.add_argument("--generators", default="", help="ddpm,ctgan or only one. Default comes from config.")
    p.add_argument("--models", default="", help="LR,SVM,SVM_lin,RF,DT or subset. Default comes from config.")

    p.add_argument("--drop-cols", default="", help="Comma-separated non-feature columns to drop")
    p.add_argument("--dataset-tag", default="")
    p.add_argument("--results-root", default="")
    p.add_argument("--num-runs", type=int, default=None)
    p.add_argument("--base-random-state", type=int, default=None)
    p.add_argument("--outlier-test-frac", type=float, default=None)
    p.add_argument("--min-outliers-per-class", type=int, default=None)
    p.add_argument("--max-aug-mode", default="", choices=["", "core_equal", "fixed"])
    p.add_argument("--max-non-core-aug-limit", type=int, default=None)
    p.add_argument("--tabddpm-repo-dir", default="")
    p.add_argument("--classifier-scaling", default="", choices=["", "standard", "minmax", "robust", "none"])
    p.add_argument("--ddpm-scaling", default="", choices=["", "minmax", "standard", "robust", "none"])
    return p.parse_args()


def run_one(args, cfg: dict, generator: str, models: list[str]) -> int:
    package_dir = Path(__file__).resolve().parent
    support_dir = package_dir / "supporting_functions"

    exp = cfg["experiment"]
    data = cfg["data"]
    out_det = cfg["outlier_detection"]
    aug = cfg["augmentation"]
    ddpm = cfg["ddpm"]
    ctgan = cfg["ctgan"]
    scaling = cfg.get("scaling", {})

    data_path_raw = args.data_path or str(data.get("data_path", "") or "")
    label_col = args.label_col or str(data.get("label_col", "") or "")
    if not data_path_raw:
        raise ValueError("Missing data path. ")
    if not label_col:
        raise ValueError("Missing label column. ")

    data_path = Path(data_path_raw).expanduser()
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()

    drop_cols = args.drop_cols if args.drop_cols else ",".join(_as_list(data.get("drop_cols", [])))
    dataset_tag = args.dataset_tag if args.dataset_tag else str(data.get("dataset_tag", "") or args.dataset or data_path.stem)
    results_root = args.results_root if args.results_root else str(data.get("results_root", "") or "")

    num_runs = args.num_runs if args.num_runs is not None else int(exp.get("num_runs", 1))
    base_random_state = args.base_random_state if args.base_random_state is not None else int(exp.get("base_random_state", 42))
    outlier_test_frac = args.outlier_test_frac if args.outlier_test_frac is not None else float(out_det.get("outlier_test_frac", 0.30))
    min_outliers = args.min_outliers_per_class if args.min_outliers_per_class is not None else int(out_det.get("min_outliers_per_class", 3))
    max_aug_mode = args.max_aug_mode if args.max_aug_mode else str(aug.get("max_aug_mode", "core_equal"))
    max_non_core = args.max_non_core_aug_limit if args.max_non_core_aug_limit is not None else int(aug.get("max_non_core_aug_limit", 50))
    tabddpm_repo = args.tabddpm_repo_dir if args.tabddpm_repo_dir else str(ddpm.get("repo_dir", ""))
    classifier_scaling = args.classifier_scaling if args.classifier_scaling else str(scaling.get("classifier", "none"))
    ddpm_scaling = args.ddpm_scaling if args.ddpm_scaling else str(scaling.get("ddpm", "minmax"))

    if generator not in {"ddpm", "ctgan"}:
        raise ValueError(f"Unsupported generator '{generator}'. Use ddpm or ctgan.")

    env = os.environ.copy()
    env.update({
        "DATA_PATH": str(data_path),
        "LABEL_COL": label_col,
        "DROP_COLS": drop_cols,
        "DATASET_TAG": dataset_tag,
        "GEN_NAME": generator,
        "SCENARIOS": "targeted",   
        "MODEL_NAMES": ",".join(models),
        "MODEL_CONFIG_JSON": json.dumps(cfg.get("models", {})),
        "NUM_RUNS": str(num_runs),
        "BASE_RANDOM_STATE": str(base_random_state),
        "OUTLIER_TEST_FRAC": str(outlier_test_frac),
        "MIN_OUTLIERS_PER_CLASS": str(min_outliers),
        "MAX_AUG_MODE": max_aug_mode,
        "MAX_NON_CORE_AUG_LIMIT": str(max_non_core),
        "CLASSIFIER_SCALING": classifier_scaling,
        "DDPM_SCALING": ddpm_scaling,
        "TABDDPM_REPO_DIR": str(Path(tabddpm_repo).expanduser()) if tabddpm_repo else "",
        "DDPM_DEVICE": str(ddpm.get("device", "auto")),
        "DDPM_NUM_TIMESTEPS": str(ddpm.get("num_timesteps", 500)),
        "DDPM_TRAIN_STEPS": str(ddpm.get("train_steps", 800)),
        "DDPM_LR": str(ddpm.get("lr", 0.001)),
        "DDPM_BATCH_SIZE": str(ddpm.get("batch_size", 16)),
        "DDPM_SEED": str(ddpm.get("seed", 12345)),
        "DDPM_ROUND_DECIMALS": str(ddpm.get("round_decimals", 3)),
        "CTGAN_EPOCHS": str(ctgan.get("epochs", 500)),
        "CTGAN_BATCH_SIZE": str(ctgan.get("batch_size", 16)),
        "CTGAN_PAC": str(ctgan.get("pac", 2)),
        "CTGAN_ROUND_DECIMALS": "none" if ctgan.get("round_decimals", None) is None else str(ctgan.get("round_decimals")),
        "PYTHONPATH": str(package_dir) + os.pathsep + str(support_dir) + os.pathsep + env.get("PYTHONPATH", ""),
    })

    if results_root:
        root = Path(results_root).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        env["RESULTS_DIR"] = str(root / f"results_{generator}_{dataset_tag}_targeted")

    print("\n" + "=" * 80)
    print(f"Running TARGETED only | dataset={dataset_tag} | generator={generator.upper()} | models={','.join(models)}")
    print(f"Scaling: classifier={classifier_scaling} | ddpm={ddpm_scaling}")
    print(f"Data: {data_path} | label_col={label_col}")
    print("=" * 80)

    cmd = [sys.executable, "-c", "import supporting_functions.hybrid_l1_knee_pipeline as p; p.main()"]
    return subprocess.call(cmd, cwd=str(package_dir), env=env)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.dataset or None)

    models = _csv_or_config(args.models, cfg["experiment"].get("models", ["LR", "SVM", "RF", "DT"]))
    if not models:
        raise ValueError("No models selected. Use --models LR,SVM or configure experiment.models.")

    generators = _csv_or_config(args.generators, cfg["experiment"].get("generators", ["ddpm"]))
    generators = [g.lower() for g in generators]
    bad_gens = [g for g in generators if g not in {"ddpm", "ctgan"}]
    if bad_gens:
        raise ValueError(f"Unsupported generator(s): {bad_gens}. Use ddpm and/or ctgan.")

    exit_codes = []
    for g in generators:
        exit_codes.append(run_one(args, cfg, g, models))

    bad = [c for c in exit_codes if c != 0]
    if bad:
        raise SystemExit(max(bad))


if __name__ == "__main__":
    main()
