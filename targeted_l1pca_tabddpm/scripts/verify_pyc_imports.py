#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    raise RuntimeError(f"Verification requires Python 3.10.x. Current: {sys.version.split()[0]}")
root = Path(__file__).resolve().parents[1]
support = root / "supporting_functions"
sys.path.insert(0, str(root)); sys.path.insert(0, str(support))
required = [support / f"{m}.pyc" for m in ["hybrid_l1_knee_pipeline", "l1pca_sbfk_v0", "tabddpm_generate_outliers_per_class", "tabgan_generate_outliers_per_class"]]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing compiled support files: " + ", ".join(missing))
import supporting_functions.hybrid_l1_knee_pipeline  # noqa: F401
print("OK: supporting_functions.hybrid_l1_knee_pipeline imports under Python 3.10")
