#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if find "$ROOT_DIR/supporting_functions" -maxdepth 1 -type f -name '*.py' ! -name '__init__.py' | grep -q .; then
  echo "ERROR: Visible .py support files found in supporting_functions/. Run scripts/build_pyc_python310.sh first." >&2
  exit 1
fi
for f in hybrid_l1_knee_pipeline l1pca_sbfk_v0 tabddpm_generate_outliers_per_class tabgan_generate_outliers_per_class; do
  if [[ ! -f "$ROOT_DIR/supporting_functions/$f.pyc" ]]; then
    echo "ERROR: Missing supporting_functions/$f.pyc" >&2
    exit 1
  fi
done
if [[ -d "$ROOT_DIR/supporting_functions_source" ]]; then
  echo "WARNING: supporting_functions_source/ exists. It is ignored by .gitignore, but remove it before public upload if needed." >&2
fi
echo "OK: release check passed for .pyc-only support modules."
