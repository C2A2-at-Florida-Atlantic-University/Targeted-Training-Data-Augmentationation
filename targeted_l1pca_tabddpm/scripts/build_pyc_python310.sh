#!/usr/bin/env bash
set -euo pipefail
PY_VER="$(python - <<'PYVER'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PYVER
)"
if [[ "$PY_VER" != "3.10" ]]; then
  echo "ERROR: This package must be compiled with Python 3.10.x. Current Python is $PY_VER" >&2
  echo "Run inside your Python 3.10 environment, for example: conda activate tabddpm" >&2
  exit 1
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/supporting_functions_source"
OUT_DIR="$ROOT_DIR/supporting_functions"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.pyc
rm -rf "$OUT_DIR/__pycache__"
cp "$SRC_DIR"/*.py "$OUT_DIR"/
touch "$OUT_DIR/__init__.py"
python -m compileall -q "$OUT_DIR"
for pyc in "$OUT_DIR"/__pycache__/*.cpython-310.pyc; do
  base="$(basename "$pyc")"
  mod="${base%%.cpython-310.pyc}"
  mv "$pyc" "$OUT_DIR/${mod}.pyc"
done
rm -rf "$OUT_DIR/__pycache__"
find "$OUT_DIR" -maxdepth 1 -type f -name '*.py' ! -name '__init__.py' -delete
python "$ROOT_DIR/scripts/verify_pyc_imports.py"
echo "OK: Python 3.10 .pyc files created in $OUT_DIR"
echo "Before public GitHub upload, remove supporting_functions_source/ or rely on .gitignore."
