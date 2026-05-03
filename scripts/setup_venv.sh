#!/usr/bin/env bash
set -euo pipefail

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN=python3.11
elif command -v python311 >/dev/null 2>&1; then
  PYTHON_BIN=python311
else
  echo "WARN: Python 3.11 is not installed; falling back to python3." >&2
  echo "      On CachyOS/Arch, install package python311 if this fallback stops working." >&2
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > requirements.lock.txt
python - <<'PY'
import torch, xgboost, sklearn, pyarrow
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("xgboost", xgboost.__version__)
print("sklearn", sklearn.__version__)
print("pyarrow", pyarrow.__version__)
PY
