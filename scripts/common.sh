#!/usr/bin/env bash
set -euo pipefail
EMS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EMS_ROOT"
export PYTHONUTF8=1
export PIP_CACHE_DIR="$EMS_ROOT/.runtime/pip-cache"
mkdir -p logs build .runtime/tmp
export TMPDIR="$EMS_ROOT/.runtime/tmp/"
EMS_PYTHON="$EMS_ROOT/.venv/bin/python"
assert_tools() {
  "$EMS_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "EMS needs Python 3.12"'
}
