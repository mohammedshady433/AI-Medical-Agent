#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
[[ "$(uname -s)" == Darwin ]] || { echo 'Run this setup on macOS.'; exit 1; }
EMS_BOOTSTRAP_PYTHON="${EMS_PYTHON312:-python3.12}"
"$EMS_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12)'
"$EMS_BOOTSTRAP_PYTHON" -m venv .venv
"$EMS_PYTHON" -m pip install --upgrade pip
"$EMS_PYTHON" -m pip install -r requirements.lock
"$EMS_PYTHON" scripts/validate.py
echo 'Setup complete. Run: .venv/bin/python scripts/test_console.py   See docs/reference/MACOS_SETUP.md for local models and voice.'
