#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
assert_tools
"$EMS_PYTHON" scripts/validate.py
"$EMS_PYTHON" -m pytest -q --basetemp=build/pytest-macos --junitxml=logs/backend-macos.xml
