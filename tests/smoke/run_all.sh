#!/usr/bin/env bash
# Run all smoke tests for the MERGE-RNA restructure.
#
# Usage: bash tests/smoke/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export RNA_STRUCT_HOME="$REPO_ROOT"

cd "$REPO_ROOT"

echo "=== smoke: imports ==="
python "$SCRIPT_DIR/test_imports.py"

echo "=== smoke: construct ==="
python "$SCRIPT_DIR/test_construct.py"

echo "=== smoke: fit (end-to-end) ==="
python "$SCRIPT_DIR/test_fit_minimal.py"

echo
echo "ALL SMOKE TESTS PASSED"
