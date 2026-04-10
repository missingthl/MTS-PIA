#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${PIA_ENV_NAME:-pia}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_in_pia.sh <script.py> [args...]
  scripts/run_in_pia.sh <command> [args...]

Examples:
  scripts/run_in_pia.sh scripts/run_route_b_pia_core_minimal_chain.py --help
  scripts/run_in_pia.sh python -m py_compile scripts/run_route_b_pia_core_minimal_chain.py
  scripts/run_in_pia.sh bash scripts/verify_current_stack_in_pia.sh

Notes:
  - Defaults to conda env: pia
  - Override with: PIA_ENV_NAME=<env> scripts/run_in_pia.sh ...
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found in PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"

first_arg="$1"
if [[ "$first_arg" == *.py ]]; then
  exec conda run --no-capture-output -n "$ENV_NAME" python "$@"
fi

exec conda run --no-capture-output -n "$ENV_NAME" "$@"
