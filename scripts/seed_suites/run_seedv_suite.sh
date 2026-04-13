#!/usr/bin/env bash
# Run a small suite of SEED-V experiments in sequence.
# Logs are appended to logs/seedv_suite_<timestamp>.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seedv_suite_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT_DIR"

echo "Writing logs to ${LOG_FILE}"

run() {
  echo -e "\n>>> $*"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

# 1) Spatial-only
run python archive/legacy_code/pia_unified_demo.py --stream spatial --dataset seedv

# 2) Manifold-only
run python archive/legacy_code/pia_unified_demo.py --stream manifold --dataset seedv

# 3) Dual-stream (DCNet + manifold)
run python archive/legacy_code/pia_unified_demo.py --stream dual --dataset seedv --fusion-alpha 0.6

echo "Done. Full log: ${LOG_FILE}"
