#!/usr/bin/env bash
# SEED1 initial suite:
# - Spatial: DCNet (Keras) frame-level acc
#
# Logs to logs/seed1_suite_<timestamp>.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seed1_suite_$(date +%Y%m%d_%H%M%S).log"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2048}"

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

# 0) Train once + report spatial acc
run python archive/legacy_code/pia_unified_demo.py --dataset seed1 --stream spatial \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"

echo "Done. Full log: ${LOG_FILE}"
