#!/usr/bin/env bash
# Reproduce SEED_DEMO DCNet-only baseline (seed1, flat input).
# Logs to logs/seed_demo_baseline_<timestamp>.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seed_demo_baseline_$(date +%Y%m%d_%H%M%S).log"

PYTHON_CMD=("${PYTHON:-python}")

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

run "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py \
  --dataset seed1 \
  --stream spatial \
  --backend torch \
  --spatial-input flat \
  --seed-de-mode author \
  --spatial-align-baseline \
  --epochs 80 \
  --batch-size 2048

echo "Done. Full log: ${LOG_FILE}"
