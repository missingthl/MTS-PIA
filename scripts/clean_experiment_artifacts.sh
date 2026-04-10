#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "No logs directory at $LOG_DIR"
  exit 0
fi

echo "Pruning logs in $LOG_DIR"

# Remove zero-byte logs
find "$LOG_DIR" -type f -name "*.log" -size 0 -print -delete || true

# Remove noisy raw suite/validate logs
rm -f "$LOG_DIR"/seedv_raw_suite_*.log
rm -f "$LOG_DIR"/seedv_raw_validate_*.log
rm -f "$LOG_DIR"/seedv_raw_shrinkage_*_s*.log

# Keep only the latest shrinkage grid log/csv
latest_grid_log=$(ls -1t "$LOG_DIR"/seedv_raw_shrinkage_grid_*.log 2>/dev/null | head -n1 || true)
latest_grid_csv=$(ls -1t "$LOG_DIR"/seedv_raw_shrinkage_grid_*.csv 2>/dev/null | head -n1 || true)
if [[ -n "$latest_grid_log" ]]; then
  find "$LOG_DIR" -maxdepth 1 -type f -name "seedv_raw_shrinkage_grid_*.log" ! -path "$latest_grid_log" -delete
fi
if [[ -n "$latest_grid_csv" ]]; then
  find "$LOG_DIR" -maxdepth 1 -type f -name "seedv_raw_shrinkage_grid_*.csv" ! -path "$latest_grid_csv" -delete
fi

# Clean python bytecode caches
find "$ROOT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$ROOT_DIR" -type f -name "*.pyc" -delete

echo "Done."
