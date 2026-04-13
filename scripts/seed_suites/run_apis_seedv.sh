#!/usr/bin/env bash
# 全流程 SEED-V 实验脚本（DCNet + manifold），包含空间/流形/双流。
# 日志写入 logs/seedv_apis_suite_<timestamp>.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seedv_apis_suite_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

# 1) 空间：DCNet 单流
run python archive/legacy_code/pia_unified_demo.py --stream spatial --dataset seedv

# 2) 流形：trial-level
run python archive/legacy_code/pia_unified_demo.py --stream manifold --dataset seedv

# 3) 双流融合
run python archive/legacy_code/pia_unified_demo.py --stream dual --dataset seedv --fusion-alpha 0.6

echo "Done. Full log: ${LOG_FILE}"
