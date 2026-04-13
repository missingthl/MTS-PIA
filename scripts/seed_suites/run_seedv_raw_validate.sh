#!/usr/bin/env bash
# Validate SEED-V RAW EEG manifold pipeline (no-RA vs RA vs dual).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seedv_raw_validate_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

SEEDV_SPLIT="${SEEDV_SPLIT:-session}"   # trial | session | subject
SUBJECT_MODE="${SUBJECT_MODE:-loso}"    # loso | kfold
SUBJECT_K="${SUBJECT_K:-5}"
SUBJECT_SEED="${SUBJECT_SEED:-0}"

RAW_CHANNELS="${RAW_CHANNELS:-62}"
RAW_REPR="${RAW_REPR:-cov}"
RAW_CACHE_FIRST="${RAW_CACHE_FIRST:-refresh}"
RAW_CACHE_MODE="${RAW_CACHE_MODE:-auto}"
RAW_CHANNEL_POLICY="${RAW_CHANNEL_POLICY:-strict}"
RAW_SHRINKAGE="${RAW_SHRINKAGE:-}"
RA_MODE="${RA_MODE:-logeuclidean}"
FUSION_ALPHA="${FUSION_ALPHA:-0.6}"
RUN_DUAL="${RUN_DUAL:-1}"
CLEAR_CACHE="${CLEAR_CACHE:-0}"
CONDA_ENV="${CONDA_ENV:-}"

PYTHON_CMD=(python)
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
fi

if [[ "$CLEAR_CACHE" == "1" ]]; then
  echo "Clearing raw cov cache..." | tee -a "$LOG_FILE"
  rm -rf "${ROOT_DIR}/data/SEED_V/cache/raw_cov"/*
fi

base_args=(
  --dataset seedv
  --seedv-split "$SEEDV_SPLIT"
  --manifold-source raw
  --manifold-mode flat
  --manifold-raw-channels "$RAW_CHANNELS"
  --manifold-raw-repr "$RAW_REPR"
  --manifold-raw-channel-policy "$RAW_CHANNEL_POLICY"
)
if [[ -n "$RAW_SHRINKAGE" ]]; then
  base_args+=(--raw-shrinkage "$RAW_SHRINKAGE")
fi
if [[ "$SEEDV_SPLIT" == "subject" ]]; then
  base_args+=(--seedv-subject-mode "$SUBJECT_MODE" --seedv-subject-k "$SUBJECT_K" --seedv-subject-seed "$SUBJECT_SEED")
fi

# 1) Manifold RAW, no RA (builds cache by default)
run "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream manifold "${base_args[@]}" --manifold-raw-cache "$RAW_CACHE_FIRST" --manifold-ra none

# 2) Manifold RAW, RA in classifier
run "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream manifold "${base_args[@]}" --manifold-raw-cache "$RAW_CACHE_MODE" --manifold-ra "$RA_MODE"

# 3) Dual (DE spatial + RAW manifold with RA)
if [[ "$RUN_DUAL" == "1" ]]; then
  run "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream dual "${base_args[@]}" --manifold-raw-cache "$RAW_CACHE_MODE" --manifold-ra "$RA_MODE" --fusion-alpha "$FUSION_ALPHA"
fi

echo -e "\n=== Extracted summaries ===" | tee -a "$LOG_FILE"
rg -n "\\[(Manifold|Dual)\\] summary" -A 12 "$LOG_FILE" | tee -a "$LOG_FILE" || true
echo "Done. Full log: ${LOG_FILE}"
