#!/usr/bin/env bash
# Run a minimal SEED-V RAW EEG suite (manifold + optional dual).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seedv_raw_suite_$(date +%Y%m%d_%H%M%S).log"

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
RAW_CACHE_MODE="${RAW_CACHE_MODE:-auto}"
RAW_CHANNEL_POLICY="${RAW_CHANNEL_POLICY:-strict}"
RAW_SHRINKAGE="${RAW_SHRINKAGE:-}"
RA_MODE="${RA_MODE:-logeuclidean}"
FUSION_ALPHA="${FUSION_ALPHA:-0.6}"
RUN_DUAL="${RUN_DUAL:-1}"

base_args=(
  --dataset seedv
  --seedv-split "$SEEDV_SPLIT"
  --manifold-source raw
  --manifold-mode flat
  --manifold-raw-channels "$RAW_CHANNELS"
  --manifold-raw-repr "$RAW_REPR"
  --manifold-raw-cache "$RAW_CACHE_MODE"
  --manifold-raw-channel-policy "$RAW_CHANNEL_POLICY"
)
if [[ -n "$RAW_SHRINKAGE" ]]; then
  base_args+=(--raw-shrinkage "$RAW_SHRINKAGE")
fi
if [[ "$SEEDV_SPLIT" == "subject" ]]; then
  base_args+=(--seedv-subject-mode "$SUBJECT_MODE" --seedv-subject-k "$SUBJECT_K" --seedv-subject-seed "$SUBJECT_SEED")
fi

# 1) Manifold RAW, no RA
run python archive/legacy_code/pia_unified_demo.py --stream manifold "${base_args[@]}" --manifold-ra none

# 2) Manifold RAW, RA in classifier
run python archive/legacy_code/pia_unified_demo.py --stream manifold "${base_args[@]}" --manifold-ra "$RA_MODE"

# 3) Dual (DE spatial + RAW manifold with RA)
if [[ "$RUN_DUAL" == "1" ]]; then
  run python archive/legacy_code/pia_unified_demo.py --stream dual "${base_args[@]}" --manifold-ra "$RA_MODE" --fusion-alpha "$FUSION_ALPHA"
fi

echo "Done. Full log: ${LOG_FILE}"
