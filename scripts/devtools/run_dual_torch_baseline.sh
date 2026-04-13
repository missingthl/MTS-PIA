#!/usr/bin/env bash
# Dual-stream baseline (no PIA): spatial (torch) + manifold (numpy) + fusion.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seedv_dual_torch_$(date +%Y%m%d_%H%M%S).log"

DATASET="${DATASET:-seedv}"
SEEDV_SPLIT="${SEEDV_SPLIT:-trial}"
BACKEND="${BACKEND:-torch}"
FUSION_ALPHA="${FUSION_ALPHA:-0.6}"

MANIFOLD_SOURCE="${MANIFOLD_SOURCE:-de}"
MANIFOLD_MODE="${MANIFOLD_MODE:-band}"
MANIFOLD_CLASSIFIER="${MANIFOLD_CLASSIFIER:-svm}"
MANIFOLD_RA="${MANIFOLD_RA:-logeuclidean}"

PYTHON_CMD=(conda run -n pia python)

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

base_args=(
  --dataset "$DATASET"
  --seedv-split "$SEEDV_SPLIT"
  --manifold-source "$MANIFOLD_SOURCE"
  --manifold-mode "$MANIFOLD_MODE"
  --manifold-classifier "$MANIFOLD_CLASSIFIER"
  --manifold-ra "$MANIFOLD_RA"
)

if [[ "$SEEDV_SPLIT" == "subject" ]]; then
  base_args+=(--seedv-subject-mode "${SUBJECT_MODE:-loso}" --seedv-subject-k "${SUBJECT_K:-5}" --seedv-subject-seed "${SUBJECT_SEED:-0}")
fi

run "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream spatial --backend "$BACKEND" "${base_args[@]}"
run "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream manifold "${base_args[@]}"
run "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream dual --backend "$BACKEND" "${base_args[@]}" --fusion-alpha "$FUSION_ALPHA"

echo "Done. Full log: ${LOG_FILE}"
