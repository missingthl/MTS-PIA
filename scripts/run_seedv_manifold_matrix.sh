#!/usr/bin/env bash
# Batch sweep for SEED-V manifold stream across split protocols and RA modes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/seedv_manifold_matrix_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT_DIR"

echo "Writing logs to ${LOG_FILE}"

run() {
  echo -e "\n>>> $*"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

RA_MODES=("none" "euclidean" "logeuclidean")
SPLITS=("session" "subject")

MANIFOLD_MODE="${MANIFOLD_MODE:-band}"
MANIFOLD_EPS="${MANIFOLD_EPS:-1e-3}"
MANIFOLD_C="${MANIFOLD_C:-1.0}"
MANIFOLD_CLASSIFIER="${MANIFOLD_CLASSIFIER:-svm}"
MANIFOLD_SOURCE="${MANIFOLD_SOURCE:-de}"

SUBJECT_MODE="${SUBJECT_MODE:-loso}"
SUBJECT_K="${SUBJECT_K:-5}"
SUBJECT_SEED="${SUBJECT_SEED:-0}"

for split in "${SPLITS[@]}"; do
  for ra in "${RA_MODES[@]}"; do
    cmd=(
      python archive/legacy_code/pia_unified_demo.py
      --stream manifold
      --dataset seedv
      --seedv-split "$split"
      --manifold-source "$MANIFOLD_SOURCE"
      --manifold-mode "$MANIFOLD_MODE"
      --manifold-eps "$MANIFOLD_EPS"
      --manifold-classifier "$MANIFOLD_CLASSIFIER"
      --manifold-C "$MANIFOLD_C"
      --manifold-ra "$ra"
    )
    if [[ "$split" == "subject" ]]; then
      cmd+=(--seedv-subject-mode "$SUBJECT_MODE" --seedv-subject-k "$SUBJECT_K" --seedv-subject-seed "$SUBJECT_SEED")
    fi
    run "${cmd[@]}"
  done

done

echo "Done. Full log: ${LOG_FILE}"
