#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/results/csta_direction_specificity_stress_v1/resnet1d_s123"
DATASETS="atrialfibrillation,ering,handmovementdirection,handwriting,japanesevowels,natops,racketsports"
SEEDS="1,2,3"
COMMON_ARGS=(
  --datasets "${DATASETS}"
  --seeds "${SEEDS}"
  --epochs 30
  --batch-size 64
  --lr 1e-3
  --patience 10
  --val-ratio 0.2
  --multiplier 10
  --k-dir 10
  --pia-gamma 0.1
  --backbone resnet1d
)

python "${ROOT_DIR}/scripts/run_external_baselines_phase1.py" \
  --out-root "${OUT_ROOT}" \
  --arms csta_topk_uniform_top5,csta_top1_current,csta_template_random_within_bank,random_cov_state,pca_cov_state \
  --eta-safe 0.75 \
  "${COMMON_ARGS[@]}"

EXTRA_ROOTS=()
for ETA in 0.75 1.0 1.25; do
  ETA_ROOT="${OUT_ROOT}_eta_stress/e${ETA}"
  EXTRA_ROOTS+=("${ETA_ROOT}")
  python "${ROOT_DIR}/scripts/run_external_baselines_phase1.py" \
    --out-root "${ETA_ROOT}" \
    --arms csta_topk_uniform_top5,csta_template_random_within_bank,random_cov_state \
    --eta-safe "${ETA}" \
    "${COMMON_ARGS[@]}"
done

IFS=,
EXTRA_JOINED="${EXTRA_ROOTS[*]}"
unset IFS
python "${ROOT_DIR}/scripts/build_direction_specificity_stress_summary.py" \
  --out-root "${OUT_ROOT}" \
  --extra-roots "${EXTRA_JOINED}"
