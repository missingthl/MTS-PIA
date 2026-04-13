#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/THL/miniconda3/envs/pia/bin/python}"
STAGE="${STAGE:-${1:-train_replace}}"
OUT_ROOT="${OUT_ROOT:-out/route_b_augmentation_${STAGE}_20260322}"
SEEDS="${SEEDS:-1,2,3}"
WAIT_FOR_LEGACY_BATCH="${WAIT_FOR_LEGACY_BATCH:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
LEGACY_PATTERN="${LEGACY_PATTERN:-run_bridge_lraes_nonseed_batch.sh}"

FIXED_SPLIT_DATASETS="${FIXED_SPLIT_DATASETS:-har,natops,fingermovements,selfregulationscp1,basicmotions,handmovementdirection,uwavegesturelibrary,epilepsy,atrialfibrillation,pendigits}"
TRAIN_REPLACE_EXT_DATASETS="${TRAIN_REPLACE_EXT_DATASETS:-mitbih}"
SEED_LIGHT_DATASETS="${SEED_LIGHT_DATASETS:-seed1,seediv,seedv}"

mkdir -p "$OUT_ROOT"

wait_for_legacy_batch() {
  if [[ "$WAIT_FOR_LEGACY_BATCH" != "1" ]]; then
    return 0
  fi
  while true; do
    match="$(pgrep -af "$LEGACY_PATTERN" || true)"
    if [[ -z "$match" ]]; then
      break
    fi
    echo "[route-b-augmentation-batch] waiting for legacy batch to finish pattern=$LEGACY_PATTERN"
    echo "$match"
    sleep "$WAIT_SECONDS"
  done
}

run_dataset() {
  local dataset="$1"
  local mode="$2"
  local ds_out="$OUT_ROOT/$dataset"
  local prefix
  case "$mode" in
    train_replace) prefix="augmentation_train_replace" ;;
    best_round_pool) prefix="augmentation_best_round_pool" ;;
    filtered_pool) prefix="augmentation_filtered_pool" ;;
    seed_light_best_round) prefix="seed_family_bridge_train_replace" ;;
    *) echo "[route-b-augmentation-batch] unknown mode=$mode" >&2; exit 1 ;;
  esac

  mkdir -p "$ds_out"
  if [[ -f "$ds_out/${prefix}_summary.csv" ]]; then
    echo "[route-b-augmentation-batch] skip dataset=$dataset mode=$mode reason=summary_exists"
    return 0
  fi

  echo "[route-b-augmentation-batch] start dataset=$dataset mode=$mode seeds=$SEEDS"
  "$PYTHON_BIN" -u scripts/route_b/run_route_b_augmentation_train.py \
    --dataset "$dataset" \
    --mode "$mode" \
    --seeds "$SEEDS" \
    --out-root "$OUT_ROOT" \
    2>&1 | tee "$ds_out/run.log"
  echo "[route-b-augmentation-batch] done dataset=$dataset mode=$mode"
}

wait_for_legacy_batch

case "$STAGE" in
  train_replace)
    IFS=',' read -r -a DS_ARR <<< "${FIXED_SPLIT_DATASETS},${TRAIN_REPLACE_EXT_DATASETS}"
    for ds in "${DS_ARR[@]}"; do
      ds="$(echo "$ds" | xargs)"
      [[ -z "$ds" ]] && continue
      run_dataset "$ds" "train_replace"
    done
    ;;
  best_round_pool)
    IFS=',' read -r -a DS_ARR <<< "$FIXED_SPLIT_DATASETS"
    for ds in "${DS_ARR[@]}"; do
      ds="$(echo "$ds" | xargs)"
      [[ -z "$ds" ]] && continue
      run_dataset "$ds" "best_round_pool"
    done
    ;;
  filtered_pool)
    IFS=',' read -r -a DS_ARR <<< "$FIXED_SPLIT_DATASETS"
    for ds in "${DS_ARR[@]}"; do
      ds="$(echo "$ds" | xargs)"
      [[ -z "$ds" ]] && continue
      run_dataset "$ds" "filtered_pool"
    done
    ;;
  seed_light_best_round)
    IFS=',' read -r -a DS_ARR <<< "$SEED_LIGHT_DATASETS"
    for ds in "${DS_ARR[@]}"; do
      ds="$(echo "$ds" | xargs)"
      [[ -z "$ds" ]] && continue
      run_dataset "$ds" "seed_light_best_round"
    done
    ;;
  *)
    echo "[route-b-augmentation-batch] unsupported stage=$STAGE" >&2
    exit 1
    ;;
esac

echo "[route-b-augmentation-batch] stage_done stage=$STAGE out=$OUT_ROOT"
