#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARM="${ARM:-e0}"
DATASETS_DEFAULT="natops,fingermovements,selfregulationscp1"
DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-100}"
TRAIN_BS="${TRAIN_BS:-64}"
TEST_BS="${TEST_BS:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOG_EVERY="${LOG_EVERY:-10}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-out/_active/verify_resnet1d_stage2_framework_compare_$(date +%Y%m%d)}"

if [[ "$ARM" != "e0" && "$ARM" != "e1" && "$ARM" != "e2" ]]; then
  echo "error: ARM must be one of e0/e1/e2, got: $ARM" >&2
  exit 1
fi

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"

for ds in "${DATASET_LIST[@]}"; do
  ds_trimmed="$(echo "$ds" | xargs)"
  if [[ -z "$ds_trimmed" ]]; then
    continue
  fi
  echo "[resnet-framework] framework=$ARM dataset=$ds_trimmed seed=$SEED device=$DEVICE epochs=$EPOCHS"
  scripts/run_in_pia.sh python scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py \
    --dataset "$ds_trimmed" \
    --arm "$ARM" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --train-batch-size "$TRAIN_BS" \
    --test-batch-size "$TEST_BS" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --log-every "$LOG_EVERY" \
    --device "$DEVICE" \
    --out-root "$OUT_ROOT" \
    --run-tag "stage2_resnet1d_${ARM}_${ds_trimmed}_seed${SEED}"
done
