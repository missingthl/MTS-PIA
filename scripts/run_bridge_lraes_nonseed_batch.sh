#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/THL/miniconda3/envs/pia/bin/python}"
OUT_ROOT="${OUT_ROOT:-out/bridge_lraes_all_nonseed_20260322}"
SEEDS="${SEEDS:-1,2,3}"
DATASETS="${DATASETS:-har,mitbih,fingermovements,basicmotions,handmovementdirection,uwavegesturelibrary,epilepsy,atrialfibrillation,pendigits}"

mkdir -p "$OUT_ROOT"

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"

for ds in "${DATASET_ARR[@]}"; do
  ds="$(echo "$ds" | xargs)"
  [[ -z "$ds" ]] && continue
  ds_out="$OUT_ROOT/$ds"
  mkdir -p "$ds_out"
  summary_glob=$(find "$ds_out" -maxdepth 1 -type f -name "*_pilot_summary.csv" | head -n 1 || true)
  if [[ -n "$summary_glob" ]]; then
    echo "[bridge-lraes-batch] skip dataset=$ds reason=summary_exists path=$summary_glob"
    continue
  fi
  echo "[bridge-lraes-batch] start dataset=$ds out=$ds_out"
  "$PYTHON_BIN" -u scripts/run_bridge_lraes_pilot.py \
    --dataset "$ds" \
    --seeds "$SEEDS" \
    --out-root "$ds_out" \
    2>&1 | tee "$ds_out/run.log"
  echo "[bridge-lraes-batch] done dataset=$ds"
done
