#!/usr/bin/env bash
# DCNet (spatial stream) tuning for SEED-V
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/seedv_spatial_tune_${TS}.log}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/seedv_spatial_tune_${TS}.csv}"

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  echo "head,lr,batch,epochs,freeze_bn,clipnorm,trial_acc,trial_f1,sample_acc,sample_f1" > "$SUMMARY_FILE"
fi

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

SEEDV_SPLIT="${SEEDV_SPLIT:-session}"  # trial | session | subject
SUBJECT_MODE="${SUBJECT_MODE:-loso}"   # loso | kfold
SUBJECT_K="${SUBJECT_K:-5}"
SUBJECT_SEED="${SUBJECT_SEED:-0}"

SPATIAL_BACKEND="${SPATIAL_BACKEND:-torch}"  # tf | torch
HEADS="${HEADS:-softmax,snn}"
LRS="${LRS:-1e-4,3e-4,1e-3}"
BATCHES="${BATCHES:-2048,4096}"
EPOCHS="${EPOCHS:-30}"
FREEZE_BN="${FREEZE_BN:-0}"
CLIPNORM="${CLIPNORM:-1.0}"

CONDA_ENV="${CONDA_ENV:-}"
PYTHON_CMD=(python)
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
fi

base_args=(
  --stream spatial
  --dataset seedv
  --backend "$SPATIAL_BACKEND"
  --seedv-split "$SEEDV_SPLIT"
  --epochs "$EPOCHS"
  --spatial-clipnorm "$CLIPNORM"
)
if [[ "$SEEDV_SPLIT" == "subject" ]]; then
  base_args+=(--seedv-subject-mode "$SUBJECT_MODE" --seedv-subject-k "$SUBJECT_K" --seedv-subject-seed "$SUBJECT_SEED")
fi
if [[ "$FREEZE_BN" == "1" ]]; then
  base_args+=(--freeze-bn)
fi

IFS=',' read -r -a HEAD_LIST <<< "$HEADS"
IFS=',' read -r -a LR_LIST <<< "$LRS"
IFS=',' read -r -a BS_LIST <<< "$BATCHES"

for head in "${HEAD_LIST[@]}"; do
  h="${head// /}"
  if [[ -z "$h" ]]; then
    continue
  fi
  for lr in "${LR_LIST[@]}"; do
    l="${lr// /}"
    if [[ -z "$l" ]]; then
      continue
    fi
    for bs in "${BS_LIST[@]}"; do
      b="${bs// /}"
      if [[ -z "$b" ]]; then
        continue
      fi
      echo -e "\n=== head=${h} lr=${l} batch=${b} epochs=${EPOCHS} ===" | tee -a "$LOG_FILE"
      run_log="${LOG_DIR}/seedv_spatial_tune_${TS}_${h}_lr${l}_bs${b}.log"
      "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py "${base_args[@]}" \
        --spatial-head "$h" \
        --spatial-lr "$l" \
        --batch-size "$b" \
        | tee "$run_log" | tee -a "$LOG_FILE"
      acc=$(awk '/spatial_trial_acc:/ {print $2; exit}' "$run_log")
      f1=$(awk '/spatial_trial_macro_f1:/ {print $2; exit}' "$run_log")
      sacc=$(awk '/spatial_sample_acc:/ {print $2; exit}' "$run_log")
      sf1=$(awk '/spatial_sample_macro_f1:/ {print $2; exit}' "$run_log")
      echo "${h},${l},${b},${EPOCHS},${FREEZE_BN},${CLIPNORM},${acc:-nan},${f1:-nan},${sacc:-nan},${sf1:-nan}" >> "$SUMMARY_FILE"
    done
  done
done

python - <<PY
import csv
best = None
with open("$SUMMARY_FILE", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            acc = float(row["trial_acc"])
            f1 = float(row["trial_f1"])
        except Exception:
            continue
        if best is None or acc > best[0] or (acc == best[0] and f1 > best[1]):
            best = (acc, f1, row)
if best:
    acc, f1, row = best
    print(f"Best DCNet: head={row['head']} lr={row['lr']} batch={row['batch']} acc={acc} f1={f1}")
else:
    print("No valid results parsed")
PY

echo -e "\nSummary CSV: ${SUMMARY_FILE}" | tee -a "$LOG_FILE"
echo "Done. Full log: ${LOG_FILE}"
