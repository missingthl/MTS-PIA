#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/THL/miniconda3/envs/pia/bin/python}"
OUT_ROOT="${OUT_ROOT:-out/route_b_unified_smoke_expansion_20260322}"
DATASETS="${DATASETS:-selfregulationscp1,natops}"
SEEDS="${SEEDS:-1,2,3}"
WAIT_FOR_LEGACY_BATCH="${WAIT_FOR_LEGACY_BATCH:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
LEGACY_PATTERN="${LEGACY_PATTERN:-run_bridge_lraes_nonseed_batch.sh}"

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
    echo "[route-b-unified-smoke-batch] waiting for legacy batch to finish pattern=$LEGACY_PATTERN"
    echo "$match"
    sleep "$WAIT_SECONDS"
  done
}

wait_for_legacy_batch

echo "[route-b-unified-smoke-batch] start datasets=$DATASETS seeds=$SEEDS out=$OUT_ROOT"
"$PYTHON_BIN" -u scripts/route_b/run_route_b_unified_probe.py \
  --datasets "$DATASETS" \
  --seeds "$SEEDS" \
  --out-root "$OUT_ROOT"

if [[ -f "$OUT_ROOT/final_coupling_summary.csv" ]]; then
  cp "$OUT_ROOT/final_coupling_summary.csv" "$OUT_ROOT/route_b_unified_smoke_expansion_summary.csv"
fi
if [[ -f "$OUT_ROOT/route_b_unified_conclusion.md" ]]; then
  cp "$OUT_ROOT/route_b_unified_conclusion.md" "$OUT_ROOT/route_b_unified_smoke_expansion_conclusion.md"
fi

echo "[route-b-unified-smoke-batch] done out=$OUT_ROOT"
