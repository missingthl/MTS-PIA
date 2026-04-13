#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_IN_PIA="$ROOT_DIR/scripts/run_in_pia.sh"
STAMP="$(date +%Y%m%d)"
OUT_DIR="$ROOT_DIR/out/_active/verify_current_stack_in_pia_${STAMP}"

cd "$ROOT_DIR"

echo "[1/4] Compile-check current active modules in pia"
"$RUN_IN_PIA" python -m py_compile \
  route_b_unified/pia_core.py \
  route_b_unified/augmentation_admission.py \
  route_b_unified/risk_aware_axis_controller.py \
  route_b_unified/regression/representation.py \
  scripts/route_b/run_route_b_pia_core_minimal_chain.py \
  scripts/route_b/run_route_b_pia_core_admission_control.py \
  scripts/route_b/run_route_b_pia_core_axis_refine.py \
  scripts/route_b/run_route_b_pia_core_axis_pullback_refine.py \
  scripts/route_b/run_route_b_pia_core_risk_aware_axis2.py \
  scripts/route_b/run_route_b_pia_core_logeuclidean_operator_probe.py \
  scripts/route_b/run_route_b_manifold_visual_probe.py \
  scripts/regression/run_route_b_zspace_regression_baseline.py

echo "[2/4] Help-check current runners in pia"
"$RUN_IN_PIA" scripts/route_b/run_route_b_pia_core_minimal_chain.py --help >/dev/null
"$RUN_IN_PIA" scripts/route_b/run_route_b_pia_core_admission_control.py --help >/dev/null
"$RUN_IN_PIA" scripts/route_b/run_route_b_pia_core_axis_refine.py --help >/dev/null
"$RUN_IN_PIA" scripts/route_b/run_route_b_pia_core_axis_pullback_refine.py --help >/dev/null
"$RUN_IN_PIA" scripts/route_b/run_route_b_pia_core_risk_aware_axis2.py --help >/dev/null
"$RUN_IN_PIA" scripts/route_b/run_route_b_pia_core_logeuclidean_operator_probe.py --help >/dev/null
"$RUN_IN_PIA" scripts/route_b/run_route_b_manifold_visual_probe.py --help >/dev/null
"$RUN_IN_PIA" scripts/regression/run_route_b_zspace_regression_baseline.py --help >/dev/null

echo "[3/4] Smoke-run current classification chain in pia"
rm -rf "$OUT_DIR"
"$RUN_IN_PIA" scripts/route_b/run_route_b_pia_core_minimal_chain.py \
  --datasets natops \
  --seeds 1 \
  --out-root "$OUT_DIR"

SUMMARY_FILE="$OUT_DIR/pia_core_minimal_chain_summary.csv"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  echo "error: expected summary not found: $SUMMARY_FILE" >&2
  exit 1
fi

echo "[4/4] Verification complete"
echo "summary: $SUMMARY_FILE"
