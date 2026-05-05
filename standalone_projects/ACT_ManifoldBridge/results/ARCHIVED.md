# Archived Result Directories

This file marks result directories that have been **superseded** by later runs
or were **development-only**.  They are retained for audit but should **not** be
used as the primary source for paper tables or current conclusions.

**Date marked**: 2026-05-05

---

## Stale — Superseded by Later Runs

| Directory | Superseded By | Reason |
|-----------|--------------|--------|
| `csta_step3_diagnostic_sweep/` | `csta_step3_diagnostic_sweep_etafix/` | Pre-eta-fix; retained as audit evidence only |
| `csta_external_baselines_phase1/` | `final20_minimal_baseline_v1/` | Pilot7-only Phase1; Final20 has full coverage |
| `csta_external_baselines_phase2/` | `csta_external_baselines_phase2_new/` | Re-run with corrected config |
| `csta_external_baselines_phase2_new/resnet1d_s123_partial_failed_launch_20260502_1838/` | `csta_external_baselines_phase2_new/resnet1d_s123/` | Failed launch; partial data only |
| `full_scale_resnet1d_v1/` | `final20_minimal_baseline_v1/` | **Uses eta_safe=0.5** (not canonical 0.75); kept for reference |

## Stale — Pilot / Development Only

| Directory | Reason |
|-----------|--------|
| `csta_group_pilot_v0/` | Pilot version; superseded by `csta_group_deep_ablation_v1/` |
| `csta_group_deep_ablation_v1/` | Pilot7 only; not Final20 scale |
| `csta_external_baselines_local/` | Local debugging runs |
| `csta_external_baselines_phase3/` | Partial Phase3 smoke; not full matrix |
| `pilot_patchtst_v1/` | Pilot7 only; not Final20 scale |
| `act_core/` | Early ACT prototype; superseded |
| `rc4_dry_run/` | Dry run only |
| `identity_check/` | Debugging |
| `full_scale_external_baselines_v1/` | Incomplete (only NATOPs); superseded |
| `full_scale_minirocket_v1/` | Historical; MiniRocket not primary backbone |

## Debug / Smoke (safe to remove)

| Directory | Reason |
|-----------|--------|
| `csta_debug_test/` | Debug output |
| `smoke_test_timevqvae/` | Smoke test artifacts |
| `smoke_test_timevqvae_v2/` | Smoke test artifacts |
| `_smoke/` | Smoke run probe outputs |
| `_logs/` | Execution logs |

---

## Active Directories (current paper sources)

See [CANONICAL_RESULTS.md](CANONICAL_RESULTS.md) for the authoritative list.
