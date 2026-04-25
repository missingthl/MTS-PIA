# Results Directory

This directory contains tracked evidence tables, active ACT-V2 result bundles,
and a reduced set of historical outputs that still matter for audit or paper
assembly.

## Current Active Result Roots

- `mba_zpia_vnext/`
  - direction-engine comparison around `lraes` vs `zpia`.
- `v2_grand_sweep_rc3_osf_safeclip/`
  - current ACT-V2 grand sweep bundle.
- `v2_taxonomy_analysis/`
  - regime taxonomy summaries.
- `paper_matrix_v2_final/`
  - paper-facing matrix tables and per-run outputs.

## Historical But Still Useful

- `paper_theory_atlas_*`
- `paper_report/`
- selected `full_sweep_*` or `gap_closure_*` folders that still support paper evidence

Older failed side-branch folders have been removed from the active tree so this
directory better reflects the current research line.

## Result Schema

Every new result table should include at least:

- `dataset`
- `seed`
- `status`
- `pipeline`
- `algo`
- `model`
- `base_f1`
- `act_f1`
- `gain`

ACT-V2 adaptive runs should additionally expose routing / utilization fields
such as:

- `direction_bank_source`
- `feedback_weight_mean`
- `last_weighted_aug_ce_loss`
- `router_p_lraes_final`
- `router_p_zpia_final`
- `adaptive_best_engine_final`
