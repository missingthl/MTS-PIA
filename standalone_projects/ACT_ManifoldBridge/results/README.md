# Results Directory

This directory contains tracked evidence tables, historical sweeps, and new
experiment outputs.

## Publishable Evidence Tables

- `paper_matrix_v2_final/`: paper-facing matrix tables and per-run outputs.
- `paper_theory_atlas_v1/`: consolidated theory-atlas evidence.
- `v2_taxonomy_analysis/`: regime taxonomy summaries.

## Historical Sweep Folders

Folders such as `full_throttle_sweep/`, `gap_closure_v*/`,
`isolated_sweep_v1/`, and `parallel_sweep_v1/` are historical experiment
records. They are useful for audit/reproduction, but are not the main project
API.

## New Experiment Convention

New runs should write into a clearly named folder, for example:

```text
results/wavelet_mba_v1/<dataset-or-matrix-name>/
results/wavelet_dual_object_v2/<dataset-or-matrix-name>/
results/mba_white_edit_vnext/<dataset-or-matrix-name>/
```

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

Wavelet runs should also include:

- `wavelet_name`
- `wavelet_object_mode`
- `wavelet_level`
- `wavelet_level_eff`
- `cA_identity_bridge_error_mean`
- `idwt_identity_recon_error_mean`

Dual-object wavelet runs should also include:

- `wavelet_secondary_detail_level`
- `wavelet_detail_gamma_scale`
- `cDm_identity_bridge_error_mean`
- `cDm_transport_error_logeuc_mean`
- `dual_transport_error_logeuc_mean`

MBA white-edit runs should also include:

- `edit_mode`
- `edit_basis`
- `edit_alpha_scale`
- `white_identity_error_mean`
- `recolor_transport_error_logeuc_mean`
- `edit_energy_mean`
- `feedback_weight_mean`
- `last_aug_margin_mean`
