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
- `wavelet_level`
- `wavelet_level_eff`
- `cA_identity_bridge_error_mean`
- `idwt_identity_recon_error_mean`
