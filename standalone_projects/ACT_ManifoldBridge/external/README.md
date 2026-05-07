# Vendored External Repositories

This directory is **not** the external-baseline comparison matrix.

It only contains heavyweight third-party or official-code trees that are
vendored for optional generator-style baselines:

```text
external/
  DiffusionTS/
    Vendored Diffusion-TS code used by `utils/diffusionts_wrapper.py`.

  TimeVQVAE/
    Vendored TimeVQVAE code used by `utils/timevqvae_wrapper.py` when present.
```

Most external comparison arms are project-native or clean-room adapters and live
elsewhere:

```text
utils/external_baseline_methods/
  raw transforms
  Mixup
  DBA / wDBA
  SPAWNER-style
  RGW / DGW
  JobDA-cleanroom
  TimeVAE-style adapter
  SMOTE
  random/PCA covariance-state controls

utils/external_aug_dispatch.py
  method name -> augmentation builder dispatch

utils/external_runner_registry.py
  phase arms, method metadata, locked-result-root guard

scripts/run_external_baselines_phase1.py
  matrix runner; historical filename kept for CLI compatibility
```

So if you are looking for `dba_sameclass`, `wdba_sameclass`, `rgw_sameclass`,
`dgw_sameclass`, `jobda_cleanroom`, or raw augmentation methods, do **not** look
under this `external/` directory.  Look under
`utils/external_baseline_methods/`.

If this project is reorganized in a future breaking cleanup, this directory
should be renamed to `third_party/` or `vendor/`.  For now it keeps its current
name because wrappers and tracked third-party paths already depend on it.
