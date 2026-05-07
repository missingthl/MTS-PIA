# External Baseline Method Implementations

This package contains the **actual external-baseline comparison arms** that are
implemented inside the ACT/CSTA project.

The root `external/` directory is only for vendored third-party repositories.
Do not use it as the baseline matrix map.

## Method Map

```text
external_baseline_methods/
  base.py
    Shared `ExternalAugResult` dataclass and small array helpers.

  raw_jitter.py
    raw_aug_jitter

  raw_scaling.py
    raw_aug_scaling

  raw_timewarp.py
    raw_aug_timewarp

  raw_magnitude_warping.py
    raw_aug_magnitude_warping

  raw_window_warping.py
    raw_aug_window_warping

  raw_window_slicing.py
    raw_aug_window_slicing

  raw_mixup.py
    raw_mixup

  dba.py
    dba_sameclass

  wdba.py
    wdba_sameclass

  spawner.py
    spawner_sameclass_style

  rgw.py
    rgw_sameclass

  dgw.py
    dgw_sameclass

  jobda.py
    jobda_cleanroom_augmented_set

  smote.py
    raw_smote_flatten_balanced

  random_cov_state.py
    random_cov_state

  pca_cov_state.py
    pca_cov_state

  timevae.py
    timevae_classwise_optional

  diffusionts.py
    diffusionts_classwise wrapper entry

  timevqvae.py
    timevqvae_classwise wrapper entry
```

## Related Files

```text
utils/external_baselines.py
  Compatibility facade that re-exports this package for historical imports.

utils/external_aug_dispatch.py
  Dispatches a method name to one function in this package.

utils/external_runner_registry.py
  Defines phase arm groups, method metadata, CSTA passthrough fields, and
  locked-root protection.

utils/external_baseline_manifest.py
  Searchable catalog used by `scripts/list_external_baselines.py`.

scripts/run_external_baselines_phase1.py
  Matrix runner.  It should orchestrate experiments, not implement algorithms.
```

## Naming Rule

Prefer `one_method_or_family.py` files here.  If a method needs official or
vendored third-party code, put only the project adapter here and keep the
third-party tree under `external/`.
