# External Baseline Groups

This document defines the paper-facing groups used for the external baseline
comparison table.  These groups are different from runtime phases: phases record
when a method entered the experiment matrix, while groups explain how the method
should be interpreted in the paper.

## Group 1: Temporal / Vicinal Heuristic Augmentations

Code facade:

```text
utils/external_baseline_methods/temporal_vicinal/
```

Methods:

- `raw_aug_jitter`
- `raw_aug_scaling`
- `raw_aug_timewarp`
- `raw_aug_magnitude_warping`
- `raw_aug_window_warping`
- `raw_aug_window_slicing`
- `raw_mixup`
- `raw_smote_flatten_balanced`

Role: fast raw-domain or vicinal baselines.  They are useful because they are
simple and cheap, but they do not explicitly control multivariate covariance
dependency structure.

## Group 2: Deep Generative Augmentations

Code facade:

```text
utils/external_baseline_methods/deep_generative/
```

Methods:

- `timevae_classwise_optional`
- `timegan_classwise`
- `timevqvae_classwise`
- `diffusionts_classwise`

Role: generator-based baselines.  They can model complex distributions, but add
extra generator training cost and heavier experimental protocols.

Current caveat: `timegan_classwise` is a compact PyTorch TimeGAN-style adapter,
not a line-by-line official TensorFlow implementation.  `TimeCAE` is not
implemented yet.

## Group 3: Analytical / Alignment-Based Structure-Preserving Generation

Code facade:

```text
utils/external_baseline_methods/alignment_structure/
```

Methods:

- `dba_sameclass`
- `wdba_sameclass`
- `spawner_sameclass_style`
- `jobda_cleanroom`
- `rgw_sameclass`
- `dgw_sameclass`

Role: external methods that preserve sequence structure through barycenters,
DTW alignment, guided warping, or supervised time-series warping.

## Not External: Internal Controls

Code facade:

```text
utils/external_baseline_methods/internal_controls/
```

Methods:

- `random_cov_state`
- `pca_cov_state`

These are mechanism controls, not external paper methods.  They should appear
in ablation/mechanism tables, not in the external-baseline main table.

## CLI Lookup

List all methods with their paper-facing group:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py
```

Filter one group:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py \
  --paper-group deep_generative
```
