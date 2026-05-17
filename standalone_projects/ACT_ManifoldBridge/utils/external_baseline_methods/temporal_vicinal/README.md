# Temporal / Vicinal Heuristic Augmentations

Paper-facing group for fast raw-domain and vicinal baselines.

Included arms:

- `raw_aug_jitter`
- `raw_aug_scaling`
- `raw_aug_timewarp`
- `raw_aug_magnitude_warping`
- `raw_aug_window_warping`
- `raw_aug_window_slicing`
- `raw_mixup`
- `manifold_mixup`
- `raw_smote_flatten_balanced`

These are external comparison baselines for the main table.  They should not be
confused with CSTA covariance-state controls such as `random_cov_state`.

Status notes:

- The raw transforms are standard transform baselines following the time-series
  augmentation survey / uchidalab convention; they are not one independent paper
  per transform.
- `raw_mixup` and `manifold_mixup` are vicinal training protocols with soft
  labels, so they are best treated as secondary vicinal baselines rather than
  offline hard-label augmentation methods.
- `raw_smote_flatten_balanced` is a flattened feature-space rebalancing
  baseline; it is useful but not MTS-structure-aware.
