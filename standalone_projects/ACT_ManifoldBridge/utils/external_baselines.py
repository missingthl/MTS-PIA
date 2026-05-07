"""Compatibility facade for external augmentation baselines.

Concrete implementations now live in ``utils.external_baseline_methods`` as
small method-specific modules.  Keep this facade so historical imports such as
``from utils.external_baselines import wdba_sameclass`` remain valid.
"""

from __future__ import annotations

from utils.external_baseline_methods import (
    ExternalAugResult,
    dba_sameclass,
    dgw_sameclass,
    diffusionts_classwise,
    jobda_cleanroom_augmented_set,
    pca_cov_state,
    random_cov_state,
    raw_aug_jitter,
    raw_aug_magnitude_warping,
    raw_aug_scaling,
    raw_aug_timewarp,
    raw_aug_window_slicing,
    raw_aug_window_warping,
    raw_mixup,
    raw_smote_flatten_balanced,
    rgw_sameclass,
    spawner_sameclass_style,
    time_series_warping_cleanroom,
    timevae_classwise_optional,
    timevqvae_classwise,
    wdba_sameclass,
)

__all__ = [
    "ExternalAugResult",
    "dba_sameclass",
    "dgw_sameclass",
    "diffusionts_classwise",
    "jobda_cleanroom_augmented_set",
    "pca_cov_state",
    "random_cov_state",
    "raw_aug_jitter",
    "raw_aug_magnitude_warping",
    "raw_aug_scaling",
    "raw_aug_timewarp",
    "raw_aug_window_slicing",
    "raw_aug_window_warping",
    "raw_mixup",
    "raw_smote_flatten_balanced",
    "rgw_sameclass",
    "spawner_sameclass_style",
    "time_series_warping_cleanroom",
    "timevae_classwise_optional",
    "timevqvae_classwise",
    "wdba_sameclass",
]
