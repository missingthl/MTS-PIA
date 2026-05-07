from __future__ import annotations

from utils.external_baseline_methods.base import ExternalAugResult
from utils.external_baseline_methods.dba import dba_sameclass
from utils.external_baseline_methods.dgw import dgw_sameclass
from utils.external_baseline_methods.diffusionts import diffusionts_classwise
from utils.external_baseline_methods.jobda import jobda_cleanroom_augmented_set, time_series_warping_cleanroom
from utils.external_baseline_methods.pca_cov_state import pca_cov_state
from utils.external_baseline_methods.random_cov_state import random_cov_state
from utils.external_baseline_methods.raw_jitter import raw_aug_jitter
from utils.external_baseline_methods.raw_magnitude_warping import raw_aug_magnitude_warping
from utils.external_baseline_methods.raw_mixup import raw_mixup
from utils.external_baseline_methods.raw_scaling import raw_aug_scaling
from utils.external_baseline_methods.raw_timewarp import raw_aug_timewarp
from utils.external_baseline_methods.raw_window_slicing import raw_aug_window_slicing
from utils.external_baseline_methods.raw_window_warping import raw_aug_window_warping
from utils.external_baseline_methods.rgw import rgw_sameclass
from utils.external_baseline_methods.smote import raw_smote_flatten_balanced
from utils.external_baseline_methods.spawner import spawner_sameclass_style
from utils.external_baseline_methods.timevae import timevae_classwise_optional
from utils.external_baseline_methods.timevqvae import timevqvae_classwise
from utils.external_baseline_methods.wdba import wdba_sameclass

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
