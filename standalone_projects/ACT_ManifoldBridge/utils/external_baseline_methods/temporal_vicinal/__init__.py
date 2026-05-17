"""Temporal and vicinal heuristic augmentation baselines.

This package is a paper-facing grouping facade.  The concrete implementations
remain in the parent method-specific modules for backward compatibility.
"""

from __future__ import annotations

from utils.external_baseline_methods.raw_jitter import raw_aug_jitter
from utils.external_baseline_methods.raw_magnitude_warping import raw_aug_magnitude_warping
from utils.external_baseline_methods.raw_mixup import raw_mixup
from utils.external_baseline_methods.raw_scaling import raw_aug_scaling
from utils.external_baseline_methods.raw_timewarp import raw_aug_timewarp
from utils.external_baseline_methods.raw_window_slicing import raw_aug_window_slicing
from utils.external_baseline_methods.raw_window_warping import raw_aug_window_warping
from utils.external_baseline_methods.smote import raw_smote_flatten_balanced

__all__ = [
    "raw_aug_jitter",
    "raw_aug_magnitude_warping",
    "raw_aug_scaling",
    "raw_aug_timewarp",
    "raw_aug_window_slicing",
    "raw_aug_window_warping",
    "raw_mixup",
    "raw_smote_flatten_balanced",
]
