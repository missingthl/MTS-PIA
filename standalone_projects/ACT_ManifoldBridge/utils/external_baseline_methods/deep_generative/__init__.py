"""Deep generative augmentation baselines.

This package is a paper-facing grouping facade.  Vendored third-party code
lives under ``external/`` when needed; these imports point to project adapters.
"""

from __future__ import annotations

from utils.external_baseline_methods.diffusionts import diffusionts_classwise
from utils.external_baseline_methods.timevae import timevae_classwise_optional
from utils.external_baseline_methods.timegan import timegan_classwise
from utils.external_baseline_methods.timevqvae import timevqvae_classwise

__all__ = [
    "diffusionts_classwise",
    "timevae_classwise_optional",
    "timegan_classwise",
    "timevqvae_classwise",
]
