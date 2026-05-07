from __future__ import annotations

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult


def diffusionts_classwise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    max_epochs: int = 500,
    batch_size: int = 128,
    device: str = "cpu",
) -> ExternalAugResult:
    """Classwise Diffusion-TS generator with classifier guidance."""
    from utils.diffusionts_wrapper import fit_sample_diffusionts

    X_aug, y_aug = fit_sample_diffusionts(
        X_train_ct=X_train,
        y_train=y_train,
        multiplier=multiplier,
        seed=seed,
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=y_aug,
        source_space="generative_model",
        label_mode="hard",
        uses_external_library=True,
        library_name="Diffusion-TS",
        budget_matched=True,
        selection_rule="classwise_diffusionts_classifier_guidance",
        meta={
            "diffusionts_max_epochs": float(max_epochs),
            "target_aug_ratio": float(multiplier),
            "actual_aug_ratio": float(X_aug.shape[0]) / max(float(len(X_train)), 1.0),
        },
    )
