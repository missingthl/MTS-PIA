from __future__ import annotations

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult


def timevqvae_classwise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    vqvae_epochs: int = 100,
    maskgit_epochs: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
) -> ExternalAugResult:
    """Classwise TimeVQVAE generator."""
    from utils.timevqvae_wrapper import fit_sample_timevqvae

    X_aug, y_aug = fit_sample_timevqvae(
        X_train_ct=X_train,
        y_train=y_train,
        multiplier=multiplier,
        seed=seed,
        device=device,
        vqvae_epochs=vqvae_epochs,
        maskgit_epochs=maskgit_epochs,
        batch_size=batch_size,
    )

    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=y_aug,
        source_space="generative_model",
        label_mode="hard",
        uses_external_library=True,
        library_name="TimeVQVAE",
        budget_matched=True,
        selection_rule="classwise_timevqvae_maskgit",
        meta={
            "timevqvae_vqvae_epochs": float(vqvae_epochs),
            "timevqvae_maskgit_epochs": float(maskgit_epochs),
            "target_aug_ratio": float(multiplier),
            "actual_aug_ratio": float(X_aug.shape[0]) / max(float(len(X_train)), 1.0),
        },
    )
