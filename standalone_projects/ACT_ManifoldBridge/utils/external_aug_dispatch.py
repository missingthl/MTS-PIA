from __future__ import annotations

from typing import Callable, Dict

from utils.external_baselines import (
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
    timevae_classwise_optional,
    timevqvae_classwise,
    wdba_sameclass,
)
from utils.external_runner_registry import parse_csv


def build_external_aug(method: str, X_train, y_train, args, seed: int, n_classes: int) -> ExternalAugResult:
    """Build one external augmentation result for the matrix runner.

    This module owns method-to-augmenter dispatch so the public runner can stay
    focused on experiment orchestration, split handling, and result rows.
    """
    builders: Dict[str, Callable[[], ExternalAugResult]] = {
        "raw_aug_jitter": lambda: raw_aug_jitter(X_train, y_train, multiplier=args.multiplier, seed=seed),
        "raw_aug_scaling": lambda: raw_aug_scaling(X_train, y_train, multiplier=args.multiplier, seed=seed),
        "raw_aug_timewarp": lambda: raw_aug_timewarp(X_train, y_train, multiplier=args.multiplier, seed=seed),
        "raw_aug_magnitude_warping": lambda: raw_aug_magnitude_warping(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            sigma=args.magnitude_warp_sigma,
            knots=args.magnitude_warp_knots,
            per_channel_curve=not args.magnitude_warp_shared_curve,
        ),
        "raw_aug_window_warping": lambda: raw_aug_window_warping(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            window_ratio=args.window_warp_ratio,
            speed_factors=tuple(float(x) for x in parse_csv(args.window_warp_speeds)),
            min_window_len=args.window_min_len,
        ),
        "raw_aug_window_slicing": lambda: raw_aug_window_slicing(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            slice_ratio=args.window_slice_ratio,
            min_window_len=args.window_min_len,
        ),
        "raw_mixup": lambda: raw_mixup(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            alpha=args.mixup_alpha,
            n_classes=n_classes,
        ),
        "dba_sameclass": lambda: dba_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            k=args.dba_k,
            max_iter=args.dba_max_iter,
        ),
        "wdba_sameclass": lambda: wdba_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            k=args.wdba_k,
            max_iter=args.wdba_max_iter,
        ),
        "spawner_sameclass_style": lambda: spawner_sameclass_style(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            noise_scale=args.spawner_noise_scale,
        ),
        "jobda_cleanroom": lambda: jobda_cleanroom_augmented_set(
            X_train,
            y_train,
            transform_subseqs=tuple(int(x) for x in parse_csv(args.jobda_transform_subseqs)),
        ),
        "rgw_sameclass": lambda: rgw_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            slope_constraint=args.guided_warp_slope_constraint,
            use_window=not args.guided_warp_no_window,
        ),
        "dgw_sameclass": lambda: dgw_sameclass(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            batch_size=args.guided_warp_batch_size,
            slope_constraint=args.guided_warp_slope_constraint,
            use_window=not args.guided_warp_no_window,
            use_variable_slice=not args.dgw_no_variable_slice,
            min_window_len=args.window_min_len,
        ),
        "timevae_classwise_optional": lambda: timevae_classwise_optional(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            epochs=args.timevae_epochs,
            batch_size=args.timevae_batch_size,
            lr=args.timevae_lr,
            latent_dim=args.timevae_latent_dim,
            hidden_dim=args.timevae_hidden_dim,
            beta=args.timevae_beta,
            min_class_size=args.timevae_min_class_size,
            device=args.device,
        ),
        "diffusionts_classwise": lambda: diffusionts_classwise(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            max_epochs=args.diffusionts_epochs,
            batch_size=args.diffusionts_batch_size,
            device=args.device,
        ),
        "timevqvae_classwise": lambda: timevqvae_classwise(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            vqvae_epochs=args.timevqvae_vqvae_epochs,
            maskgit_epochs=args.timevqvae_maskgit_epochs,
            batch_size=args.timevqvae_batch_size,
            device=args.device,
        ),
        "raw_smote_flatten_balanced": lambda: raw_smote_flatten_balanced(X_train, y_train, seed=seed),
        "random_cov_state": lambda: random_cov_state(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            gamma=args.pia_gamma,
        ),
        "pca_cov_state": lambda: pca_cov_state(
            X_train,
            y_train,
            multiplier=args.multiplier,
            seed=seed,
            gamma=args.pia_gamma,
            k_dir=args.k_dir,
        ),
    }
    if method not in builders:
        raise ValueError(f"No external augmenter registered for method={method}")
    return builders[method]()
