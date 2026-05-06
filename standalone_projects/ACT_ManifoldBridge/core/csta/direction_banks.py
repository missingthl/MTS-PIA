from __future__ import annotations

from typing import Optional

import numpy as np

from core.pia import (
    FisherPIAConfig,
    LRAESConfig,
    build_ao_pia_direction_bank,
    build_lraes_direction_bank,
    build_pia_direction_bank,
    build_pca_direction_bank,
    build_random_orthogonal_direction_bank,
    build_zpia_direction_bank,
)


def build_direction_bank_for_args(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    algo_override: Optional[str] = None,
) -> dict:
    algo_name = str(algo_override or args.algo)
    if algo_name == "lraes":
        direction_bank, direction_meta = build_lraes_direction_bank(
            X_train_z,
            y_train,
            k_dir=args.k_dir,
            fisher_cfg=FisherPIAConfig(),
            lraes_cfg=LRAESConfig(),
        )
    elif algo_name == "zpia":
        direction_bank, direction_meta = build_zpia_direction_bank(
            X_train_z,
            k_dir=args.k_dir,
            seed=seed,
            telm2_n_iters=args.telm2_n_iters,
            telm2_c_repr=args.telm2_c_repr,
            telm2_activation=args.telm2_activation,
            telm2_bias_update_mode=args.telm2_bias_update_mode,
        )
    elif algo_name == "pca":
        direction_bank, direction_meta = build_pca_direction_bank(
            X_train_z,
            k_dir=args.k_dir,
            seed=seed,
        )
    elif algo_name == "random_orth":
        direction_bank, direction_meta = build_random_orthogonal_direction_bank(
            X_train_z,
            k_dir=args.k_dir,
            seed=seed,
        )
    elif algo_name == "ao_fisher":
        direction_bank, direction_meta = build_ao_pia_direction_bank(
            X_train_z, y_train,
            k_dir=args.k_dir, rho_scale=getattr(args, "ao_rho_scale", 1e-3),
            mode="ao_fisher", seed=seed,
        )
    elif algo_name == "ao_contrastive":
        direction_bank, direction_meta = build_ao_pia_direction_bank(
            X_train_z, y_train,
            k_dir=args.k_dir, rho_scale=getattr(args, "ao_rho_scale", 1e-3),
            mode="ao_contrastive",
            lambda_pos=getattr(args, "ao_lambda_pos", 0.5),
            lambda_neg=getattr(args, "ao_lambda_neg", 0.5),
            k_pos=getattr(args, "ao_k_pos", 5),
            k_neg=getattr(args, "ao_k_neg", 5),
            seed=seed,
        )
    else:
        direction_bank, direction_meta = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)
    return {"bank": direction_bank, "meta": direction_meta}
