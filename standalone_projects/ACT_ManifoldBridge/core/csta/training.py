from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from utils.evaluators import (
    ManifoldAugDataset,
    TauScheduler,
    build_model,
    fit_eval_minirocket,
    fit_eval_moderntcn,
    fit_eval_mptsnet,
    fit_eval_patchtst,
    fit_eval_patchtst_weighted_aug_ce,
    fit_eval_resnet1d,
    fit_eval_resnet1d_weighted_aug_ce,
    fit_eval_timesnet,
    fit_eval_timesnet_weighted_aug_ce,
)


def fit_host_model(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    return_model_obj: bool = False,
    loader_seed: Optional[int] = None,
) -> Dict[str, object]:
    kwargs = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "device": args.device,
        "return_model_obj": return_model_obj,
    }
    if args.model == "resnet1d":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_resnet1d(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "patchtst":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_patchtst(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "timesnet":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_timesnet(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "mptsnet":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_mptsnet(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "moderntcn":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_moderntcn(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)

    model = build_model(n_kernels=args.n_kernels, random_state=loader_seed or 42)
    return fit_eval_minirocket(model, X_tr, y_tr, X_test_raw, y_test)


def fit_host_model_weighted_aug_ce(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_aug: Optional[np.ndarray],
    y_aug: Optional[np.ndarray],
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    loader_seed: Optional[int] = None,
    aug_dataset: Optional[ManifoldAugDataset] = None,
    tau_scheduler: Optional[TauScheduler] = None,
) -> Dict[str, object]:
    kwargs = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "device": args.device,
        "feedback_margin_temperature": args.feedback_margin_temperature,
        "aug_loss_weight": args.aug_loss_weight,
        "loader_seed": loader_seed,
        "aug_dataset": aug_dataset,
        "weight_mode": args.aug_weight_mode,
        "tau_scheduler": tau_scheduler,
        "steps_per_epoch": args.steps_per_epoch,
    }
    if args.model == "resnet1d":
        return fit_eval_resnet1d_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    if args.model == "patchtst":
        return fit_eval_patchtst_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    if args.model == "timesnet":
        return fit_eval_timesnet_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    raise ValueError("Weighted aug-CE training supports resnet1d, patchtst, and timesnet only.")

