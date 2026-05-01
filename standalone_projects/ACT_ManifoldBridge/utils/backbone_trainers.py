from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from utils.evaluators import (
    build_model,
    fit_eval_minirocket,
    fit_eval_mptsnet,
    fit_eval_patchtst,
    fit_eval_resnet1d,
    fit_eval_resnet1d_jobda_joint_labels,
    fit_eval_resnet1d_manifold_mixup,
    fit_eval_resnet1d_soft_labels,
    fit_eval_timesnet,
)


SUPPORTED_BACKBONES = ("resnet1d", "minirocket", "patchtst", "timesnet", "mptsnet")
SOFT_LABEL_BACKBONES = ("resnet1d",)
MANIFOLD_MIXUP_BACKBONES = ("resnet1d",)
JOBDA_BACKBONES = ("resnet1d",)


def fit_hard_backbone(
    backbone: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    n_kernels: int = 10000,
) -> Dict[str, float]:
    """Fit/evaluate a hard-label baseline with a named backbone."""
    if backbone == "resnet1d":
        return fit_eval_resnet1d(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
            loader_seed=int(seed),
        )
    if backbone == "minirocket":
        model = build_model(n_kernels=int(n_kernels), random_state=int(seed))
        return fit_eval_minirocket(model, X_train, y_train, X_test, y_test)
    if backbone == "patchtst":
        return fit_eval_patchtst(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
            loader_seed=int(seed),
        )
    if backbone == "timesnet":
        return fit_eval_timesnet(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
            loader_seed=int(seed),
        )
    if backbone == "mptsnet":
        return fit_eval_mptsnet(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
            loader_seed=int(seed),
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def fit_soft_backbone(
    backbone: str,
    X_train: np.ndarray,
    y_train_soft: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
) -> Dict[str, float]:
    """Fit/evaluate a soft-label baseline.

    Phase 1 keeps raw_mixup soft-label training on ResNet1D only. Other
    backbones fail fast rather than silently routing through an invalid model.
    """
    if backbone != "resnet1d":
        raise NotImplementedError(
            f"Soft-label training is currently supported only for resnet1d; got backbone={backbone}."
        )
    return fit_eval_resnet1d_soft_labels(
        X_train,
        y_train_soft,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        loader_seed=int(seed),
    )


def fit_manifold_mixup_backbone(
    backbone: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    mixup_alpha: float,
) -> Dict[str, float]:
    if backbone != "resnet1d":
        raise NotImplementedError(
            f"manifold_mixup is currently supported only for resnet1d; got backbone={backbone}."
        )
    return fit_eval_resnet1d_manifold_mixup(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        mixup_alpha=mixup_alpha,
        loader_seed=int(seed),
    )


def fit_jobda_backbone(
    backbone: str,
    X_train: np.ndarray,
    y_train_joint: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_classes: int,
    num_transforms: int,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
) -> Dict[str, float]:
    """Fit/evaluate JobDA joint labels.

    Clean-room JobDA v1 follows the paper's ResNet joint-label setup. Other
    backbones fail fast until a matching joint-label inference wrapper exists.
    """
    if backbone != "resnet1d":
        raise NotImplementedError(
            f"JobDA joint-label training is currently supported only for resnet1d; got backbone={backbone}."
        )
    return fit_eval_resnet1d_jobda_joint_labels(
        X_train,
        y_train_joint,
        X_val,
        y_val,
        X_test,
        y_test,
        num_classes=int(num_classes),
        num_transforms=int(num_transforms),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        loader_seed=int(seed),
    )
