from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from datasets.trial_dataset_factory import DEFAULT_IEEEPPG_ROOT, load_trials_for_dataset
from route_b_unified.spd_features import logm_spd, vec_utri


@dataclass(frozen=True)
class RegressionRepresentationConfig:
    dataset: str = "ieeeppg"
    spd_eps: float = 1e-4
    ieeeppg_root: str = DEFAULT_IEEEPPG_ROOT


@dataclass
class RegressionRepresentationState:
    dataset: str
    split_meta: Dict[str, object]
    mean_log_train: np.ndarray
    train_trial_dicts: List[Dict[str, object]]
    test_trial_dicts: List[Dict[str, object]]
    X_train_z: np.ndarray
    y_train: np.ndarray
    tid_train: np.ndarray
    X_test_z: np.ndarray
    y_test: np.ndarray
    tid_test: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


def _covariance_from_trial(x: np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    x_c = arr - arr.mean(axis=1, keepdims=True)
    denom = max(1, int(arr.shape[1]) - 1)
    cov = (x_c @ x_c.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(eps) * np.eye(cov.shape[0], dtype=np.float64)
    return cov.astype(np.float32)


def _build_log_cov_batch(trials: List[Dict[str, object]], spd_eps: float) -> tuple[list[np.ndarray], np.ndarray]:
    log_covs = [
        logm_spd(_covariance_from_trial(np.asarray(t["x_trial"], dtype=np.float32), spd_eps), spd_eps).astype(np.float32)
        for t in trials
    ]
    mean_log = np.mean(np.stack(log_covs, axis=0), axis=0).astype(np.float32)
    return log_covs, mean_log


def _apply_mean_log(log_covs: List[np.ndarray], mean_log: np.ndarray) -> np.ndarray:
    return np.stack(
        [vec_utri(np.asarray(log_cov, dtype=np.float64) - np.asarray(mean_log, dtype=np.float64)).astype(np.float32) for log_cov in log_covs],
        axis=0,
    )


def build_regression_representation(cfg: RegressionRepresentationConfig) -> RegressionRepresentationState:
    dataset = str(cfg.dataset).strip().lower()
    if dataset != "ieeeppg":
        raise ValueError(f"Unsupported regression dataset for phase-1 baseline: {cfg.dataset}")

    all_trials = load_trials_for_dataset(dataset=dataset, ieeeppg_root=cfg.ieeeppg_root)
    train_trials = [t for t in all_trials if str(t.get("split", "")).lower() == "train"]
    test_trials = [t for t in all_trials if str(t.get("split", "")).lower() == "test"]
    if not train_trials or not test_trials:
        raise RuntimeError("IEEEPPG adapter must provide official split=train/test trials.")

    train_log_covs, mean_log_train = _build_log_cov_batch(train_trials, float(cfg.spd_eps))
    test_log_covs = [
        logm_spd(_covariance_from_trial(np.asarray(t["x_trial"], dtype=np.float32), float(cfg.spd_eps)), float(cfg.spd_eps)).astype(np.float32)
        for t in test_trials
    ]

    X_train_z = _apply_mean_log(train_log_covs, mean_log_train)
    X_test_z = _apply_mean_log(test_log_covs, mean_log_train)
    y_train = np.asarray([float(t["y_value"]) for t in train_trials], dtype=np.float64)
    y_test = np.asarray([float(t["y_value"]) for t in test_trials], dtype=np.float64)
    tid_train = np.asarray([str(t["trial_id_str"]) for t in train_trials], dtype=object)
    tid_test = np.asarray([str(t["trial_id_str"]) for t in test_trials], dtype=object)

    channels = int(np.asarray(train_trials[0]["x_trial"]).shape[0])
    length = int(np.asarray(train_trials[0]["x_trial"]).shape[1])

    return RegressionRepresentationState(
        dataset=dataset,
        split_meta={
            "protocol_type": "aeon_regression_official",
            "protocol_note": "aeon load_regression official train/test split",
        },
        mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
        train_trial_dicts=list(train_trials),
        test_trial_dicts=list(test_trials),
        X_train_z=np.asarray(X_train_z, dtype=np.float32),
        y_train=y_train,
        tid_train=tid_train,
        X_test_z=np.asarray(X_test_z, dtype=np.float32),
        y_test=y_test,
        tid_test=tid_test,
        meta={
            "n_train": int(len(train_trials)),
            "n_test": int(len(test_trials)),
            "channels": channels,
            "length": length,
            "z_dim": int(X_train_z.shape[1]),
            "spd_eps": float(cfg.spd_eps),
            "geometry": "log_euclidean_covariance_upper_triangle",
            "y_train_mean": float(np.mean(y_train)),
            "y_train_std": float(np.std(y_train)),
            "y_train_min": float(np.min(y_train)),
            "y_train_max": float(np.max(y_train)),
            "y_test_mean": float(np.mean(y_test)),
            "y_test_std": float(np.std(y_test)),
            "y_test_min": float(np.min(y_test)),
            "y_test_max": float(np.max(y_test)),
        },
    )
