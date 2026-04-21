from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS
from sklearn.preprocessing import StandardScaler

from datasets.forecast_ett import (
    ETT_HOURLY_SPLIT_POLICY,
    ETT_HOURLY_TEST_ROWS,
    ETT_HOURLY_TRAIN_ROWS,
    ETT_HOURLY_USED_ROWS,
    ETT_HOURLY_VAL_ROWS,
)


@dataclass(frozen=True)
class ZSeqRunSpec:
    dataset: str
    dataset_path: str
    feature_columns: List[str]
    lookback: int
    horizon: int
    cov_estimator: str
    spd_eps: float


@dataclass(frozen=True)
class ZPairSplit:
    name: str
    raw_row_start: int
    raw_row_end: int
    current_end_start: int
    current_end_end: int
    X: np.ndarray
    Y: np.ndarray
    current_end_indices: np.ndarray
    target_end_indices: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def delta(self) -> np.ndarray:
        return np.asarray(self.Y - self.X, dtype=np.float32)


@dataclass(frozen=True)
class ZSeqPreparedData:
    spec: ZSeqRunSpec
    train: ZPairSplit
    val: ZPairSplit
    test: ZPairSplit
    input_scaler: StandardScaler
    mean_log_train: np.ndarray
    z_dim: int
    total_rows: int
    used_rows: int
    split_policy: str = ETT_HOURLY_SPLIT_POLICY
    representation_space: str = "z"
    raw_norm_mode: str = "train_only_feature_standardization"
    tangent_center_mode: str = "logcenter_train_only"


def _load_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ETT csv not found: {csv_path}")
    df = pd.read_csv(path)
    required = {"date", "OT"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ETT csv missing required columns: {sorted(missing)}")
    if len(df) < ETT_HOURLY_USED_ROWS:
        raise ValueError(
            f"ETT file has {len(df)} rows, fewer than required official rows {ETT_HOURLY_USED_ROWS}"
        )
    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "date"]


def _fit_input_scaler(train_frame: pd.DataFrame, feature_columns: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_frame[feature_columns].to_numpy(dtype=np.float64))
    return scaler


def _regularize_spd(cov: np.ndarray, eps: float) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(eps) * np.eye(cov.shape[0], dtype=np.float64)
    return cov


def _window_cov(window: np.ndarray, estimator: str, spd_eps: float) -> np.ndarray:
    est = str(estimator).lower()
    if est == "sample":
        x = np.asarray(window, dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        denom = max(1, x.shape[0] - 1)
        cov = (x.T @ x) / float(denom)
    elif est == "oas":
        cov = OAS().fit(np.asarray(window, dtype=np.float64)).covariance_
    elif est in {"lw", "ledoitwolf"}:
        cov = LedoitWolf().fit(np.asarray(window, dtype=np.float64)).covariance_
    else:
        raise ValueError(f"Unsupported cov_estimator: {estimator}")
    return _regularize_spd(cov, spd_eps)


def _logmap_spd(cov: np.ndarray, eps: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, float(eps))
    log_vals = np.log(vals)
    return (vecs * log_vals) @ vecs.T


def _vec_utri(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0])
    return np.asarray(mat[idx], dtype=np.float32)


def _compute_all_covs(values_scaled: np.ndarray, lookback: int, estimator: str, spd_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    covs: List[np.ndarray] = []
    end_indices: List[int] = []
    for end_idx in range(lookback - 1, values_scaled.shape[0]):
        start_idx = end_idx - lookback + 1
        window = values_scaled[start_idx:end_idx + 1]
        covs.append(_window_cov(window, estimator=estimator, spd_eps=spd_eps))
        end_indices.append(end_idx)
    return np.asarray(covs, dtype=np.float32), np.asarray(end_indices, dtype=np.int64)


def _build_pair_split(
    *,
    name: str,
    z_by_end: Dict[int, np.ndarray],
    current_end_start: int,
    current_end_end: int,
    horizon: int,
    raw_row_start: int,
    raw_row_end: int,
) -> ZPairSplit:
    current_indices = np.arange(current_end_start, current_end_end, dtype=np.int64)
    target_indices = current_indices + int(horizon)
    X = np.asarray([z_by_end[int(idx)] for idx in current_indices], dtype=np.float32)
    Y = np.asarray([z_by_end[int(idx)] for idx in target_indices], dtype=np.float32)
    return ZPairSplit(
        name=name,
        raw_row_start=int(raw_row_start),
        raw_row_end=int(raw_row_end),
        current_end_start=int(current_end_start),
        current_end_end=int(current_end_end),
        X=X,
        Y=Y,
        current_end_indices=current_indices,
        target_end_indices=target_indices,
    )


def prepare_ett_zseq_probe(
    *,
    dataset: str,
    csv_path: str,
    lookback: int,
    horizon: int = 1,
    cov_estimator: str = "oas",
    spd_eps: float = 1e-4,
) -> ZSeqPreparedData:
    if lookback <= 1:
        raise ValueError("lookback must be > 1 for z-sequence probe")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    df = _load_dataframe(csv_path)
    feature_columns = _feature_columns(df)
    used_df = df.iloc[:ETT_HOURLY_USED_ROWS].copy()

    train_end = ETT_HOURLY_TRAIN_ROWS
    val_end = ETT_HOURLY_TRAIN_ROWS + ETT_HOURLY_VAL_ROWS
    test_end = ETT_HOURLY_TRAIN_ROWS + ETT_HOURLY_VAL_ROWS + ETT_HOURLY_TEST_ROWS

    train_frame = used_df.iloc[:train_end]
    input_scaler = _fit_input_scaler(train_frame, feature_columns)
    values_scaled = input_scaler.transform(used_df[feature_columns].to_numpy(dtype=np.float64))

    covs_all, end_indices = _compute_all_covs(values_scaled, lookback, cov_estimator, spd_eps)
    train_mask = end_indices < train_end
    train_log = np.asarray([_logmap_spd(c, spd_eps) for c in covs_all[train_mask]], dtype=np.float32)
    mean_log_train = train_log.mean(axis=0)
    z_all = np.asarray([_vec_utri(_logmap_spd(c, spd_eps) - mean_log_train) for c in covs_all], dtype=np.float32)
    z_by_end = {int(end_idx): z for end_idx, z in zip(end_indices.tolist(), z_all)}

    train_split = _build_pair_split(
        name="train",
        z_by_end=z_by_end,
        current_end_start=lookback - 1,
        current_end_end=train_end - horizon,
        horizon=horizon,
        raw_row_start=0,
        raw_row_end=train_end,
    )
    val_split = _build_pair_split(
        name="val",
        z_by_end=z_by_end,
        current_end_start=train_end - 1,
        current_end_end=val_end - horizon,
        horizon=horizon,
        raw_row_start=train_end,
        raw_row_end=val_end,
    )
    test_split = _build_pair_split(
        name="test",
        z_by_end=z_by_end,
        current_end_start=val_end - 1,
        current_end_end=test_end - horizon,
        horizon=horizon,
        raw_row_start=val_end,
        raw_row_end=test_end,
    )

    return ZSeqPreparedData(
        spec=ZSeqRunSpec(
            dataset=str(dataset),
            dataset_path=str(Path(csv_path).resolve()),
            feature_columns=feature_columns,
            lookback=int(lookback),
            horizon=int(horizon),
            cov_estimator=str(cov_estimator),
            spd_eps=float(spd_eps),
        ),
        train=train_split,
        val=val_split,
        test=test_split,
        input_scaler=input_scaler,
        mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
        z_dim=int(z_all.shape[1]),
        total_rows=int(len(df)),
        used_rows=int(len(used_df)),
    )


def summarize_zseq_rows(prepared: ZSeqPreparedData) -> Dict[str, int]:
    return {
        "total_rows_csv": int(prepared.total_rows),
        "used_rows_official": int(prepared.used_rows),
        "train_rows": int(prepared.train.raw_row_end - prepared.train.raw_row_start),
        "val_rows": int(prepared.val.raw_row_end - prepared.val.raw_row_start),
        "test_rows": int(prepared.test.raw_row_end - prepared.test.raw_row_start),
    }
