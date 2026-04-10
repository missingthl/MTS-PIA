from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ETT_HOURLY_TRAIN_ROWS = 12 * 30 * 24
ETT_HOURLY_VAL_ROWS = 4 * 30 * 24
ETT_HOURLY_TEST_ROWS = 4 * 30 * 24
ETT_HOURLY_USED_ROWS = ETT_HOURLY_TRAIN_ROWS + ETT_HOURLY_VAL_ROWS + ETT_HOURLY_TEST_ROWS
ETT_HOURLY_SPLIT_POLICY = "official_ett_hourly_12_4_4"


@dataclass(frozen=True)
class ETTRunSpec:
    dataset: str
    dataset_path: str
    feature_columns: List[str]
    target_column: str
    lookback: int
    horizon: int


@dataclass(frozen=True)
class ETTSampleSplit:
    name: str
    raw_row_start: int
    raw_row_end: int
    source_row_start: int
    source_row_end: int
    X: np.ndarray
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True)
class ETTPreparedData:
    spec: ETTRunSpec
    train: ETTSampleSplit
    val: ETTSampleSplit
    test: ETTSampleSplit
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    input_dim: int
    total_rows: int
    used_rows: int
    split_policy: str = ETT_HOURLY_SPLIT_POLICY
    input_norm_mode: str = "train_only_feature_standardization"
    target_norm_mode: str = "train_only_target_standardization"
    forecast_mode: str = "direct_multi_output"


def _validate_dataset_name(dataset: str) -> str:
    key = str(dataset or "").strip()
    if not key:
        raise ValueError("dataset must be non-empty")
    return key


def _load_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ETT csv not found: {csv_path}")
    df = pd.read_csv(path)
    required = {"date", "OT"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ETT csv missing required columns: {sorted(missing)}")
    return df


def _feature_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    cols = [c for c in df.columns if c != "date"]
    if target_column not in cols:
        raise ValueError(f"target column {target_column} not found in numeric columns")
    return cols


def _fit_scalers(
    train_frame: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
) -> Tuple[StandardScaler, StandardScaler]:
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    input_scaler.fit(train_frame[feature_columns].to_numpy(dtype=np.float64))
    target_scaler.fit(train_frame[[target_column]].to_numpy(dtype=np.float64))
    return input_scaler, target_scaler


def _build_direct_split(
    *,
    name: str,
    raw_values_scaled: np.ndarray,
    raw_target_scaled: np.ndarray,
    raw_row_start: int,
    raw_row_end: int,
    lookback: int,
    horizon: int,
) -> ETTSampleSplit:
    source_row_start = raw_row_start if name == "train" else raw_row_start - lookback
    if source_row_start < 0:
        raise ValueError(f"{name} split requires negative source start; lookback too large")
    source_row_end = raw_row_end

    source_X = raw_values_scaled[source_row_start:source_row_end]
    source_y = raw_target_scaled[source_row_start:source_row_end]

    n_total = int(source_X.shape[0])
    n_samples = n_total - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"{name} split has no valid samples for lookback={lookback}, horizon={horizon}"
        )

    X = np.empty((n_samples, lookback, source_X.shape[1]), dtype=np.float32)
    y = np.empty((n_samples, horizon), dtype=np.float32)
    for i in range(n_samples):
        start = i
        cut = i + lookback
        stop = cut + horizon
        X[i] = source_X[start:cut]
        y[i] = source_y[cut:stop, 0]

    return ETTSampleSplit(
        name=name,
        raw_row_start=raw_row_start,
        raw_row_end=raw_row_end,
        source_row_start=source_row_start,
        source_row_end=source_row_end,
        X=X,
        y=y,
    )


def prepare_ett_direct_forecast(
    *,
    dataset: str,
    csv_path: str,
    target_column: str = "OT",
    lookback: int,
    horizon: int,
) -> ETTPreparedData:
    ds = _validate_dataset_name(dataset)
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")

    df = _load_dataframe(csv_path)
    if len(df) < ETT_HOURLY_USED_ROWS:
        raise ValueError(
            f"{ds} has only {len(df)} rows, fewer than required official ETT hourly rows {ETT_HOURLY_USED_ROWS}"
        )

    feature_columns = _feature_columns(df, target_column)
    used_df = df.iloc[:ETT_HOURLY_USED_ROWS].copy()

    train_end = ETT_HOURLY_TRAIN_ROWS
    val_end = ETT_HOURLY_TRAIN_ROWS + ETT_HOURLY_VAL_ROWS
    test_end = ETT_HOURLY_USED_ROWS

    train_frame = used_df.iloc[:train_end]
    input_scaler, target_scaler = _fit_scalers(train_frame, feature_columns, target_column)

    values_scaled = input_scaler.transform(used_df[feature_columns].to_numpy(dtype=np.float64))
    target_scaled = target_scaler.transform(used_df[[target_column]].to_numpy(dtype=np.float64))

    train_split = _build_direct_split(
        name="train",
        raw_values_scaled=values_scaled,
        raw_target_scaled=target_scaled,
        raw_row_start=0,
        raw_row_end=train_end,
        lookback=lookback,
        horizon=horizon,
    )
    val_split = _build_direct_split(
        name="val",
        raw_values_scaled=values_scaled,
        raw_target_scaled=target_scaled,
        raw_row_start=train_end,
        raw_row_end=val_end,
        lookback=lookback,
        horizon=horizon,
    )
    test_split = _build_direct_split(
        name="test",
        raw_values_scaled=values_scaled,
        raw_target_scaled=target_scaled,
        raw_row_start=val_end,
        raw_row_end=test_end,
        lookback=lookback,
        horizon=horizon,
    )

    return ETTPreparedData(
        spec=ETTRunSpec(
            dataset=ds,
            dataset_path=str(Path(csv_path).resolve()),
            feature_columns=feature_columns,
            target_column=target_column,
            lookback=int(lookback),
            horizon=int(horizon),
        ),
        train=train_split,
        val=val_split,
        test=test_split,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        input_dim=len(feature_columns),
        total_rows=len(df),
        used_rows=len(used_df),
    )


def inverse_transform_target(target_scaler: StandardScaler, y_scaled: np.ndarray) -> np.ndarray:
    y2 = np.asarray(y_scaled, dtype=np.float64)
    flat = y2.reshape(-1, 1)
    restored = target_scaler.inverse_transform(flat)
    return restored.reshape(y2.shape)


def summarize_split_rows(prepared: ETTPreparedData) -> Dict[str, int]:
    return {
        "total_rows_csv": int(prepared.total_rows),
        "used_rows_official": int(prepared.used_rows),
        "train_rows": int(prepared.train.raw_row_end - prepared.train.raw_row_start),
        "val_rows": int(prepared.val.raw_row_end - prepared.val.raw_row_start),
        "test_rows": int(prepared.test.raw_row_end - prepared.test.raw_row_start),
    }
