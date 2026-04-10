from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.forecast_ett import (  # noqa: E402
    ETTPreparedData,
    inverse_transform_target,
    prepare_ett_direct_forecast,
    summarize_split_rows,
)


DEFAULT_ETTH1_PATH = PROJECT_ROOT / "data" / "MTSFC" / "EET" / "ETTh1.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "out" / "forecast_probe_ett"
TORCH_MODELS = {"dlinear", "nlinear"}
SUMMARY_DEFAULTS = {
    "run_name": "run0",
    "variant_label": "",
    "pia_plugin_enabled": False,
    "pia_plugin_mode": "none",
    "pia_epsilon": 0.0,
    "pia_aug_ratio": 0.0,
    "anchor_protection_enabled": False,
    "anchor_protection_scope": "none",
}


@dataclass
class ForecastResult:
    dataset: str
    model_type: str
    lookback: int
    horizon: int
    mae: float
    rmse: float
    val_mae: float
    val_rmse: float
    run_dir: str
    run_name: str
    variant_label: str
    pia_plugin_enabled: bool
    pia_plugin_mode: str
    pia_epsilon: float
    pia_aug_ratio: float
    anchor_protection_enabled: bool
    anchor_protection_scope: str


@dataclass
class TorchFitArtifacts:
    y_val_pred_scaled: np.ndarray
    y_test_pred_scaled: np.ndarray
    model_params: Dict[str, object]
    pia_n_aug_samples: int
    pia_n_aug_samples_total_seen: int
    pia_runtime_overhead_sec: float
    anchor_protection_enabled: bool
    anchor_protection_scope: str


class DirectMultiOutputELMRegressor:
    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        activation: str = "tanh",
        alpha: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.activation = str(activation)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DirectMultiOutputELMRegressor":
        rng = np.random.RandomState(self.random_state)
        n_features = int(X.shape[1])
        self.W = rng.normal(0.0, 1.0, size=(n_features, self.hidden_dim)).astype(np.float64)
        self.b = rng.normal(0.0, 1.0, size=(self.hidden_dim,)).astype(np.float64)
        H = self._act(X @ self.W + self.b)
        gram = H.T @ H
        gram.flat[:: self.hidden_dim + 1] += self.alpha
        self.beta = np.linalg.solve(gram, H.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None or self.beta is None:
            raise RuntimeError("ELM model must be fit before predict")
        H = self._act(X @ self.W + self.b)
        return H @ self.beta


def _flatten_X(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)


def _inverse_target(prepared: ETTPreparedData, arr: np.ndarray) -> np.ndarray:
    return inverse_transform_target(prepared.target_scaler, arr)


def _persistence_predict(X_scaled: np.ndarray, horizon: int, target_idx: int) -> np.ndarray:
    last_target = X_scaled[:, -1, target_idx][:, None]
    return np.repeat(last_target, horizon, axis=1)


def _evaluate(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true_raw.reshape(-1), y_pred_raw.reshape(-1)))
    rmse = float(math.sqrt(mean_squared_error(y_true_raw.reshape(-1), y_pred_raw.reshape(-1))))
    return {"mae": mae, "rmse": rmse}


def _lazy_import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "Torch is required for dlinear/nlinear forecasting runs. "
            "Use `conda run -n pia python ...`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _resolve_plugin_flags(model_type: str, args: argparse.Namespace) -> Tuple[bool, bool, str]:
    plugin_enabled = str(args.pia_plugin_mode).lower() != "none"
    if plugin_enabled and model_type not in TORCH_MODELS:
        raise ValueError("PIA-lite plugin is only supported for dlinear/nlinear in this phase")

    anchor_arg = str(args.anchor_protection).lower()
    if anchor_arg == "true":
        anchor_enabled = True
    elif anchor_arg == "false":
        anchor_enabled = False
    else:
        anchor_enabled = plugin_enabled and model_type == "nlinear"

    scope = args.anchor_protection_scope if anchor_enabled else "none"
    return plugin_enabled, anchor_enabled, scope


def _variant_label(
    model_type: str,
    *,
    plugin_enabled: bool,
    plugin_mode: str,
    epsilon: float,
    anchor_enabled: bool,
) -> str:
    if not plugin_enabled:
        return f"{model_type}__baseline"
    anchor_tag = "_anchor" if anchor_enabled else ""
    return f"{model_type}__{plugin_mode}_eps{epsilon:.3f}{anchor_tag}"


def _model_params(
    model_type: str,
    args: argparse.Namespace,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if model_type == "persistence":
        params = {"strategy": "repeat_last_observed_target"}
    elif model_type == "ridge":
        params = {"alpha": float(args.ridge_alpha), "solver": "auto"}
    elif model_type == "elm":
        params = {
            "hidden_dim": int(args.elm_hidden_dim),
            "activation": str(args.elm_activation),
            "alpha": float(args.elm_alpha),
            "random_state": int(args.seed),
        }
    elif model_type == "dlinear":
        params = {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stop_patience": int(args.early_stop_patience),
            "decomp_kernel_size": int(args.decomp_kernel_size),
            "individual": bool(args.individual),
            "seed": int(args.seed),
            "device": str(args.device),
        }
    elif model_type == "nlinear":
        params = {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stop_patience": int(args.early_stop_patience),
            "individual": bool(args.individual),
            "seed": int(args.seed),
            "device": str(args.device),
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    if extra:
        params.update(extra)
    return params


def _resolve_device(device_arg: str, torch) -> object:
    key = str(device_arg).lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key in {"cpu", "cuda"}:
        return torch.device(key)
    raise ValueError(f"Unsupported device: {device_arg}")


def _predict_torch_in_batches(model, X: np.ndarray, batch_size: int, device, torch) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            yb = model(xb)
            preds.append(yb.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _fit_torch_forecaster(
    *,
    prepared: ETTPreparedData,
    model_type: str,
    args: argparse.Namespace,
) -> TorchFitArtifacts:
    torch, nn, DataLoader, TensorDataset = _lazy_import_torch()
    from models.forecast_linear_torch import DLinearMS, ForecastLinearConfig, NLinearMS

    plugin_enabled, anchor_enabled, anchor_scope = _resolve_plugin_flags(model_type, args)
    target_idx = prepared.spec.feature_columns.index(prepared.spec.target_column)
    device = _resolve_device(args.device, torch)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = ForecastLinearConfig(
        lookback=prepared.spec.lookback,
        horizon=prepared.spec.horizon,
        n_features=prepared.input_dim,
        target_idx=target_idx,
        individual=bool(args.individual),
        decomp_kernel_size=int(args.decomp_kernel_size),
    )
    if model_type == "dlinear":
        model = DLinearMS(cfg)
    elif model_type == "nlinear":
        model = NLinearMS(cfg)
    else:
        raise ValueError(f"Unsupported torch model_type: {model_type}")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.as_tensor(prepared.train.X, dtype=torch.float32),
        torch.as_tensor(prepared.train.y, dtype=torch.float32),
    )
    train_gen = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        generator=train_gen,
        drop_last=False,
    )

    pia_n_aug_per_epoch = int(round(prepared.train.n_samples * float(args.pia_aug_ratio))) if plugin_enabled else 0
    pia_n_aug_total_seen = 0
    pia_runtime_overhead_sec = 0.0
    best_state = None
    best_epoch = 0
    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    epochs_completed = 0
    patience_counter = 0
    fit_start = time.perf_counter()

    plugin_cfg = None
    if plugin_enabled:
        from transforms.forecast_pia_lite import PIALiteConfig, apply_pia_lite_batch

        plugin_cfg = PIALiteConfig(
            epsilon=float(args.pia_epsilon),
            aug_ratio=float(args.pia_aug_ratio),
            direction_source=str(args.pia_direction_source),
            anchor_protection_enabled=anchor_enabled,
            anchor_protection_scope=anchor_scope,
        )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if plugin_enabled and plugin_cfg is not None:
                x_aug, n_aug_batch, runtime_sec = apply_pia_lite_batch(xb, cfg=plugin_cfg)
                pia_runtime_overhead_sec += float(runtime_sec)
                pia_n_aug_total_seen += int(n_aug_batch)
                if n_aug_batch > 0:
                    y_aug = yb[:n_aug_batch]
                    xb = torch.cat([xb, x_aug], dim=0)
                    yb = torch.cat([yb, y_aug], dim=0)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected for model={model_type}, epoch={epoch}")
            loss.backward()
            optimizer.step()

        epochs_completed = epoch
        val_pred_scaled = _predict_torch_in_batches(model, prepared.val.X, int(args.batch_size), device, torch)
        val_pred_raw = _inverse_target(prepared, val_pred_scaled)
        val_true_raw = _inverse_target(prepared, prepared.val.y)
        val_metrics = _evaluate(val_true_raw, val_pred_raw)

        is_better = (
            val_metrics["mae"] < best_val_mae - 1e-12 or
            (abs(val_metrics["mae"] - best_val_mae) <= 1e-12 and val_metrics["rmse"] < best_val_rmse)
        )
        if is_better:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_mae = float(val_metrics["mae"])
            best_val_rmse = float(val_metrics["rmse"])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(args.early_stop_patience):
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint obtained for model={model_type}")
    model.load_state_dict(best_state)

    y_val_pred_scaled = _predict_torch_in_batches(model, prepared.val.X, int(args.batch_size), device, torch)
    y_test_pred_scaled = _predict_torch_in_batches(model, prepared.test.X, int(args.batch_size), device, torch)
    fit_runtime_sec = time.perf_counter() - fit_start

    model_params = _model_params(
        model_type,
        args,
        extra={
            "best_epoch": int(best_epoch),
            "best_val_mae": float(best_val_mae),
            "best_val_rmse": float(best_val_rmse),
            "epochs_completed": int(epochs_completed),
            "fit_runtime_sec": float(fit_runtime_sec),
        },
    )
    return TorchFitArtifacts(
        y_val_pred_scaled=y_val_pred_scaled,
        y_test_pred_scaled=y_test_pred_scaled,
        model_params=model_params,
        pia_n_aug_samples=int(pia_n_aug_per_epoch),
        pia_n_aug_samples_total_seen=int(pia_n_aug_total_seen),
        pia_runtime_overhead_sec=float(pia_runtime_overhead_sec),
        anchor_protection_enabled=bool(anchor_enabled),
        anchor_protection_scope=str(anchor_scope),
    )


def _normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    for col, default in SUMMARY_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    if "variant_label" in df.columns and "model_type" in df.columns:
        missing = df["variant_label"].astype(str).str.strip().isin({"", "nan", "None"})
        df.loc[missing, "variant_label"] = df.loc[missing, "model_type"].astype(str) + "__baseline"
    return df


def _run_single(
    *,
    prepared: ETTPreparedData,
    model_type: str,
    args: argparse.Namespace,
) -> ForecastResult:
    target_idx = prepared.spec.feature_columns.index(prepared.spec.target_column)
    plugin_enabled, anchor_enabled, anchor_scope = _resolve_plugin_flags(model_type, args)
    pia_plugin_mode = str(args.pia_plugin_mode).lower() if plugin_enabled else "none"
    pia_epsilon = float(args.pia_epsilon) if plugin_enabled else 0.0
    pia_aug_ratio = float(args.pia_aug_ratio) if plugin_enabled else 0.0
    variant_label = _variant_label(
        model_type,
        plugin_enabled=plugin_enabled,
        plugin_mode=pia_plugin_mode,
        epsilon=pia_epsilon,
        anchor_enabled=anchor_enabled,
    )

    y_val_raw = _inverse_target(prepared, prepared.val.y)
    y_test_raw = _inverse_target(prepared, prepared.test.y)
    model_params_extra: Dict[str, object] | None = None
    pia_n_aug_samples = 0
    pia_n_aug_samples_total_seen = 0
    pia_runtime_overhead_sec = 0.0

    if model_type == "persistence":
        y_val_pred_scaled = _persistence_predict(prepared.val.X, prepared.spec.horizon, target_idx)
        y_test_pred_scaled = _persistence_predict(prepared.test.X, prepared.spec.horizon, target_idx)
    elif model_type in TORCH_MODELS:
        arts = _fit_torch_forecaster(prepared=prepared, model_type=model_type, args=args)
        y_val_pred_scaled = arts.y_val_pred_scaled
        y_test_pred_scaled = arts.y_test_pred_scaled
        model_params_extra = arts.model_params
        pia_n_aug_samples = arts.pia_n_aug_samples
        pia_n_aug_samples_total_seen = arts.pia_n_aug_samples_total_seen
        pia_runtime_overhead_sec = arts.pia_runtime_overhead_sec
        anchor_enabled = arts.anchor_protection_enabled
        anchor_scope = arts.anchor_protection_scope
        variant_label = _variant_label(
            model_type,
            plugin_enabled=plugin_enabled,
            plugin_mode=pia_plugin_mode,
            epsilon=pia_epsilon,
            anchor_enabled=anchor_enabled,
        )
    else:
        X_train = _flatten_X(prepared.train.X)
        X_val = _flatten_X(prepared.val.X)
        X_test = _flatten_X(prepared.test.X)
        y_train = np.asarray(prepared.train.y, dtype=np.float64)
        if model_type == "ridge":
            model = Ridge(alpha=float(args.ridge_alpha))
        elif model_type == "elm":
            model = DirectMultiOutputELMRegressor(
                hidden_dim=int(args.elm_hidden_dim),
                activation=str(args.elm_activation),
                alpha=float(args.elm_alpha),
                random_state=int(args.seed),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        model.fit(X_train, y_train)
        y_val_pred_scaled = np.asarray(model.predict(X_val), dtype=np.float64)
        y_test_pred_scaled = np.asarray(model.predict(X_test), dtype=np.float64)

    y_val_pred_raw = _inverse_target(prepared, y_val_pred_scaled)
    y_test_pred_raw = _inverse_target(prepared, y_test_pred_scaled)
    val_metrics = _evaluate(y_val_raw, y_val_pred_raw)
    test_metrics = _evaluate(y_test_raw, y_test_pred_raw)

    run_dir = (
        Path(args.out_root)
        / prepared.spec.dataset
        / model_type
        / f"L{prepared.spec.lookback}_H{prepared.spec.horizon}"
        / args.run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "dataset": prepared.spec.dataset,
        "target": prepared.spec.target_column,
        "lookback": prepared.spec.lookback,
        "horizon": prepared.spec.horizon,
        "model_type": model_type,
        "run_name": args.run_name,
        "variant_label": variant_label,
        "pia_plugin_mode": pia_plugin_mode,
        "pia_epsilon": pia_epsilon,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    split_rows = summarize_split_rows(prepared)
    run_meta = {
        "dataset": prepared.spec.dataset,
        "dataset_path": prepared.spec.dataset_path,
        "split_policy": prepared.split_policy,
        "target": prepared.spec.target_column,
        "feature_columns": prepared.spec.feature_columns,
        "lookback": prepared.spec.lookback,
        "horizon": prepared.spec.horizon,
        "forecast_mode": prepared.forecast_mode,
        "train_rows": split_rows["train_rows"],
        "val_rows": split_rows["val_rows"],
        "test_rows": split_rows["test_rows"],
        "total_rows_csv": split_rows["total_rows_csv"],
        "used_rows_official": split_rows["used_rows_official"],
        "n_train_samples": prepared.train.n_samples,
        "n_val_samples": prepared.val.n_samples,
        "n_test_samples": prepared.test.n_samples,
        "input_dim": prepared.input_dim,
        "n_features_per_step": prepared.input_dim,
        "flattened_input_dim": int(prepared.input_dim * prepared.spec.lookback),
        "input_norm_mode": prepared.input_norm_mode,
        "target_norm_mode": prepared.target_norm_mode,
        "target_scaling_used": True,
        "metrics_on_original_target_scale": True,
        "inverse_transform_before_metrics": True,
        "model_type": model_type,
        "run_name": args.run_name,
        "variant_label": variant_label,
        "model_params": model_params_extra or _model_params(model_type, args),
        "pia_plugin_enabled": bool(plugin_enabled),
        "pia_plugin_mode": pia_plugin_mode,
        "pia_target_space": "raw_history_flat",
        "pia_epsilon": float(pia_epsilon),
        "pia_aug_ratio": float(pia_aug_ratio),
        "pia_direction_source": str(args.pia_direction_source) if plugin_enabled else "none",
        "pia_n_aug_samples": int(pia_n_aug_samples),
        "pia_n_aug_samples_total_seen": int(pia_n_aug_samples_total_seen),
        "pia_runtime_overhead_sec": float(pia_runtime_overhead_sec),
        "anchor_protection_enabled": bool(anchor_enabled),
        "anchor_protection_scope": str(anchor_scope),
        "split_row_ranges": {
            "train": {
                "raw_start": prepared.train.raw_row_start,
                "raw_end": prepared.train.raw_row_end,
                "source_start": prepared.train.source_row_start,
                "source_end": prepared.train.source_row_end,
            },
            "val": {
                "raw_start": prepared.val.raw_row_start,
                "raw_end": prepared.val.raw_row_end,
                "source_start": prepared.val.source_row_start,
                "source_end": prepared.val.source_row_end,
            },
            "test": {
                "raw_start": prepared.test.raw_row_start,
                "raw_end": prepared.test.raw_row_end,
                "source_start": prepared.test.source_row_start,
                "source_end": prepared.test.source_row_end,
            },
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return ForecastResult(
        dataset=prepared.spec.dataset,
        model_type=model_type,
        lookback=prepared.spec.lookback,
        horizon=prepared.spec.horizon,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        val_mae=val_metrics["mae"],
        val_rmse=val_metrics["rmse"],
        run_dir=str(run_dir.resolve()),
        run_name=str(args.run_name),
        variant_label=variant_label,
        pia_plugin_enabled=bool(plugin_enabled),
        pia_plugin_mode=pia_plugin_mode,
        pia_epsilon=float(pia_epsilon),
        pia_aug_ratio=float(pia_aug_ratio),
        anchor_protection_enabled=bool(anchor_enabled),
        anchor_protection_scope=str(anchor_scope),
    )


def _write_summaries(out_root: Path, dataset: str, results: List[ForecastResult]) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "dataset": r.dataset,
                "model_type": r.model_type,
                "run_name": r.run_name,
                "variant_label": r.variant_label,
                "lookback": r.lookback,
                "horizon": r.horizon,
                "mae": r.mae,
                "rmse": r.rmse,
                "val_mae": r.val_mae,
                "val_rmse": r.val_rmse,
                "pia_plugin_enabled": r.pia_plugin_enabled,
                "pia_plugin_mode": r.pia_plugin_mode,
                "pia_epsilon": r.pia_epsilon,
                "pia_aug_ratio": r.pia_aug_ratio,
                "anchor_protection_enabled": r.anchor_protection_enabled,
                "anchor_protection_scope": r.anchor_protection_scope,
                "run_dir": r.run_dir,
            }
        )
    df = pd.DataFrame(rows)
    dataset_root = Path(out_root) / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    summary_path = dataset_root / "summary_per_run.csv"
    lock_path = dataset_root / ".summary.lock"
    with lock_path.open("w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if summary_path.exists():
            prev = pd.read_csv(summary_path)
            prev = _normalize_summary_df(prev)
            df = _normalize_summary_df(df)
            df = pd.concat([prev, df], ignore_index=True)
            df = df.drop_duplicates(subset=["run_dir"], keep="last")
        else:
            df = _normalize_summary_df(df)
        df = df.sort_values(["model_type", "pia_plugin_mode", "pia_epsilon", "lookback", "horizon", "run_dir"]).reset_index(drop=True)
        df.to_csv(summary_path, index=False)

        agg = (
            df.groupby(["dataset", "model_type", "pia_plugin_mode", "pia_epsilon", "variant_label"], as_index=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                n_runs=("mae", "size"),
            )
            .fillna(0.0)
        )
        agg.to_csv(dataset_root / "summary_agg.csv", index=False)

        base_cols = ["dataset", "lookback", "horizon", "mae", "rmse"]
        df_p = df[(df["model_type"] == "persistence") & (df["pia_plugin_mode"] == "none")][base_cols].rename(
            columns={"mae": "persistence_mae", "rmse": "persistence_rmse"}
        )
        compare = df.merge(df_p, on=["dataset", "lookback", "horizon"], how="left")
        compare["delta_mae_vs_persistence"] = compare["mae"] - compare["persistence_mae"]
        compare["delta_rmse_vs_persistence"] = compare["rmse"] - compare["persistence_rmse"]
        compare = compare.sort_values(["model_type", "pia_plugin_mode", "pia_epsilon", "lookback", "horizon", "run_dir"]).reset_index(drop=True)
        compare.to_csv(dataset_root / "summary_compare_vs_persistence.csv", index=False)
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETT forecasting probe runner")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--csv-path", default=str(DEFAULT_ETTH1_PATH))
    parser.add_argument("--target", default="OT")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[96])
    parser.add_argument("--horizons", nargs="+", type=int, default=[24])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["persistence"],
        choices=["persistence", "ridge", "elm", "dlinear", "nlinear"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--elm-hidden-dim", type=int, default=512)
    parser.add_argument("--elm-activation", default="tanh", choices=["tanh", "relu", "sigmoid"])
    parser.add_argument("--elm-alpha", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--decomp-kernel-size", type=int, default=25)
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--pia-plugin-mode", default="none", choices=["none", "pia_lite_pca_top1"])
    parser.add_argument("--pia-epsilon", type=float, default=0.0)
    parser.add_argument("--pia-aug-ratio", type=float, default=0.0)
    parser.add_argument("--pia-direction-source", default="batch_local", choices=["batch_local"])
    parser.add_argument("--anchor-protection", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--anchor-protection-scope", default="last_step_only", choices=["last_step_only", "none"])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-name", default="run0")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(int(args.seed))
    out_root = Path(args.out_root)

    results: List[ForecastResult] = []
    for lookback in args.lookbacks:
        for horizon in args.horizons:
            prepared = prepare_ett_direct_forecast(
                dataset=args.dataset,
                csv_path=args.csv_path,
                target_column=args.target,
                lookback=int(lookback),
                horizon=int(horizon),
            )
            for model_type in args.models:
                res = _run_single(prepared=prepared, model_type=model_type, args=args)
                print(
                    f"[done] dataset={res.dataset} model={res.model_type} "
                    f"plugin={res.pia_plugin_mode} eps={res.pia_epsilon:.3f} "
                    f"L={res.lookback} H={res.horizon} mae={res.mae:.6f} rmse={res.rmse:.6f}",
                    flush=True,
                )
                results.append(res)

    _write_summaries(out_root, args.dataset, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
