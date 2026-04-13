from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.forecast_zseq import ZSeqPreparedData, prepare_ett_zseq_probe, summarize_zseq_rows


DEFAULT_CSV = PROJECT_ROOT / "data" / "MTSFC" / "EET" / "ETTh1.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "out" / "pia_dynamics_probe"


@dataclass(frozen=True)
class ProbeResult:
    dataset: str
    stage_id: str
    variant: str
    lookback: int
    horizon: int
    z_mae: float | None
    z_mse: float | None
    val_z_mae: float | None
    val_z_mse: float | None
    direction_source: str
    transition_model: str
    center_update_mode: str
    loss_mode: str
    n_dirs: int
    run_dir: str


def _z_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return {
        "z_mae": float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),
        "z_mse": float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))),
        "z_rmse": float(math.sqrt(np.mean(diff ** 2))),
    }


def _principal_angles_deg(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    # basis_*: [k, d], PCA rows are orthonormal
    m = np.asarray(basis_a, dtype=np.float64) @ np.asarray(basis_b, dtype=np.float64).T
    _, s, _ = np.linalg.svd(m, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def _projection_energy_ratio(delta: np.ndarray, basis: np.ndarray) -> float:
    coef = np.asarray(delta, dtype=np.float64) @ np.asarray(basis, dtype=np.float64).T
    proj = coef @ np.asarray(basis, dtype=np.float64)
    denom = float(np.sum(np.asarray(delta, dtype=np.float64) ** 2)) + 1e-12
    num = float(np.sum(proj ** 2))
    return num / denom


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summaries(out_root: Path, dataset: str, results: List[ProbeResult]) -> None:
    rows = [
        {
            "dataset": r.dataset,
            "stage_id": r.stage_id,
            "variant": r.variant,
            "lookback": r.lookback,
            "horizon": r.horizon,
            "z_mae": r.z_mae,
            "z_mse": r.z_mse,
            "val_z_mae": r.val_z_mae,
            "val_z_mse": r.val_z_mse,
            "direction_source": r.direction_source,
            "transition_model": r.transition_model,
            "center_update_mode": r.center_update_mode,
            "loss_mode": r.loss_mode,
            "n_dirs": r.n_dirs,
            "run_dir": r.run_dir,
        }
        for r in results
    ]
    df_new = pd.DataFrame(rows)
    dataset_root = out_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    summary_path = dataset_root / "summary_per_run.csv"
    lock_path = dataset_root / ".summary.lock"
    with lock_path.open("w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        if summary_path.exists():
            df_old = pd.read_csv(summary_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
            key_cols = ["dataset", "stage_id", "variant", "lookback", "horizon", "n_dirs"]
            df = df.drop_duplicates(subset=key_cols, keep="last")
        else:
            df = df_new
        df.to_csv(summary_path, index=False)

        agg = (
            df.groupby(
                ["dataset", "stage_id", "variant", "direction_source", "transition_model", "center_update_mode", "loss_mode", "n_dirs"],
                as_index=False,
            )
            .agg(
                lookback=("lookback", "first"),
                horizon=("horizon", "first"),
                z_mae=("z_mae", "mean"),
                z_mse=("z_mse", "mean"),
                val_z_mae=("val_z_mae", "mean"),
                val_z_mse=("val_z_mse", "mean"),
            )
        )
        agg.to_csv(dataset_root / "summary_agg.csv", index=False)
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _common_meta(prepared: ZSeqPreparedData, args: argparse.Namespace) -> Dict[str, object]:
    split_rows = summarize_zseq_rows(prepared)
    return {
        "dataset": prepared.spec.dataset,
        "dataset_path": prepared.spec.dataset_path,
        "representation_space": "z",
        "lookback": int(prepared.spec.lookback),
        "horizon": int(prepared.spec.horizon),
        "cov_estimator": prepared.spec.cov_estimator,
        "spd_eps": float(prepared.spec.spd_eps),
        "feature_columns": prepared.spec.feature_columns,
        "split_policy": prepared.split_policy,
        "raw_norm_mode": prepared.raw_norm_mode,
        "tangent_center_mode": prepared.tangent_center_mode,
        "z_dim": int(prepared.z_dim),
        "total_rows_csv": split_rows["total_rows_csv"],
        "used_rows_official": split_rows["used_rows_official"],
        "train_rows": split_rows["train_rows"],
        "val_rows": split_rows["val_rows"],
        "test_rows": split_rows["test_rows"],
        "n_train_samples": int(prepared.train.n_samples),
        "n_val_samples": int(prepared.val.n_samples),
        "n_test_samples": int(prepared.test.n_samples),
        "eval_metrics": ["z_mae", "z_mse"],
        "center_update_mode": "none",
        "loss_mode": "transition_only",
        "max_stage_requested": str(args.max_stage),
    }


def _run_v0(prepared: ZSeqPreparedData, args: argparse.Namespace, results: List[ProbeResult]) -> bool:
    dataset_root = Path(args.out_root) / prepared.spec.dataset / f"L{prepared.spec.lookback}_H{prepared.spec.horizon}"

    # z-persistence
    for variant in ("z_persistence", "z_ridge"):
        if variant == "z_persistence":
            y_val_pred = prepared.val.X
            y_test_pred = prepared.test.X
            model_params = {"strategy": "identity_transition"}
        else:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_train_s = x_scaler.fit_transform(prepared.train.X)
            Y_train_s = y_scaler.fit_transform(prepared.train.Y)
            X_val_s = x_scaler.transform(prepared.val.X)
            X_test_s = x_scaler.transform(prepared.test.X)
            ridge = Ridge(alpha=float(args.ridge_alpha), random_state=int(args.seed))
            ridge.fit(X_train_s, Y_train_s)
            y_val_pred = y_scaler.inverse_transform(ridge.predict(X_val_s))
            y_test_pred = y_scaler.inverse_transform(ridge.predict(X_test_s))
            model_params = {
                "alpha": float(args.ridge_alpha),
                "seed": int(args.seed),
                "x_norm_mode": "train_only_standardization",
                "y_norm_mode": "train_only_standardization",
            }

        val_metrics = _z_metrics(prepared.val.Y, y_val_pred)
        test_metrics = _z_metrics(prepared.test.Y, y_test_pred)
        run_dir = dataset_root / "V0" / variant / args.run_name
        metrics = {
            "dataset": prepared.spec.dataset,
            "stage_id": "V0",
            "variant": variant,
            "lookback": prepared.spec.lookback,
            "horizon": prepared.spec.horizon,
            "z_mae": test_metrics["z_mae"],
            "z_mse": test_metrics["z_mse"],
            "val_z_mae": val_metrics["z_mae"],
            "val_z_mse": val_metrics["z_mse"],
        }
        meta = _common_meta(prepared, args)
        meta.update(
            {
                "stage_id": "V0",
                "direction_source": "none",
                "transition_model": variant,
                "n_dirs": 0,
                "model_params": model_params,
            }
        )
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "run_meta.json", meta)
        results.append(
            ProbeResult(
                dataset=prepared.spec.dataset,
                stage_id="V0",
                variant=variant,
                lookback=prepared.spec.lookback,
                horizon=prepared.spec.horizon,
                z_mae=test_metrics["z_mae"],
                z_mse=test_metrics["z_mse"],
                val_z_mae=val_metrics["z_mae"],
                val_z_mse=val_metrics["z_mse"],
                direction_source="none",
                transition_model=variant,
                center_update_mode="none",
                loss_mode="transition_only",
                n_dirs=0,
                run_dir=str(run_dir.resolve()),
            )
        )

    per = next(r for r in results if r.stage_id == "V0" and r.variant == "z_persistence")
    rid = next(r for r in results if r.stage_id == "V0" and r.variant == "z_ridge")
    return bool((rid.z_mae < per.z_mae) and (rid.z_mse < per.z_mse))


def _run_v1(prepared: ZSeqPreparedData, args: argparse.Namespace, results: List[ProbeResult]) -> None:
    dataset_root = Path(args.out_root) / prepared.spec.dataset / f"L{prepared.spec.lookback}_H{prepared.spec.horizon}"
    X_train = np.asarray(prepared.train.X, dtype=np.float64)
    D_train = np.asarray(prepared.train.delta, dtype=np.float64)

    for k in args.n_dirs:
        k_eff = min(int(k), X_train.shape[1], D_train.shape[1])
        pca_state = PCA(n_components=k_eff, random_state=int(args.seed)).fit(X_train)
        pca_delta = PCA(n_components=k_eff, random_state=int(args.seed)).fit(D_train)
        basis_state = np.asarray(pca_state.components_, dtype=np.float64)
        basis_delta = np.asarray(pca_delta.components_, dtype=np.float64)
        angles = _principal_angles_deg(basis_state, basis_delta)
        top1_abs_cos = float(abs(np.dot(basis_state[0], basis_delta[0])))
        state_energy = _projection_energy_ratio(D_train, basis_state)
        delta_energy = _projection_energy_ratio(D_train, basis_delta)

        run_dir = dataset_root / "V1" / f"state_vs_delta_k{k_eff}" / args.run_name
        metrics = {
            "dataset": prepared.spec.dataset,
            "stage_id": "V1",
            "variant": f"state_vs_delta_k{k_eff}",
            "lookback": prepared.spec.lookback,
            "horizon": prepared.spec.horizon,
            "top1_abs_cosine": top1_abs_cos,
            "mean_principal_angle_deg": float(np.mean(angles)),
            "max_principal_angle_deg": float(np.max(angles)),
            "transition_energy_capture_state": state_energy,
            "transition_energy_capture_delta": delta_energy,
        }
        meta = _common_meta(prepared, args)
        meta.update(
            {
                "stage_id": "V1",
                "direction_source": "state_vs_delta",
                "transition_model": "subspace_comparison",
                "n_dirs": int(k_eff),
                "state_explained_variance_ratio": pca_state.explained_variance_ratio_.tolist(),
                "delta_explained_variance_ratio": pca_delta.explained_variance_ratio_.tolist(),
                "principal_angles_deg": angles.tolist(),
            }
        )
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "run_meta.json", meta)
        results.append(
            ProbeResult(
                dataset=prepared.spec.dataset,
                stage_id="V1",
                variant=f"state_vs_delta_k{k_eff}",
                lookback=prepared.spec.lookback,
                horizon=prepared.spec.horizon,
                z_mae=None,
                z_mse=None,
                val_z_mae=None,
                val_z_mse=None,
                direction_source="state_vs_delta",
                transition_model="subspace_comparison",
                center_update_mode="none",
                loss_mode="transition_only",
                n_dirs=int(k_eff),
                run_dir=str(run_dir.resolve()),
            )
        )


def _oracle_project_transition(X: np.ndarray, Y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    delta = np.asarray(Y, dtype=np.float64) - np.asarray(X, dtype=np.float64)
    coef = delta @ np.asarray(basis, dtype=np.float64).T
    delta_proj = coef @ np.asarray(basis, dtype=np.float64)
    return np.asarray(X, dtype=np.float64) + delta_proj


def _run_v2(prepared: ZSeqPreparedData, args: argparse.Namespace, results: List[ProbeResult]) -> None:
    dataset_root = Path(args.out_root) / prepared.spec.dataset / f"L{prepared.spec.lookback}_H{prepared.spec.horizon}"
    X_train = np.asarray(prepared.train.X, dtype=np.float64)
    D_train = np.asarray(prepared.train.delta, dtype=np.float64)

    for source_name, source_data in (("state", X_train), ("delta", D_train)):
        for k in args.n_dirs:
            k_eff = min(int(k), source_data.shape[1])
            pca = PCA(n_components=k_eff, random_state=int(args.seed)).fit(source_data)
            basis = np.asarray(pca.components_, dtype=np.float64)
            y_val_pred = _oracle_project_transition(prepared.val.X, prepared.val.Y, basis)
            y_test_pred = _oracle_project_transition(prepared.test.X, prepared.test.Y, basis)
            val_metrics = _z_metrics(prepared.val.Y, y_val_pred)
            test_metrics = _z_metrics(prepared.test.Y, y_test_pred)
            run_dir = dataset_root / "V2" / f"{source_name}_basis_k{k_eff}" / args.run_name
            metrics = {
                "dataset": prepared.spec.dataset,
                "stage_id": "V2",
                "variant": f"{source_name}_basis_k{k_eff}",
                "lookback": prepared.spec.lookback,
                "horizon": prepared.spec.horizon,
                "z_mae": test_metrics["z_mae"],
                "z_mse": test_metrics["z_mse"],
                "val_z_mae": val_metrics["z_mae"],
                "val_z_mse": val_metrics["z_mse"],
                "transition_energy_capture_train": _projection_energy_ratio(D_train, basis),
            }
            meta = _common_meta(prepared, args)
            meta.update(
                {
                    "stage_id": "V2",
                    "direction_source": source_name,
                    "transition_model": "oracle_projection_basis",
                    "n_dirs": int(k_eff),
                    "basis_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "interpretation_note": "Oracle subspace projection on true delta; representational upper bound, not deployable predictor",
                }
            )
            _write_json(run_dir / "metrics.json", metrics)
            _write_json(run_dir / "run_meta.json", meta)
            results.append(
                ProbeResult(
                    dataset=prepared.spec.dataset,
                    stage_id="V2",
                    variant=f"{source_name}_basis_k{k_eff}",
                    lookback=prepared.spec.lookback,
                    horizon=prepared.spec.horizon,
                    z_mae=test_metrics["z_mae"],
                    z_mse=test_metrics["z_mse"],
                    val_z_mae=val_metrics["z_mae"],
                    val_z_mse=val_metrics["z_mse"],
                    direction_source=source_name,
                    transition_model="oracle_projection_basis",
                    center_update_mode="none",
                    loss_mode="transition_only",
                    n_dirs=int(k_eff),
                    run_dir=str(run_dir.resolve()),
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIA dynamics feasibility probe in z-space")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--cov-estimator", choices=["sample", "oas", "lw", "ledoitwolf"], default="oas")
    parser.add_argument("--spd-eps", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-dirs", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="run0")
    parser.add_argument("--max-stage", choices=["V0", "V1", "V2"], default="V2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_ett_zseq_probe(
        dataset=args.dataset,
        csv_path=args.csv_path,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        cov_estimator=str(args.cov_estimator),
        spd_eps=float(args.spd_eps),
    )
    results: List[ProbeResult] = []
    v0_pass = _run_v0(prepared, args, results)
    _write_summaries(Path(args.out_root), prepared.spec.dataset, results)
    if args.max_stage == "V0" or not v0_pass:
        return

    _run_v1(prepared, args, results)
    _write_summaries(Path(args.out_root), prepared.spec.dataset, results)
    if args.max_stage == "V1":
        return

    _run_v2(prepared, args, results)
    _write_summaries(Path(args.out_root), prepared.spec.dataset, results)


if __name__ == "__main__":
    main()
