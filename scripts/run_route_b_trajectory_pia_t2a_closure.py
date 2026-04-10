#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from route_b_unified.trajectory_classifier import TrajectoryModelConfig  # noqa: E402
from route_b_unified.trajectory_pia_evaluator import (  # noqa: E402
    TrajectoryPIAEvalConfig,
    TrajectoryPIAEvalResult,
    evaluate_trajectory_pia_t2a,
)
from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig  # noqa: E402
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
    TrajectoryRepresentationState,
    build_trajectory_representation,
)


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _parse_float_list(text: str) -> List[float]:
    out = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("float list cannot be empty")
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _mean_std(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _format_mean_std(values: List[float]) -> str:
    mean, std = _mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def _float_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "")


def _load_t2a_reference_maps(per_seed_csv: str, summary_csv: str) -> tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], float], Dict[str, Dict[str, float]]]:
    baseline_map: Dict[Tuple[str, int], float] = {}
    best_map: Dict[Tuple[str, int], float] = {}
    summary_best_cfg: Dict[str, Dict[str, float]] = {}

    if os.path.isfile(per_seed_csv):
        df = pd.read_csv(per_seed_csv)
        if not df.empty:
            for _, row in df.iterrows():
                key = (str(row["dataset"]).strip().lower(), int(row["seed"]))
                mode = str(row["operator_mode"]).strip().lower()
                score = float(row["test_macro_f1"])
                if mode == "baseline":
                    baseline_map[key] = score
                else:
                    best_map[key] = max(float(best_map.get(key, float("-inf"))), score)

    if os.path.isfile(summary_csv):
        sdf = pd.read_csv(summary_csv)
        for _, row in sdf.iterrows():
            dataset = str(row["dataset"]).strip().lower()
            best_mode = str(row.get("best_mode", "")).strip().lower()
            if best_mode == "operator_smoothed":
                summary_best_cfg[dataset] = {"gamma_main": 0.10, "smooth_lambda": 0.50}
            elif best_mode == "operator_unsmoothed":
                summary_best_cfg[dataset] = {"gamma_main": 0.10, "smooth_lambda": 0.00}
    return baseline_map, best_map, summary_best_cfg


def _build_states_and_basis(
    *,
    datasets: List[str],
    seeds: List[int],
    repr_cfg_kwargs: Dict[str, object],
) -> tuple[Dict[Tuple[str, int], TrajectoryRepresentationState], Dict[Tuple[str, int], TrajectoryPIAOperator]]:
    states: Dict[Tuple[str, int], TrajectoryRepresentationState] = {}
    basis_map: Dict[Tuple[str, int], TrajectoryPIAOperator] = {}
    for dataset in datasets:
        for seed in seeds:
            state = build_trajectory_representation(
                TrajectoryRepresentationConfig(
                    dataset=str(dataset),
                    seed=int(seed),
                    val_fraction=float(repr_cfg_kwargs["val_fraction"]),
                    spd_eps=float(repr_cfg_kwargs["spd_eps"]),
                    prop_win_ratio=float(repr_cfg_kwargs["prop_win_ratio"]),
                    prop_hop_ratio=float(repr_cfg_kwargs["prop_hop_ratio"]),
                    min_window_extra_channels=int(repr_cfg_kwargs["min_window_extra_channels"]),
                    min_hop_len=int(repr_cfg_kwargs["min_hop_len"]),
                )
            )
            key = (str(dataset), int(seed))
            states[key] = state
            basis = TrajectoryPIAOperator(
                TrajectoryPIAOperatorConfig(seed=int(seed))
            ).fit(state.train.z_seq_list)
            basis_map[key] = basis
    return states, basis_map


def _evaluate_cached(
    *,
    cache: Dict[Tuple[str, int, str, float, float], TrajectoryPIAEvalResult],
    state: TrajectoryRepresentationState,
    basis: TrajectoryPIAOperator,
    seed: int,
    model_cfg: TrajectoryModelConfig,
    eval_common: Dict[str, object],
    dataset: str,
    operator_mode: str,
    gamma_main: float,
    smooth_lambda: float,
) -> TrajectoryPIAEvalResult:
    key = (str(dataset), int(seed), str(operator_mode), round(float(gamma_main), 4), round(float(smooth_lambda), 4))
    if key in cache:
        return cache[key]
    eval_cfg = TrajectoryPIAEvalConfig(
        operator_mode=str(operator_mode),
        gamma_main=float(gamma_main),
        smooth_lambda=float(smooth_lambda),
        epochs=int(eval_common["epochs"]),
        batch_size=int(eval_common["batch_size"]),
        lr=float(eval_common["lr"]),
        weight_decay=float(eval_common["weight_decay"]),
        patience=int(eval_common["patience"]),
        device=str(eval_common["device"]),
    )
    result = evaluate_trajectory_pia_t2a(
        state,
        seed=int(seed),
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
        operator_cfg=TrajectoryPIAOperatorConfig(seed=int(seed)),
        prefit_operator=basis,
    )
    cache[key] = result
    return result


def _best_config_frequency(df: pd.DataFrame) -> tuple[int, str]:
    if df.empty:
        return 0, ""
    winners = (
        df.sort_values(["seed", "test_macro_f1"], ascending=[True, False])
        .groupby("seed", as_index=False)
        .first()["config_id"]
        .tolist()
    )
    if not winners:
        return 0, ""
    values, counts = np.unique(np.asarray(winners, dtype=object), return_counts=True)
    idx = int(np.argmax(counts))
    return int(counts[idx]), str(values[idx])


def _detect_phase_b_mode(summary_df: pd.DataFrame) -> tuple[str | None, Dict[str, float]]:
    gamma_df = summary_df[(summary_df["dataset"] == "selfregulationscp1") & (summary_df["phase"] == "phase_a_gamma")].copy()
    smooth_df = summary_df[(summary_df["dataset"] == "selfregulationscp1") & (summary_df["phase"] == "phase_a_smooth")].copy()
    gamma_vals = gamma_df["macro_f1_mean"].tolist()
    smooth_vals = smooth_df["macro_f1_mean"].tolist()
    gamma_range = float(max(gamma_vals) - min(gamma_vals)) if gamma_vals else 0.0
    smooth_range = float(max(smooth_vals) - min(smooth_vals)) if smooth_vals else 0.0
    gamma_freq, gamma_win = _best_config_frequency(
        summary_df[(summary_df["dataset"] == "selfregulationscp1") & (summary_df["phase"] == "phase_a_gamma_per_seed")]
    )
    smooth_freq, smooth_win = _best_config_frequency(
        summary_df[(summary_df["dataset"] == "selfregulationscp1") & (summary_df["phase"] == "phase_a_smooth_per_seed")]
    )

    sensitivity_floor = 0.003
    gamma_stable = gamma_range >= sensitivity_floor and gamma_freq >= 2
    smooth_stable = smooth_range >= sensitivity_floor and smooth_freq >= 2

    if gamma_stable and smooth_stable:
        mode = "both"
    elif gamma_stable:
        mode = "gamma"
    elif smooth_stable:
        mode = "smooth"
    else:
        mode = None
    return mode, {
        "gamma_range": float(gamma_range),
        "smooth_range": float(smooth_range),
        "gamma_best_freq": int(gamma_freq),
        "smooth_best_freq": int(smooth_freq),
        "gamma_best_config_id": str(gamma_win),
        "smooth_best_config_id": str(smooth_win),
        "phase_b_mode": mode or "none",
    }


def _pick_single_candidate(summary_df: pd.DataFrame) -> Dict[str, object]:
    scp1 = summary_df[
        (summary_df["dataset"] == "selfregulationscp1")
        & (summary_df["phase"].isin(["phase_a_gamma", "phase_a_smooth", "phase_b"]))
    ].copy()
    if scp1.empty:
        raise ValueError("cannot pick candidate from empty SCP1 summary")
    scp1 = scp1.sort_values(
        ["macro_f1_mean", "macro_f1_std", "continuity_distortion_ratio_mean", "config_id"],
        ascending=[False, True, True, True],
    )
    row = scp1.iloc[0]
    return {
        "config_id": str(row["config_id"]),
        "gamma_main": float(row["gamma_main"]),
        "smooth_lambda": float(row["smooth_lambda"]),
        "macro_f1_mean": float(row["macro_f1_mean"]),
        "macro_f1_std": float(row["macro_f1_std"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="T2a closure round: SCP1-focused gamma/smooth sensitivity under fixed basis.")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_trajectory_pia_t2a_closure_20260329_formal")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--gru-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--gamma-sweep", type=str, default="0.05,0.08,0.10,0.12,0.15")
    p.add_argument("--smooth-sweep", type=str, default="0.00,0.25,0.50,0.75")
    p.add_argument(
        "--t2a-reference-per-seed-csv",
        type=str,
        default="/home/THL/project/MTS-PIA/out/route_b_trajectory_pia_t2a_20260329_formal/trajectory_pia_t2a_per_seed.csv",
    )
    p.add_argument(
        "--t2a-reference-summary-csv",
        type=str,
        default="/home/THL/project/MTS-PIA/out/route_b_trajectory_pia_t2a_20260329_formal/trajectory_pia_t2a_dataset_summary.csv",
    )
    args = p.parse_args()

    seeds = _parse_seed_list(args.seeds)
    gamma_sweep = _parse_float_list(args.gamma_sweep)
    smooth_sweep = _parse_float_list(args.smooth_sweep)
    out_root = str(args.out_root)
    _ensure_dir(out_root)

    t2a_baseline_ref, t2a_best_ref, t2a_best_cfg = _load_t2a_reference_maps(
        str(args.t2a_reference_per_seed_csv),
        str(args.t2a_reference_summary_csv),
    )

    repr_cfg_kwargs = {
        "val_fraction": float(args.val_fraction),
        "spd_eps": float(args.spd_eps),
        "prop_win_ratio": float(args.prop_win_ratio),
        "prop_hop_ratio": float(args.prop_hop_ratio),
        "min_window_extra_channels": int(args.min_window_extra_channels),
        "min_hop_len": int(args.min_hop_len),
    }
    eval_common = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "device": str(args.device),
    }

    states, basis_map = _build_states_and_basis(
        datasets=["selfregulationscp1", "natops"],
        seeds=seeds,
        repr_cfg_kwargs=repr_cfg_kwargs,
    )
    for (dataset, seed), state in states.items():
        seed_dir = os.path.join(out_root, f"{dataset}_seed{seed}")
        _ensure_dir(seed_dir)
        basis = basis_map[(dataset, seed)].get_artifacts()
        _write_json(
            os.path.join(seed_dir, "trajectory_pia_t2a_closure_split_meta.json"),
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "split_meta": dict(state.split_meta),
                "meta": dict(state.meta),
                "window_len": int(state.window_len),
                "hop_len": int(state.hop_len),
                "basis_meta": {
                    "pooled_window_count": int(basis.pooled_window_count),
                    "z_dim": int(basis.z_dim),
                    "shared_basis_mode": str(basis.meta.get("shared_basis_mode", "")),
                    "basis_learning_scope": str(basis.meta.get("basis_learning_scope", "")),
                },
            },
        )

    model_cfgs = {
        key: TrajectoryModelConfig(
            z_dim=int(state.z_dim),
            num_classes=int(state.num_classes),
            gru_hidden_dim=int(args.gru_hidden_dim),
            dropout=float(args.dropout),
        )
        for key, state in states.items()
    }
    result_cache: Dict[Tuple[str, int, str, float, float], TrajectoryPIAEvalResult] = {}
    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []

    def record_result(
        *,
        dataset: str,
        phase: str,
        config_id: str,
        gamma_main: float,
        smooth_lambda: float,
        operator_mode: str,
    ) -> None:
        for seed in seeds:
            key = (dataset, int(seed))
            result = _evaluate_cached(
                cache=result_cache,
                state=states[key],
                basis=basis_map[key],
                seed=int(seed),
                model_cfg=model_cfgs[key],
                eval_common=eval_common,
                dataset=str(dataset),
                operator_mode=str(operator_mode),
                gamma_main=float(gamma_main),
                smooth_lambda=float(smooth_lambda),
            )
            seed_dir = os.path.join(out_root, f"{dataset}_seed{seed}")
            _write_json(
                os.path.join(seed_dir, f"{config_id}_result.json"),
                {
                    "dataset": result.dataset,
                    "seed": result.seed,
                    "phase": str(phase),
                    "config_id": str(config_id),
                    "operator_mode": result.operator_mode,
                    "gamma_main": float(gamma_main),
                    "smooth_lambda": float(smooth_lambda),
                    "train_metrics": result.train_metrics,
                    "val_metrics": result.val_metrics,
                    "test_metrics": result.test_metrics,
                    "best_epoch": result.best_epoch,
                    "diagnostics": result.diagnostics,
                    "operator_meta": result.operator_meta,
                    "meta": result.meta,
                    "history_rows": result.history_rows,
                },
            )
            config_rows.append(
                {
                    "dataset": str(dataset),
                    "phase": str(phase),
                    "config_id": str(config_id),
                    "seed": int(seed),
                    "operator_mode": str(operator_mode),
                    "gamma_main": float(gamma_main),
                    "smooth_lambda": float(smooth_lambda),
                    "axis_count": 1,
                    "basis_reused": True,
                    "window_len": int(states[key].window_len),
                    "hop_len": int(states[key].hop_len),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                }
            )
            per_seed_rows.append(
                {
                    "dataset": str(dataset),
                    "phase": str(phase),
                    "config_id": str(config_id),
                    "seed": int(seed),
                    "operator_mode": str(operator_mode),
                    "gamma_main": float(gamma_main),
                    "smooth_lambda": float(smooth_lambda),
                    "test_macro_f1": float(result.test_metrics["macro_f1"]),
                    "delta_vs_t2a_baseline": np.nan,
                    "delta_vs_best_t2a": float(result.test_metrics["macro_f1"]) - float(
                        t2a_best_ref.get((str(dataset), int(seed)), np.nan)
                    ),
                }
            )
            diagnostics_rows.append(
                {
                    "dataset": str(dataset),
                    "phase": str(phase),
                    "config_id": str(config_id),
                    "seed": int(seed),
                    **{k: float(v) for k, v in result.diagnostics.items()},
                }
            )

    # SCP1 baseline + Phase A
    record_result(
        dataset="selfregulationscp1",
        phase="baseline",
        config_id="scp1_baseline",
        gamma_main=0.10,
        smooth_lambda=0.00,
        operator_mode="baseline",
    )
    for gamma in gamma_sweep:
        record_result(
            dataset="selfregulationscp1",
            phase="phase_a_gamma",
            config_id=f"scp1_phaseA_gamma_g{_float_tag(gamma)}_l050",
            gamma_main=float(gamma),
            smooth_lambda=0.50,
            operator_mode="operator_smoothed",
        )
    for lam in smooth_sweep:
        mode = "operator_unsmoothed" if float(lam) <= 1e-12 else "operator_smoothed"
        record_result(
            dataset="selfregulationscp1",
            phase="phase_a_smooth",
            config_id=f"scp1_phaseA_smooth_g010_l{_float_tag(lam)}",
            gamma_main=0.10,
            smooth_lambda=float(lam),
            operator_mode=str(mode),
        )

    per_seed_df = pd.DataFrame(per_seed_rows)
    scp1_baseline_map = {
        int(row["seed"]): float(row["test_macro_f1"])
        for _, row in per_seed_df[
            (per_seed_df["dataset"] == "selfregulationscp1") & (per_seed_df["phase"] == "baseline")
        ].iterrows()
    }
    mask = per_seed_df["dataset"] == "selfregulationscp1"
    per_seed_df.loc[mask, "delta_vs_t2a_baseline"] = [
        float(per_seed_df.loc[idx, "test_macro_f1"]) - float(scp1_baseline_map.get(int(per_seed_df.loc[idx, "seed"]), np.nan))
        for idx in per_seed_df[mask].index
    ]

    # Phase A summary first
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    summary_rows: List[Dict[str, object]] = []

    def append_summary_for(df: pd.DataFrame, *, dataset: str, phase: str) -> None:
        if df.empty:
            return
        grouped = df.groupby("config_id", as_index=False)
        for config_id, g in grouped:
            diag_g = diagnostics_df[
                (diagnostics_df["dataset"] == dataset)
                & (diagnostics_df["phase"] == phase)
                & (diagnostics_df["config_id"] == config_id)
            ].copy()
            summary_rows.append(
                {
                    "dataset": str(dataset),
                    "phase": str(phase),
                    "config_id": str(config_id),
                    "gamma_main": float(g["gamma_main"].iloc[0]),
                    "smooth_lambda": float(g["smooth_lambda"].iloc[0]),
                    "macro_f1_mean": _mean_std(g["test_macro_f1"].tolist())[0],
                    "macro_f1_std": _mean_std(g["test_macro_f1"].tolist())[1],
                    "delta_vs_t2a_baseline_mean": _mean_std(g["delta_vs_t2a_baseline"].tolist())[0],
                    "delta_vs_best_t2a_mean": _mean_std(g["delta_vs_best_t2a"].tolist())[0],
                    "is_candidate": False,
                    "continuity_distortion_ratio_mean": _mean_std(diag_g["continuity_distortion_ratio"].tolist())[0] if not diag_g.empty else np.nan,
                }
            )

    append_summary_for(
        per_seed_df[(per_seed_df["dataset"] == "selfregulationscp1") & (per_seed_df["phase"] == "phase_a_gamma")],
        dataset="selfregulationscp1",
        phase="phase_a_gamma",
    )
    append_summary_for(
        per_seed_df[(per_seed_df["dataset"] == "selfregulationscp1") & (per_seed_df["phase"] == "phase_a_smooth")],
        dataset="selfregulationscp1",
        phase="phase_a_smooth",
    )

    # helper rows for phase-A sensitivity detection, per-seed
    phase_a_gamma_seed_df = per_seed_df[
        (per_seed_df["dataset"] == "selfregulationscp1") & (per_seed_df["phase"] == "phase_a_gamma")
    ].copy()
    phase_a_gamma_seed_df["phase"] = "phase_a_gamma_per_seed"
    phase_a_smooth_seed_df = per_seed_df[
        (per_seed_df["dataset"] == "selfregulationscp1") & (per_seed_df["phase"] == "phase_a_smooth")
    ].copy()
    phase_a_smooth_seed_df["phase"] = "phase_a_smooth_per_seed"
    detect_df = pd.concat([pd.DataFrame(summary_rows), phase_a_gamma_seed_df, phase_a_smooth_seed_df], ignore_index=True, sort=False)
    phase_b_mode, sensitivity_meta = _detect_phase_b_mode(detect_df)

    # Phase B if needed
    phase_b_configs: List[Tuple[float, float, str]] = []
    if phase_b_mode == "gamma":
        phase_b_configs = [(0.06, 0.25, "operator_smoothed"), (0.06, 0.50, "operator_smoothed"), (0.08, 0.25, "operator_smoothed"), (0.08, 0.50, "operator_smoothed"), (0.10, 0.25, "operator_smoothed"), (0.10, 0.50, "operator_smoothed")]
    elif phase_b_mode == "smooth":
        phase_b_configs = [(0.08, 0.00, "operator_unsmoothed"), (0.08, 0.25, "operator_smoothed"), (0.08, 0.50, "operator_smoothed"), (0.10, 0.00, "operator_unsmoothed"), (0.10, 0.25, "operator_smoothed"), (0.10, 0.50, "operator_smoothed")]
    elif phase_b_mode == "both":
        phase_b_configs = [
            (0.06, 0.00, "operator_unsmoothed"), (0.06, 0.25, "operator_smoothed"), (0.06, 0.50, "operator_smoothed"),
            (0.08, 0.00, "operator_unsmoothed"), (0.08, 0.25, "operator_smoothed"), (0.08, 0.50, "operator_smoothed"),
            (0.10, 0.00, "operator_unsmoothed"), (0.10, 0.25, "operator_smoothed"), (0.10, 0.50, "operator_smoothed"),
        ]

    for gamma, lam, mode in phase_b_configs:
        record_result(
            dataset="selfregulationscp1",
            phase="phase_b",
            config_id=f"scp1_phaseB_g{_float_tag(gamma)}_l{_float_tag(lam)}",
            gamma_main=float(gamma),
            smooth_lambda=float(lam),
            operator_mode=str(mode),
        )

    # recompute per-seed with SCP1 baseline deltas for newly added rows
    per_seed_df = pd.DataFrame(per_seed_rows)
    mask = per_seed_df["dataset"] == "selfregulationscp1"
    per_seed_df.loc[mask, "delta_vs_t2a_baseline"] = [
        float(per_seed_df.loc[idx, "test_macro_f1"]) - float(scp1_baseline_map.get(int(per_seed_df.loc[idx, "seed"]), np.nan))
        for idx in per_seed_df[mask].index
    ]

    # choose SCP1 candidate
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    summary_rows = []
    for dataset in ["selfregulationscp1"]:
        for phase in ["phase_a_gamma", "phase_a_smooth", "phase_b"]:
            df = per_seed_df[(per_seed_df["dataset"] == dataset) & (per_seed_df["phase"] == phase)].copy()
            if df.empty:
                continue
            grouped = df.groupby("config_id", as_index=False)
            for config_id, g in grouped:
                diag_g = diagnostics_df[
                    (diagnostics_df["dataset"] == dataset)
                    & (diagnostics_df["phase"] == phase)
                    & (diagnostics_df["config_id"] == config_id)
                ].copy()
                summary_rows.append(
                    {
                        "dataset": str(dataset),
                        "phase": str(phase),
                        "config_id": str(config_id),
                        "gamma_main": float(g["gamma_main"].iloc[0]),
                        "smooth_lambda": float(g["smooth_lambda"].iloc[0]),
                        "macro_f1_mean": _mean_std(g["test_macro_f1"].tolist())[0],
                        "macro_f1_std": _mean_std(g["test_macro_f1"].tolist())[1],
                        "delta_vs_t2a_baseline_mean": _mean_std(g["delta_vs_t2a_baseline"].tolist())[0],
                        "delta_vs_best_t2a_mean": _mean_std(g["delta_vs_best_t2a"].tolist())[0],
                        "is_candidate": False,
                        "continuity_distortion_ratio_mean": _mean_std(diag_g["continuity_distortion_ratio"].tolist())[0] if not diag_g.empty else np.nan,
                    }
                )
    summary_df = pd.DataFrame(summary_rows)
    candidate = _pick_single_candidate(summary_df)
    summary_df.loc[
        (summary_df["dataset"] == "selfregulationscp1") & (summary_df["config_id"] == candidate["config_id"]),
        "is_candidate",
    ] = True

    # NATOPS anchors after candidate is chosen
    natops_best_cfg = t2a_best_cfg.get("natops", {"gamma_main": 0.10, "smooth_lambda": 0.50})
    natops_anchor_specs = [
        ("natops_baseline", 0.10, 0.00, "baseline"),
        (
            f"natops_formal_best_g{_float_tag(float(natops_best_cfg['gamma_main']))}_l{_float_tag(float(natops_best_cfg['smooth_lambda']))}",
            float(natops_best_cfg["gamma_main"]),
            float(natops_best_cfg["smooth_lambda"]),
            "operator_unsmoothed" if float(natops_best_cfg["smooth_lambda"]) <= 1e-12 else "operator_smoothed",
        ),
        (
            f"natops_candidate_g{_float_tag(float(candidate['gamma_main']))}_l{_float_tag(float(candidate['smooth_lambda']))}",
            float(candidate["gamma_main"]),
            float(candidate["smooth_lambda"]),
            "operator_unsmoothed" if float(candidate["smooth_lambda"]) <= 1e-12 else "operator_smoothed",
        ),
    ]
    seen_natops = set()
    for config_id, gamma, lam, mode in natops_anchor_specs:
        spec_key = (round(float(gamma), 4), round(float(lam), 4), str(mode))
        if spec_key in seen_natops:
            continue
        seen_natops.add(spec_key)
        record_result(
            dataset="natops",
            phase="natops_anchor",
            config_id=str(config_id),
            gamma_main=float(gamma),
            smooth_lambda=float(lam),
            operator_mode=str(mode),
        )

    # Final metrics tables
    per_seed_df = pd.DataFrame(per_seed_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    # baseline maps for delta_vs_t2a_baseline now for both datasets
    baseline_rows = per_seed_df[per_seed_df["operator_mode"] == "baseline"].copy()
    closure_baseline_map = {
        (str(row["dataset"]), int(row["seed"])): float(row["test_macro_f1"])
        for _, row in baseline_rows.iterrows()
    }
    per_seed_df["delta_vs_t2a_baseline"] = [
        float(row["test_macro_f1"]) - float(closure_baseline_map.get((str(row["dataset"]), int(row["seed"])), np.nan))
        for _, row in per_seed_df.iterrows()
    ]

    summary_rows = []
    group_cols = ["dataset", "phase", "config_id"]
    for (dataset, phase, config_id), g in per_seed_df.groupby(group_cols):
        diag_g = diagnostics_df[
            (diagnostics_df["dataset"] == dataset)
            & (diagnostics_df["phase"] == phase)
            & (diagnostics_df["config_id"] == config_id)
        ].copy()
        summary_rows.append(
            {
                "dataset": str(dataset),
                "phase": str(phase),
                "config_id": str(config_id),
                "gamma_main": float(g["gamma_main"].iloc[0]),
                "smooth_lambda": float(g["smooth_lambda"].iloc[0]),
                "macro_f1_mean": _mean_std(g["test_macro_f1"].tolist())[0],
                "macro_f1_std": _mean_std(g["test_macro_f1"].tolist())[1],
                "macro_f1": _format_mean_std(g["test_macro_f1"].tolist()),
                "delta_vs_t2a_baseline_mean": _mean_std(g["delta_vs_t2a_baseline"].tolist())[0],
                "delta_vs_best_t2a_mean": _mean_std(g["delta_vs_best_t2a"].tolist())[0],
                "step_change_mean": _mean_std(diag_g["step_change_mean"].tolist())[0] if not diag_g.empty else np.nan,
                "local_curvature_proxy": _mean_std(diag_g["local_curvature_proxy"].tolist())[0] if not diag_g.empty else np.nan,
                "transition_separation_proxy": _mean_std(diag_g["transition_separation_proxy"].tolist())[0] if not diag_g.empty else np.nan,
                "continuity_distortion_ratio": _mean_std(diag_g["continuity_distortion_ratio"].tolist())[0] if not diag_g.empty else np.nan,
                "is_candidate": bool(str(dataset) == "selfregulationscp1" and str(config_id) == str(candidate["config_id"])),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    config_df = pd.DataFrame(config_rows)
    config_csv = os.path.join(out_root, "trajectory_pia_t2a_closure_config_table.csv")
    per_seed_csv = os.path.join(out_root, "trajectory_pia_t2a_closure_per_seed.csv")
    summary_csv = os.path.join(out_root, "trajectory_pia_t2a_closure_dataset_summary.csv")
    diagnostics_csv = os.path.join(out_root, "trajectory_pia_t2a_closure_diagnostics_summary.csv")
    phase_meta_json = os.path.join(out_root, "trajectory_pia_t2a_closure_phase_meta.json")
    config_df.to_csv(config_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)
    _write_json(
        phase_meta_json,
        {
            "phase_a_default_center": {"gamma_main": 0.10, "smooth_lambda": 0.50},
            "phase_b_mode": phase_b_mode or "none",
            "phase_a_sensitivity": sensitivity_meta,
            "scp1_candidate": candidate,
            "natops_formal_best_reference": natops_best_cfg,
        },
    )

    lines: List[str] = [
        "# Trajectory PIA T2a Closure Conclusion",
        "",
        "更新时间：2026-03-29",
        "",
        "本轮收口目标：固定 T2a 的其余一切，仅围绕 `gamma_main / smooth_lambda` 判清 SCP1 的主要限制因素。",
        "",
        f"- `Phase B mode`: `{phase_b_mode or 'none'}`",
        f"- `gamma_range`: `{sensitivity_meta['gamma_range']:.4f}`",
        f"- `smooth_range`: `{sensitivity_meta['smooth_range']:.4f}`",
        f"- `gamma_best_freq`: `{int(sensitivity_meta['gamma_best_freq'])}`",
        f"- `smooth_best_freq`: `{int(sensitivity_meta['smooth_best_freq'])}`",
        "",
        "## SCP1 Candidate",
        "",
        f"- `config_id`: `{candidate['config_id']}`",
        f"- `gamma_main`: `{float(candidate['gamma_main']):.2f}`",
        f"- `smooth_lambda`: `{float(candidate['smooth_lambda']):.2f}`",
        f"- `macro_f1_mean/std`: `{float(candidate['macro_f1_mean']):.4f} +/- {float(candidate['macro_f1_std']):.4f}`",
        "",
    ]

    scp1_phase_rows = summary_df[summary_df["dataset"] == "selfregulationscp1"].copy()
    natops_rows = summary_df[summary_df["dataset"] == "natops"].copy()
    scp1_best_row = scp1_phase_rows[scp1_phase_rows["config_id"] == candidate["config_id"]].iloc[0]
    lines.append("## SCP1 Judgment")
    lines.append("")
    lines.append(f"- `best candidate vs closure baseline`: `{float(scp1_best_row['delta_vs_t2a_baseline_mean']):+.4f}`")
    lines.append(f"- `best candidate vs prior T2a best`: `{float(scp1_best_row['delta_vs_best_t2a_mean']):+.4f}`")
    lines.append(
        f"- `continuity_distortion_ratio`: `{float(scp1_best_row['continuity_distortion_ratio']):.4f}`"
    )
    lines.append(
        f"- `transition_separation_proxy`: `{float(scp1_best_row['transition_separation_proxy']):.4f}`"
    )
    lines.append("")

    natops_candidate_rows = natops_rows[natops_rows["config_id"].str.startswith("natops_candidate_")].copy()
    natops_best_rows = natops_rows[natops_rows["config_id"].str.startswith("natops_formal_best_")].copy()
    if not natops_candidate_rows.empty and not natops_best_rows.empty:
        natops_candidate_row = natops_candidate_rows.iloc[0]
        natops_best_row = natops_best_rows.iloc[0]
        lines.append("## NATOPS Anchor Check")
        lines.append("")
        lines.append(f"- `candidate vs closure baseline`: `{float(natops_candidate_row['delta_vs_t2a_baseline_mean']):+.4f}`")
        lines.append(f"- `candidate vs current best T2a`: `{float(natops_candidate_row['delta_vs_best_t2a_mean']):+.4f}`")
        lines.append(f"- `current best T2a rerun`: `{str(natops_best_row['macro_f1'])}`")
        lines.append(f"- `candidate rerun`: `{str(natops_candidate_row['macro_f1'])}`")
        lines.append("")

    if phase_b_mode == "gamma":
        main_factor = "gamma_main"
    elif phase_b_mode == "smooth":
        main_factor = "smooth_lambda"
    elif phase_b_mode == "both":
        main_factor = "both"
    else:
        main_factor = "not_clear"

    lines.append("## Final Answers")
    lines.append("")
    lines.append(f"1. `SCP1` 的主要限制因素：`{main_factor}`")

    natops_not_damaged = False
    if not natops_candidate_rows.empty:
        natops_not_damaged = float(natops_candidate_rows.iloc[0]["delta_vs_best_t2a_mean"]) >= -0.0100
    candidate_improved = float(scp1_best_row["delta_vs_t2a_baseline_mean"]) > 1e-9
    closure_found = candidate_improved and natops_not_damaged
    lines.append(f"2. 是否存在不明显损伤 `NATOPS` 的 `SCP1` 收口点：`{'yes' if closure_found else 'not_yet'}`")

    worth_t2b = (not closure_found) and (main_factor == "not_clear" or main_factor == "both")
    lines.append(f"3. 当前是否值得进入 `T2b`：`{'yes' if worth_t2b else 'not_yet'}`")
    lines.append("")

    conclusion_md = os.path.join(out_root, "trajectory_pia_t2a_closure_conclusion.md")
    with open(conclusion_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[trajectory-pia-t2a-closure] wrote {config_csv}")
    print(f"[trajectory-pia-t2a-closure] wrote {per_seed_csv}")
    print(f"[trajectory-pia-t2a-closure] wrote {summary_csv}")
    print(f"[trajectory-pia-t2a-closure] wrote {diagnostics_csv}")
    print(f"[trajectory-pia-t2a-closure] wrote {phase_meta_json}")
    print(f"[trajectory-pia-t2a-closure] wrote {conclusion_md}")


if __name__ == "__main__":
    main()
