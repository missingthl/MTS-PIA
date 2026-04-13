#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from route_b_unified import (  # noqa: E402
    BridgeConfig,
    MiniRocketEvalConfig,
    PIACore,
    PIACoreConfig,
    PolicyAction,
    RepresentationConfig,
    TargetRoundState,
    apply_bridge,
    build_representation,
    evaluate_bridge,
)
from route_b_unified.types import BridgeResult  # noqa: E402


def _parse_csv_list(text: str) -> List[str]:
    out = [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("dataset list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _format_mean_std(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "0.0000 +/- 0.0000"
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}"


def _raw_reference_bridge_result(dataset: str, seed: int, rep_state) -> BridgeResult:
    return BridgeResult(
        dataset=str(dataset),
        seed=int(seed),
        variant="raw_only",
        round_index=0,
        train_trials=list(rep_state.train_trial_dicts),
        val_trials=list(rep_state.val_trial_dicts),
        test_trials=list(rep_state.test_trial_dicts),
        global_fidelity={
            "bridge_cov_match_error_mean": 0.0,
            "bridge_cov_to_orig_distance_logeuc_mean": 0.0,
            "energy_ratio_mean": 1.0,
            "cond_A_mean": 1.0,
            "raw_mean_shift_abs_mean": 0.0,
        },
        classwise_fidelity={
            "classwise_mean_shift_summary": {},
            "classwise_covariance_distortion_summary": {},
            "classwise_covariance_distortion_mean": 0.0,
        },
        margin_proxy={"delta_mean": 0.0},
        task_risk_comment="raw_reference",
    )


def _label(delta_vs_raw: float) -> str:
    if delta_vs_raw >= 0.002:
        return "positive"
    if delta_vs_raw <= -0.002:
        return "negative"
    return "flat"


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal PIA Core chain: representation -> PIA Core -> bridge -> raw MiniROCKET")
    p.add_argument("--datasets", type=str, default="selfregulationscp1,natops")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/route_b_pia_core_minimal_chain_20260327")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bridge-eps", type=float, default=1e-4)
    p.add_argument("--pia-r-dimension", type=int, default=3)
    p.add_argument("--pia-n-iters", type=int, default=3)
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--pia-activation", type=str, default="sine")
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--gamma-main", type=float, default=0.10)
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--prop-win-ratio", type=float, default=0.5)
    p.add_argument("--prop-hop-ratio", type=float, default=0.25)
    p.add_argument("--min-window-len-samples", type=int, default=16)
    p.add_argument("--min-hop-len-samples", type=int, default=8)
    p.add_argument("--nominal-cap-k", type=int, default=120)
    p.add_argument("--cap-sampling-policy", type=str, default="random")
    p.add_argument("--aggregation-mode", type=str, default="majority")
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--memmap-threshold-gb", type=float, default=1.0)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    eval_cfg = MiniRocketEvalConfig(
        out_root=str(args.out_root),
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        prop_win_ratio=float(args.prop_win_ratio),
        prop_hop_ratio=float(args.prop_hop_ratio),
        min_window_len_samples=int(args.min_window_len_samples),
        min_hop_len_samples=int(args.min_hop_len_samples),
        nominal_cap_k=int(args.nominal_cap_k),
        cap_sampling_policy=str(args.cap_sampling_policy),
        aggregation_mode=str(args.aggregation_mode),
        n_kernels=int(args.n_kernels),
        n_jobs=int(args.n_jobs),
        memmap_threshold_gb=float(args.memmap_threshold_gb),
    )
    bridge_cfg = BridgeConfig(eps=float(args.bridge_eps))

    per_seed_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            print(f"[pia-core-minimal][{dataset}][seed={seed}] representation_start", flush=True)
            rep_state = build_representation(
                RepresentationConfig(
                    dataset=str(dataset),
                    seed=int(seed),
                    val_fraction=float(args.val_fraction),
                    spd_eps=float(args.spd_eps),
                )
            )
            raw_ref = _raw_reference_bridge_result(dataset, seed, rep_state)
            raw_val = evaluate_bridge(
                raw_ref,
                eval_cfg,
                split_name="val",
                target_state=TargetRoundState(
                    round_index=0,
                    z_aug=np.zeros((0, rep_state.X_train.shape[1]), dtype=np.float32),
                    y_aug=np.zeros((0,), dtype=np.int64),
                    tid_aug=np.asarray([], dtype=object),
                    mech={"dir_profile": {}},
                    dir_maps={"margin_drop_median": {}, "flip_rate": {}, "intrusion": {}},
                    aug_meta={},
                    action=PolicyAction(
                        round_index=0,
                        selected_dir_ids=[],
                        direction_weights={},
                        step_sizes={},
                        direction_probs_vector=np.zeros((0,), dtype=np.float64),
                        gamma_vector=np.zeros((0,), dtype=np.float64),
                        entropy=0.0,
                        stop_flag=False,
                        stop_reason="raw_reference",
                    ),
                    direction_state={},
                    direction_budget_score={},
                ),
                round_gain_proxy=0.0,
            )
            raw_test = evaluate_bridge(
                raw_ref,
                eval_cfg,
                split_name="test",
                target_state=TargetRoundState(
                    round_index=0,
                    z_aug=np.zeros((0, rep_state.X_train.shape[1]), dtype=np.float32),
                    y_aug=np.zeros((0,), dtype=np.int64),
                    tid_aug=np.asarray([], dtype=object),
                    mech={"dir_profile": {}},
                    dir_maps={"margin_drop_median": {}, "flip_rate": {}, "intrusion": {}},
                    aug_meta={},
                    action=PolicyAction(
                        round_index=0,
                        selected_dir_ids=[],
                        direction_weights={},
                        step_sizes={},
                        direction_probs_vector=np.zeros((0,), dtype=np.float64),
                        gamma_vector=np.zeros((0,), dtype=np.float64),
                        entropy=0.0,
                        stop_flag=False,
                        stop_reason="raw_reference",
                    ),
                    direction_state={},
                    direction_budget_score={},
                ),
                round_gain_proxy=0.0,
            )

            pia_core = PIACore(
                PIACoreConfig(
                    r_dimension=int(args.pia_r_dimension),
                    n_iters=int(args.pia_n_iters),
                    C_repr=float(args.pia_c_repr),
                    activation=str(args.pia_activation),
                    bias_update_mode=str(args.pia_bias_update_mode),
                    seed=int(seed),
                )
            ).fit(rep_state.X_train)
            arts = pia_core.get_artifacts()
            gamma_vector = np.zeros((arts.directions.shape[0],), dtype=np.float64)
            gamma_vector[0] = float(args.gamma_main)
            op_result = pia_core.apply_affine(rep_state.X_train, gamma_vector=gamma_vector.tolist())

            target_state = TargetRoundState(
                round_index=1,
                z_aug=np.asarray(op_result.X_aug, dtype=np.float32),
                y_aug=np.asarray(rep_state.y_train, dtype=np.int64),
                tid_aug=np.asarray(rep_state.tid_train, dtype=object),
                mech={"dir_profile": {"worst_dir_id": 0, "dir_profile_summary": "pia_core_single_axis_affine"}},
                dir_maps={"margin_drop_median": {}, "flip_rate": {}, "intrusion": {}},
                aug_meta=dict(op_result.meta),
                action=PolicyAction(
                    round_index=1,
                    selected_dir_ids=[0],
                    direction_weights={0: 1.0},
                    step_sizes={0: float(args.gamma_main)},
                    direction_probs_vector=np.asarray([1.0] + [0.0] * max(0, arts.directions.shape[0] - 1), dtype=np.float64),
                    gamma_vector=np.asarray(gamma_vector, dtype=np.float64),
                    entropy=0.0,
                    stop_flag=False,
                    stop_reason="single_main_template_affine",
                ),
                direction_state={0: "single_axis_affine_active"},
                direction_budget_score={0: 1.0},
            )
            bridge_result = apply_bridge(rep_state, target_state, bridge_cfg, variant="pia_core_single_axis")
            pia_val = evaluate_bridge(
                bridge_result,
                eval_cfg,
                split_name="val",
                target_state=target_state,
                round_gain_proxy=0.0,
            )
            pia_test = evaluate_bridge(
                bridge_result,
                eval_cfg,
                split_name="test",
                target_state=target_state,
                round_gain_proxy=0.0,
            )

            delta_vs_raw = float(pia_test.macro_f1) - float(raw_test.macro_f1)
            row = {
                "dataset": str(dataset),
                "seed": int(seed),
                "protocol_type": str(rep_state.meta.get("protocol_type", "")),
                "raw_val_acc": float(raw_val.acc),
                "raw_val_macro_f1": float(raw_val.macro_f1),
                "raw_test_acc": float(raw_test.acc),
                "raw_test_macro_f1": float(raw_test.macro_f1),
                "pia_core_val_acc": float(pia_val.acc),
                "pia_core_val_macro_f1": float(pia_val.macro_f1),
                "pia_core_test_acc": float(pia_test.acc),
                "pia_core_test_macro_f1": float(pia_test.macro_f1),
                "delta_vs_raw": delta_vs_raw,
                "selected_axis_ids": json.dumps([0], ensure_ascii=False),
                "gamma_vector": json.dumps([float(v) for v in gamma_vector.tolist()], ensure_ascii=False),
                "label": _label(delta_vs_raw),
                "operator_semantics": str(op_result.meta["operator_semantics"]),
                "bridge_task_risk_comment": str(bridge_result.task_risk_comment),
            }
            per_seed_rows.append(row)

            run_dir = os.path.join(args.out_root, str(dataset), f"seed{seed}")
            _ensure_dir(run_dir)
            _write_json(
                os.path.join(run_dir, "pia_core_artifacts.json"),
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "protocol_type": str(rep_state.meta.get("protocol_type", "")),
                    "W_shape": list(np.asarray(arts.W).shape),
                    "b_shape": list(np.asarray(arts.b).shape),
                    "recon_err": list(arts.recon_err),
                    "operator_meta": dict(op_result.meta),
                    "bridge_global_fidelity": dict(bridge_result.global_fidelity),
                    "bridge_classwise_fidelity": dict(bridge_result.classwise_fidelity),
                    "bridge_margin_proxy": dict(bridge_result.margin_proxy),
                    "bridge_task_risk_comment": str(bridge_result.task_risk_comment),
                    "metrics": row,
                },
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(args.out_root, "pia_core_minimal_chain_per_seed.csv")
    per_seed_df.to_csv(per_seed_path, index=False)

    summary_rows: List[Dict[str, object]] = []
    for dataset in sorted(per_seed_df["dataset"].unique().tolist()):
        df_ds = per_seed_df.loc[per_seed_df["dataset"] == dataset].copy()
        best_label = "flat"
        mean_delta = float(df_ds["delta_vs_raw"].mean())
        if mean_delta >= 0.002:
            best_label = "positive"
        elif mean_delta <= -0.002:
            best_label = "negative"
        summary_rows.append(
            {
                "dataset": dataset,
                "protocol_type": str(df_ds["protocol_type"].iloc[0]),
                "raw_test_acc": _format_mean_std(df_ds["raw_test_acc"].tolist()),
                "raw_test_macro_f1": _format_mean_std(df_ds["raw_test_macro_f1"].tolist()),
                "pia_core_test_acc": _format_mean_std(df_ds["pia_core_test_acc"].tolist()),
                "pia_core_test_macro_f1": _format_mean_std(df_ds["pia_core_test_macro_f1"].tolist()),
                "delta_vs_raw_mean": mean_delta,
                "label": best_label,
                "note": "single_main_template_affine_gamma=[0.1,0,...]",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_root, "pia_core_minimal_chain_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[pia-core-minimal] wrote {per_seed_path}", flush=True)
    print(f"[pia-core-minimal] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
