from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from route_b_unified.types import BridgeResult, EvaluatorPosterior, TargetRoundState
from scripts.raw_baselines.run_bridge_curriculum_pilot import _fit_raw_minirocket


@dataclass(frozen=True)
class MiniRocketEvalConfig:
    out_root: str
    window_sec: float = 4.0
    hop_sec: float = 1.0
    prop_win_ratio: float = 0.5
    prop_hop_ratio: float = 0.25
    min_window_len_samples: int = 16
    min_hop_len_samples: int = 8
    nominal_cap_k: int = 120
    cap_sampling_policy: str = "random"
    aggregation_mode: str = "majority"
    n_kernels: int = 10000
    n_jobs: int = 1
    memmap_threshold_gb: float = 1.0


def _to_args(cfg: MiniRocketEvalConfig) -> SimpleNamespace:
    return SimpleNamespace(
        out_root=str(cfg.out_root),
        window_sec=float(cfg.window_sec),
        hop_sec=float(cfg.hop_sec),
        prop_win_ratio=float(cfg.prop_win_ratio),
        prop_hop_ratio=float(cfg.prop_hop_ratio),
        min_window_len_samples=int(cfg.min_window_len_samples),
        min_hop_len_samples=int(cfg.min_hop_len_samples),
        nominal_cap_k=int(cfg.nominal_cap_k),
        cap_sampling_policy=str(cfg.cap_sampling_policy),
        aggregation_mode=str(cfg.aggregation_mode),
        n_kernels=int(cfg.n_kernels),
        n_jobs=int(cfg.n_jobs),
        memmap_threshold_gb=float(cfg.memmap_threshold_gb),
    )


def evaluate_bridge(
    bridge_result: BridgeResult,
    eval_cfg: MiniRocketEvalConfig,
    *,
    split_name: str,
    target_state: TargetRoundState,
    round_gain_proxy: float,
) -> EvaluatorPosterior:
    test_trials = bridge_result.val_trials if split_name == "val" else bridge_result.test_trials
    metrics, run_meta = _fit_raw_minirocket(
        dataset=str(bridge_result.dataset),
        train_trials=bridge_result.train_trials,
        test_trials=test_trials,
        seed=int(bridge_result.seed),
        args=_to_args(eval_cfg),
    )
    dir_summary = target_state.mech.get("dir_profile", {})
    return EvaluatorPosterior(
        dataset=str(bridge_result.dataset),
        seed=int(bridge_result.seed),
        variant=str(bridge_result.variant),
        round_index=int(bridge_result.round_index),
        split_name=str(split_name),
        acc=float(metrics["trial_acc"]),
        macro_f1=float(metrics["trial_macro_f1"]),
        metrics=dict(metrics) | {"run_meta": run_meta},
        round_gain_proxy=float(round_gain_proxy),
        direction_usage_entropy=float(target_state.action.entropy),
        worst_dir_id=dir_summary.get("worst_dir_id"),
        worst_dir_summary=str(dir_summary.get("dir_profile_summary", "n/a")),
        direction_metrics={
            "margin_drop_median": dict(target_state.dir_maps.get("margin_drop_median", {})),
            "flip_rate": dict(target_state.dir_maps.get("flip_rate", {})),
            "intrusion": dict(target_state.dir_maps.get("intrusion", {})),
        },
        selected_dir_ids=list(target_state.action.selected_dir_ids),
        selected_dir_weights=dict(target_state.action.direction_weights),
        classwise_distortion_summary={
            str(k): float(v)
            for k, v in bridge_result.classwise_fidelity.get("classwise_covariance_distortion_summary", {}).items()
        },
        inter_class_margin_proxy={str(k): float(v) for k, v in bridge_result.margin_proxy.items()},
        task_risk_comment=str(bridge_result.task_risk_comment),
        bridge_meta=dict(bridge_result.global_fidelity) | dict(bridge_result.classwise_fidelity),
    )
