from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from core.pia_audit import write_candidate_audit


def build_failure_result_row(*, dataset_name: str, seed: int, args, fail_reason: str) -> Dict[str, object]:
    return {
        "dataset": dataset_name,
        "seed": seed,
        "status": "failed",
        "fail_reason": str(fail_reason),
        "requested_k_dir": args.k_dir,
        "effective_k_dir": 0,
        "algo": args.algo,
        "model": args.model,
        "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
    }


def build_success_result_row(
    *,
    dataset_name: str,
    seed: int,
    args,
    pipeline_out: Dict[str, object],
    y_train: np.ndarray,
) -> Dict[str, object]:
    res_base = pipeline_out["res_base"]
    res_act = pipeline_out["res_act"]
    avg_bridge = pipeline_out["avg_bridge"]
    gain = float(res_act["macro_f1"] - res_base["macro_f1"])
    summary = {
        "dataset": dataset_name,
        "seed": seed,
        "status": "success",
        "algo": args.algo,
        "model": args.model,
        "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
        "base_f1": float(res_base["macro_f1"]),
        "act_f1": float(res_act["macro_f1"]),
        "gain": gain,
        "f1_gain_pct": gain / (float(res_base["macro_f1"]) + 1e-7) * 100.0,
        "base_stop_epoch": int(res_base.get("stop_epoch", 0)),
        "act_stop_epoch": int(res_act.get("stop_epoch", 0)),
        "base_best_val_f1": float(res_base.get("best_val_f1", 0.0)),
        "act_best_val_f1": float(res_act.get("best_val_f1", 0.0)),
        "transport_error_fro_mean": float(avg_bridge.get("transport_error_fro", 0.0)),
        "transport_error_logeuc_mean": float(avg_bridge.get("transport_error_logeuc", 0.0)),
        "bridge_cond_A_mean": float(avg_bridge.get("bridge_cond_A", 0.0)),
        "metric_preservation_error_mean": float(avg_bridge.get("metric_preservation_error", 0.0)),
        "safe_radius_ratio_mean": float(pipeline_out.get("safe_radius_ratio_mean", 1.0)),
        "manifold_margin_mean": float(pipeline_out.get("manifold_margin_mean", 0.0)),
        "gamma_requested_mean": float(pipeline_out.get("gamma_requested_mean", 0.0)),
        "gamma_used_mean": float(pipeline_out.get("gamma_used_mean", 0.0)),
        "gamma_zero_rate": float(pipeline_out.get("gamma_zero_rate", 0.0)),
        "host_geom_cosine_mean": float(pipeline_out.get("host_geom_cosine_mean", 0.0)),
        "host_conflict_rate": float(pipeline_out.get("host_conflict_rate", 0.0)),
        "candidate_total_count": int(pipeline_out.get("candidate_total_count", 0)),
        "aug_total_count": int(pipeline_out.get("aug_total_count", 0)),
        "requested_k_dir": int(args.k_dir),
        "effective_k_dir": int(pipeline_out.get("effective_k", 0)),
        "safe_clip_rate": float(pipeline_out.get("safe_clip_rate", 0.0)),
        "template_usage_entropy": float(pipeline_out.get("template_usage_entropy", 0.0)),
        "top_template_concentration": float(pipeline_out.get("top_template_concentration", 0.0)),
        "selection_stage": str(pipeline_out.get("selection_stage", "response_only")),
        "selector_name": str(pipeline_out.get("selector_name", getattr(args, "template_selection", ""))),
        "feasible_rate": float(pipeline_out.get("feasible_rate", 1.0)),
        "selector_accept_rate": float(pipeline_out.get("selector_accept_rate", 1.0)),
        "pre_filter_reject_count": int(pipeline_out.get("pre_filter_reject_count", 0)),
        "post_bridge_reject_count": int(pipeline_out.get("post_bridge_reject_count", 0)),
        "reject_reason_zero_gamma": int(pipeline_out.get("reject_reason_zero_gamma", 0)),
        "reject_reason_safe_radius": int(pipeline_out.get("reject_reason_safe_radius", 0)),
        "reject_reason_zero_direction": int(pipeline_out.get("reject_reason_zero_direction", 0)),
        "reject_reason_zero_margin": int(pipeline_out.get("reject_reason_zero_margin", 0)),
        "reject_reason_bridge_fail": int(pipeline_out.get("reject_reason_bridge_fail", 0)),
        "reject_reason_transport_error": int(pipeline_out.get("reject_reason_transport_error", 0)),
        "relevance_score_mean": float(pipeline_out.get("relevance_score_mean", np.nan)),
        "safe_balance_score_mean": float(pipeline_out.get("safe_balance_score_mean", np.nan)),
        "fidelity_score_mean": float(pipeline_out.get("fidelity_score_mean", np.nan)),
        "variety_score_mean": float(pipeline_out.get("variety_score_mean", np.nan)),
        "fv_score_mean": float(pipeline_out.get("fv_score_mean", np.nan)),
        "z_displacement_norm_mean": float(pipeline_out.get("z_displacement_norm_mean", 0.0)),
        "template_response_top1_mean": float(pipeline_out.get("template_response_top1_mean", np.nan)),
        "template_response_top5_mean": float(pipeline_out.get("template_response_top5_mean", np.nan)),
        "template_response_gap_top1_top5_mean": float(
            pipeline_out.get("template_response_gap_top1_top5_mean", np.nan)
        ),
        "template_response_entropy_mean": float(pipeline_out.get("template_response_entropy_mean", np.nan)),
        "pre_safe_displacement_norm_mean": float(pipeline_out.get("pre_safe_displacement_norm_mean", np.nan)),
        "post_safe_displacement_norm_mean": float(pipeline_out.get("post_safe_displacement_norm_mean", np.nan)),
        "gamma_used_ratio_mean": float(pipeline_out.get("gamma_used_ratio_mean", np.nan)),
    }
    direction_meta = dict(pipeline_out.get("direction_bank_meta", {}))
    summary["direction_bank_source"] = str(direction_meta.get("bank_source", args.algo))
    if direction_meta.get("bank_source") == "zpia_telm2" or args.algo == "zpia":
        summary.update(
            {
                "zpia_z_dim": int(direction_meta.get("z_dim", 0)),
                "zpia_n_train": int(direction_meta.get("n_train", 0)),
                "zpia_n_train_lt_z_dim": bool(direction_meta.get("n_train_lt_z_dim", False)),
                "zpia_row_norm_min": float(direction_meta.get("row_norm_min", 0.0)),
                "zpia_row_norm_max": float(direction_meta.get("row_norm_max", 0.0)),
                "zpia_row_norm_mean": float(direction_meta.get("row_norm_mean", 0.0)),
                "zpia_fallback_row_count": int(direction_meta.get("fallback_row_count", 0)),
                "telm2_recon_last": float(direction_meta.get("telm2_recon_last", 0.0)),
                "telm2_recon_mean": float(direction_meta.get("telm2_recon_mean", 0.0)),
                "telm2_recon_std": float(direction_meta.get("telm2_recon_std", 0.0)),
                "telm2_n_iters": int(direction_meta.get("telm2_n_iters", args.telm2_n_iters)),
                "telm2_c_repr": float(direction_meta.get("telm2_c_repr", args.telm2_c_repr)),
                "telm2_activation": str(direction_meta.get("telm2_activation", args.telm2_activation)),
                "telm2_bias_update_mode": str(
                    direction_meta.get("telm2_bias_update_mode", args.telm2_bias_update_mode)
                ),
            }
        )
    if args.algo in {"zpia_top1_pool", "zpia_multidir_pool", "rc4_multiz_fused"}:
        summary.update(
            {
                "utilization_mode": "core_concat",
                "core_training_mode": "concat_all",
                "aug_train_ratio": float(pipeline_out.get("aug_total_count", 0)) / max(float(len(y_train)), 1.0),
                "multi_template_pairs": int(pipeline_out.get("multi_template_pairs", 0)),
                "template_selection": str(direction_meta.get("template_selection", "anchor_top_abs_response")),
                "template_usage_entropy": float(pipeline_out.get("template_usage_entropy", 0.0)),
                "top_template_concentration": float(pipeline_out.get("top_template_concentration", 0.0)),
                "effective_k_dir_zpia": int(pipeline_out.get("effective_k_zpia", 0)),
                "u_perp_norm_ratio_mean": float(pipeline_out.get("u_perp_norm_ratio_mean", 0.0)),
                "u_perp_zero_rate": float(pipeline_out.get("u_perp_zero_rate", 0.0)),
            }
        )
        zpia_meta = dict(direction_meta.get("zpia_meta", {}))
        if zpia_meta:
            summary.update(
                {
                    "zpia_z_dim": int(zpia_meta.get("z_dim", 0)),
                    "zpia_n_train": int(zpia_meta.get("n_train", 0)),
                    "zpia_n_train_lt_z_dim": bool(zpia_meta.get("n_train_lt_z_dim", False)),
                    "zpia_row_norm_min": float(zpia_meta.get("row_norm_min", 0.0)),
                    "zpia_row_norm_max": float(zpia_meta.get("row_norm_max", 0.0)),
                    "zpia_row_norm_mean": float(zpia_meta.get("row_norm_mean", 0.0)),
                    "zpia_fallback_row_count": int(zpia_meta.get("fallback_row_count", 0)),
                    "telm2_recon_last": float(zpia_meta.get("telm2_recon_last", 0.0)),
                    "telm2_recon_mean": float(zpia_meta.get("telm2_recon_mean", 0.0)),
                    "telm2_recon_std": float(zpia_meta.get("telm2_recon_std", 0.0)),
                }
            )
        if args.algo == "rc4_multiz_fused":
            summary.update(
                {
                    "osf_alpha": float(args.osf_alpha),
                    "osf_beta": float(args.osf_beta),
                    "osf_kappa": float(args.osf_kappa),
                    "effective_k_dir_lraes": int(pipeline_out.get("effective_k_lraes", 0)),
                    "adaptive_engine_sources": ",".join([str(x) for x in direction_meta.get("engine_sources", [])]),
                    "osf_structure_overflow_rate": float(pipeline_out.get("osf_structure_overflow_rate", 0.0)),
                    "osf_alpha_eff_mean": float(pipeline_out.get("osf_alpha_eff_mean", 0.0)),
                    "osf_risk_scale_mean": float(pipeline_out.get("osf_risk_scale_mean", 0.0)),
                    "osf_risk_zero_perp_rate": float(pipeline_out.get("osf_risk_zero_perp_rate", 0.0)),
                    "osf_risk_clipped_rate": float(pipeline_out.get("osf_risk_clipped_rate", 0.0)),
                    "safe_clip_rate": float(pipeline_out.get("safe_clip_rate", 0.0)),
                    "selected_template_histogram": str(pipeline_out.get("template_counts", {})),
                }
            )
    return summary


def merge_candidate_audit_summary(
    *,
    summary: Dict[str, object],
    audit_rows: List[Dict[str, object]],
    args,
    dataset_name: str,
    seed: int,
    eta_safe,
) -> Dict[str, object]:
    if not audit_rows:
        return summary
    audit_summary = write_candidate_audit(
        audit_rows,
        out_dir=os.path.join(args.out_root, "candidate_audit"),
        dataset=dataset_name,
        seed=int(seed),
        method=str(getattr(args, "audit_method_label", "") or args.algo),
        activation_policy=str(getattr(args, "template_selection", args.algo)),
        eta_safe=eta_safe,
    )
    summary.update(audit_summary)
    return summary
