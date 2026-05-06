from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from utils.evaluators import TauScheduler

from .augment_builders import (
    _attach_feedback_scores_to_aug_out,
    _build_act_realized_augmentations,
    _build_rc4_fused_aug_out,
    _build_rc4_multiz_fused_aug_out,
    _build_zpia_template_pool_aug_out,
    _summarize_osf_audit_rows,
    _summarize_spectral_audit_rows,
)
from .diagnostics import (
    run_analysis_probe as _run_analysis_probe,
    summarize_multitemplate_audit_rows as _summarize_multitemplate_audit_rows,
)
from .state import TrialRecord
from .training import fit_host_model as _fit_host_model
from .training import fit_host_model_weighted_aug_ce as _fit_host_model_weighted_aug_ce

def _run_act_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    aug_out = _build_act_realized_augmentations(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
    )

    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    if aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=aug_out["tid_aug"],
        X_aug=aug_out["X_aug_raw"],
        tid_to_rec=aug_out["tid_to_rec"],
    )

    print(f"Fitting ACT model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "gamma_requested_mean": aug_out.get("gamma_requested_mean", 0.0),
        "gamma_used_mean": aug_out.get("gamma_used_mean", 0.0),
        "gamma_zero_rate": aug_out.get("gamma_zero_rate", 0.0),
        "safe_clip_rate": aug_out.get("safe_clip_rate", 0.0),
        "selection_stage": aug_out.get("selection_stage", "response_only"),
        "selector_name": aug_out.get("selector_name", str(getattr(args, "template_selection", ""))),
        "feasible_rate": aug_out.get("feasible_rate", 1.0),
        "selector_accept_rate": aug_out.get("selector_accept_rate", 1.0),
        "pre_filter_reject_count": aug_out.get("pre_filter_reject_count", 0),
        "post_bridge_reject_count": aug_out.get("post_bridge_reject_count", 0),
        "reject_reason_zero_gamma": aug_out.get("reject_reason_zero_gamma", 0),
        "reject_reason_safe_radius": aug_out.get("reject_reason_safe_radius", 0),
        "reject_reason_bridge_fail": aug_out.get("reject_reason_bridge_fail", 0),
        "reject_reason_transport_error": aug_out.get("reject_reason_transport_error", 0),
        "relevance_score_mean": aug_out.get("relevance_score_mean", np.nan),
        "safe_balance_score_mean": aug_out.get("safe_balance_score_mean", np.nan),
        "fidelity_score_mean": aug_out.get("fidelity_score_mean", np.nan),
        "variety_score_mean": aug_out.get("variety_score_mean", np.nan),
        "fv_score_mean": aug_out.get("fv_score_mean", np.nan),
        "z_displacement_norm_mean": aug_out.get("z_displacement_norm_mean", 0.0),
        "template_response_top1_mean": aug_out.get("template_response_top1_mean", np.nan),
        "template_response_top5_mean": aug_out.get("template_response_top5_mean", np.nan),
        "template_response_gap_top1_top5_mean": aug_out.get("template_response_gap_top1_top5_mean", np.nan),
        "template_response_entropy_mean": aug_out.get("template_response_entropy_mean", np.nan),
        "pre_safe_displacement_norm_mean": aug_out.get("pre_safe_displacement_norm_mean", np.nan),
        "post_safe_displacement_norm_mean": aug_out.get("post_safe_displacement_norm_mean", np.nan),
        "gamma_used_ratio_mean": aug_out.get("gamma_used_ratio_mean", np.nan),
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "eta_safe": aug_out.get("eta_safe", None),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]]) if aug_out["aug_trials"] else None,
        },
    }


def _run_act_zpia_template_pool_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    algo_label: str,
    top1_only: bool,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    aug_out = _build_zpia_template_pool_aug_out(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_label=algo_label,
        top1_only=top1_only,
    )

    if aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=aug_out["tid_aug"],
        X_aug=aug_out["X_aug_raw"],
        tid_to_rec=aug_out["tid_to_rec"],
    )

    print(f"Fitting {algo_label} core model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "gamma_requested_mean": aug_out.get("gamma_requested_mean", 0.0),
        "gamma_used_mean": aug_out.get("gamma_used_mean", 0.0),
        "gamma_zero_rate": aug_out.get("gamma_zero_rate", 0.0),
        "safe_clip_rate": aug_out.get("safe_clip_rate", 0.0),
        "selection_stage": aug_out.get("selection_stage", "response_only"),
        "selector_name": aug_out.get("selector_name", str(getattr(args, "template_selection", ""))),
        "feasible_rate": aug_out.get("feasible_rate", 1.0),
        "selector_accept_rate": aug_out.get("selector_accept_rate", 1.0),
        "pre_filter_reject_count": aug_out.get("pre_filter_reject_count", 0),
        "post_bridge_reject_count": aug_out.get("post_bridge_reject_count", 0),
        "reject_reason_zero_gamma": aug_out.get("reject_reason_zero_gamma", 0),
        "reject_reason_safe_radius": aug_out.get("reject_reason_safe_radius", 0),
        "reject_reason_bridge_fail": aug_out.get("reject_reason_bridge_fail", 0),
        "reject_reason_transport_error": aug_out.get("reject_reason_transport_error", 0),
        "relevance_score_mean": aug_out.get("relevance_score_mean", np.nan),
        "safe_balance_score_mean": aug_out.get("safe_balance_score_mean", np.nan),
        "fidelity_score_mean": aug_out.get("fidelity_score_mean", np.nan),
        "variety_score_mean": aug_out.get("variety_score_mean", np.nan),
        "fv_score_mean": aug_out.get("fv_score_mean", np.nan),
        "z_displacement_norm_mean": aug_out.get("z_displacement_norm_mean", 0.0),
        "template_response_top1_mean": aug_out.get("template_response_top1_mean", np.nan),
        "template_response_top5_mean": aug_out.get("template_response_top5_mean", np.nan),
        "template_response_gap_top1_top5_mean": aug_out.get("template_response_gap_top1_top5_mean", np.nan),
        "template_response_entropy_mean": aug_out.get("template_response_entropy_mean", np.nan),
        "pre_safe_displacement_norm_mean": aug_out.get("pre_safe_displacement_norm_mean", np.nan),
        "post_safe_displacement_norm_mean": aug_out.get("post_safe_displacement_norm_mean", np.nan),
        "gamma_used_ratio_mean": aug_out.get("gamma_used_ratio_mean", np.nan),
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "effective_k_zpia": aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "eta_safe": aug_out.get("eta_safe", None),
        **_summarize_multitemplate_audit_rows(aug_out.get("audit_rows", [])),
        "multi_template_pairs": int(aug_out.get("multi_template_pairs", 0)),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]])
            if aug_out["aug_trials"]
            else None,
        },
    }


def _run_act_rc4_multiz_fused_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    aug_out = _build_rc4_multiz_fused_aug_out(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        model_obj=res_base.get("model_obj"),
        batch_size=batch_size,
    )

    if aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=aug_out["tid_aug"],
        X_aug=aug_out["X_aug_raw"],
        tid_to_rec=aug_out["tid_to_rec"],
    )

    print(f"Fitting RC4-MultiZ ACT model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "gamma_requested_mean": aug_out.get("gamma_requested_mean", 0.0),
        "gamma_used_mean": aug_out.get("gamma_used_mean", 0.0),
        "gamma_zero_rate": aug_out.get("gamma_zero_rate", 0.0),
        "safe_clip_rate": aug_out.get("safe_clip_rate", 0.0),
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "effective_k_lraes": aug_out.get("effective_k_lraes", 0),
        "effective_k_zpia": aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "eta_safe": aug_out.get("eta_safe", None),
        **_summarize_multitemplate_audit_rows(aug_out.get("audit_rows", [])),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]])
            if aug_out["aug_trials"]
            else None,
        },
    }


def _run_mba_feedback_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=True,
        loader_seed=seed,
    )

    aug_out = _build_act_realized_augmentations(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
    )
    score_out = _attach_feedback_scores_to_aug_out(
        aug_out=aug_out,
        model_obj=res_base.get("model_obj"),
        device=args.device,
        batch_size=batch_size,
        feedback_margin_temperature=args.feedback_margin_temperature,
    )
    margins = score_out["margins"]
    weights = score_out["weights"]

    # Build TauScheduler if requested
    _tau_sched: Optional[TauScheduler] = None
    if getattr(args, "aug_weight_mode", "sigmoid") != "sigmoid" or getattr(args, "tau_max", None) is not None:
        _tau_sched = TauScheduler(
            total_epochs=epochs,
            tau_max=getattr(args, "tau_max", 2.0),
            tau_min=getattr(args, "tau_min", 0.1),
            warmup_ratio=getattr(args, "tau_warmup_ratio", 0.3),
        )

    print(f"Fitting MBA feedback model ({len(y_train)} orig + {len(aug_out['y_aug_np']) if aug_out['y_aug_np'] is not None else 0} aug stream)...")
    res_act = _fit_host_model_weighted_aug_ce(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_aug=aug_out["X_aug_raw"] if aug_out["aug_dataset"] is None else None,
        y_aug=aug_out["y_aug_np"] if aug_out["aug_dataset"] is None else None,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        loader_seed=seed,
        aug_dataset=aug_out["aug_dataset"],
        tau_scheduler=_tau_sched,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "audit_rows": aug_out.get("audit_rows", []),
        "eta_safe": aug_out.get("eta_safe", None),
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "host_geom_cosine_mean": 0.0,
        "host_conflict_rate": 0.0,
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "feedback_weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        "feedback_weight_std": float(np.std(weights)) if weights.size else 0.0,
        "last_aug_margin_mean": float(np.mean(margins)) if margins.size else 0.0,
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]]) if aug_out["aug_trials"] else None,
        },
}


def _run_act_fused_core_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    fusion_mode: str,
    algo_label: str,
) -> Dict[str, object]:
    print("Fitting baseline...")
    res_base = _fit_host_model(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=args.theory_diagnostics,
        loader_seed=seed,
    )

    fused_aug_out = _build_rc4_fused_aug_out(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        model_obj=res_base.get("model_obj"),
        batch_size=batch_size,
        include_feedback_scores=False,
        algo_label=algo_label,
        fusion_mode=fusion_mode,
    )

    if fused_aug_out["aug_trials"]:
        X_mix = np.concatenate([X_train_raw, fused_aug_out["X_aug_raw"]], axis=0)
        y_mix = np.concatenate([y_train, fused_aug_out["y_aug_np"]], axis=0)
    else:
        X_mix = X_train_raw
        y_mix = y_train

    alignment_metrics = _run_analysis_probe(
        args=args,
        model_obj=res_base.get("model_obj"),
        tid_aug=fused_aug_out["tid_aug"],
        X_aug=fused_aug_out["X_aug_raw"],
        tid_to_rec=fused_aug_out["tid_to_rec"],
    )

    print(f"Fitting {algo_label} core model ({len(X_mix)} samples)...")
    res_act = _fit_host_model(
        args=args,
        X_tr=X_mix,
        y_tr=y_mix,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        return_model_obj=False,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": fused_aug_out["avg_bridge"],
        "safe_radius_ratio_mean": fused_aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": fused_aug_out["manifold_margin_mean"],
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": fused_aug_out["candidate_total_count"],
        "aug_total_count": fused_aug_out["aug_total_count"],
        "effective_k": fused_aug_out["effective_k"],
        "effective_k_lraes": fused_aug_out.get("effective_k_lraes", 0),
        "effective_k_zpia": fused_aug_out.get("effective_k_zpia", 0),
        "direction_bank_meta": fused_aug_out.get("direction_bank_meta", {}),
        "audit_rows": fused_aug_out.get("audit_rows", []),
        "eta_safe": fused_aug_out.get("eta_safe", None),
        **_summarize_osf_audit_rows(fused_aug_out.get("audit_rows", [])),
        **(_summarize_spectral_audit_rows(fused_aug_out.get("audit_rows", [])) if fusion_mode == "spectral_osf" else {}),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": fused_aug_out["z_aug"],
            "y_aug": fused_aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in fused_aug_out["aug_trials"][:20]])
            if fused_aug_out["aug_trials"]
            else None,
        },
    }


def _run_act_rc4_fused_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_z: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    return _run_act_fused_core_pipeline(
        args=args,
        seed=seed,
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        X_train_z=X_train_z,
        train_recs=train_recs,
        mean_log=mean_log,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        fusion_mode="rank1_osf",
        algo_label="rc4_fused",
    )
