from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.pia_operator import pia_operator_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = [
    "atrialfibrillation",
    "ering",
    "handmovementdirection",
    "handwriting",
    "japanesevowels",
    "natops",
    "racketsports",
]

LOCKED_RESULT_ROOTS = (
    PROJECT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123",
    PROJECT_ROOT / "results" / "csta_external_baselines_phase2" / "resnet1d_s123",
)

DEFAULT_ARMS = [
    "no_aug",
    "raw_aug_jitter",
    "raw_aug_scaling",
    "raw_aug_timewarp",
    "raw_mixup",
    "dba_sameclass",
    "raw_smote_flatten_balanced",
    "random_cov_state",
    "pca_cov_state",
    "csta_top1_current",
    "csta_group_template_top",
]

PHASE2_ARMS = [
    "raw_aug_magnitude_warping",
    "raw_aug_window_warping",
    "raw_aug_window_slicing",
    "wdba_sameclass",
    "spawner_sameclass_style",
    "jobda_cleanroom",
    "rgw_sameclass",
    "dgw_sameclass",
]

PHASE3_ARMS = [
    "manifold_mixup",
    "timevae_classwise_optional",
    "diffusionts_classwise",
    "timevqvae_classwise",
    "random_cov_state",
    "pca_cov_state",
]

CSTA_RESULT_PASSTHROUGH_FIELDS = [
    "transport_error_fro_mean",
    "transport_error_logeuc_mean",
    "bridge_cond_A_mean",
    "metric_preservation_error_mean",
    "safe_radius_ratio_mean",
    "manifold_margin_mean",
    "gamma_requested_mean",
    "gamma_used_mean",
    "gamma_zero_rate",
    "host_geom_cosine_mean",
    "host_conflict_rate",
    "candidate_total_count",
    "aug_total_count",
    "requested_k_dir",
    "effective_k_dir",
    "safe_clip_rate",
    "template_usage_entropy",
    "selected_template_entropy",
    "top_template_concentration",
    "aug_valid_rate",
    "candidate_audit_rows",
    "candidate_audit_available",
    "candidate_accept_rate",
    "candidate_physics_ok",
    "candidate_audit_path",
    "z_displacement_norm_mean",
    "template_response_abs_mean",
    "template_response_top1_mean",
    "template_response_top5_mean",
    "template_response_gap_top1_top5_mean",
    "template_response_entropy_mean",
    "pre_safe_displacement_norm_mean",
    "post_safe_displacement_norm_mean",
    "gamma_used_ratio_mean",
    "gamma_requested_mean_audit",
    "gamma_used_mean_audit",
    "safe_radius_ratio_mean_audit",
    "safe_clip_rate_audit",
    "gamma_zero_rate_audit",
    "manifold_margin_mean_audit",
    "transport_error_logeuc_mean_audit",
    "template_response_top1_mean_audit",
    "template_response_top5_mean_audit",
    "template_response_gap_top1_top5_mean_audit",
    "template_response_entropy_mean_audit",
    "selected_template_response_abs_mean_audit",
    "gamma_used_ratio_mean_audit",
    "pre_safe_displacement_norm_mean_audit",
    "post_safe_displacement_norm_mean_audit",
    "template_usage_entropy_audit",
    "top_template_concentration_audit",
    "selection_stage",
    "selector_name",
    "feasible_rate",
    "selector_accept_rate",
    "pre_filter_reject_count",
    "post_bridge_reject_count",
    "reject_reason_zero_gamma",
    "reject_reason_safe_radius",
    "reject_reason_zero_direction",
    "reject_reason_zero_margin",
    "reject_reason_bridge_fail",
    "reject_reason_transport_error",
    "relevance_score_mean",
    "safe_balance_score_mean",
    "fidelity_score_mean",
    "variety_score_mean",
    "fv_score_mean",
    "feasible_rate_audit",
    "selector_accept_rate_audit",
    "relevance_score_mean_audit",
    "safe_balance_score_mean_audit",
    "fidelity_score_mean_audit",
    "variety_score_mean_audit",
    "fv_score_mean_audit",
    "gamma_used_gt_requested_count",
    "safe_radius_ratio_out_of_bounds_count",
    "direction_bank_source",
    "utilization_mode",
    "core_training_mode",
    "aug_train_ratio",
    "multi_template_pairs",
    "template_selection",
    "eta_safe",
    "zpia_z_dim",
    "zpia_n_train",
    "zpia_n_train_lt_z_dim",
    "zpia_row_norm_min",
    "zpia_row_norm_max",
    "zpia_row_norm_mean",
    "zpia_fallback_row_count",
    "telm2_recon_last",
    "telm2_recon_mean",
    "telm2_recon_std",
    "telm2_n_iters",
    "telm2_c_repr",
    "telm2_activation",
    "telm2_bias_update_mode",
]

RAW_AUG_METHODS = {
    "raw_aug_jitter",
    "raw_aug_scaling",
    "raw_aug_timewarp",
    "raw_aug_magnitude_warping",
    "raw_aug_window_warping",
    "raw_aug_window_slicing",
}

REF_METHODS = [
    "no_aug",
    "best_rawaug",
    "raw_mixup",
    "dba_sameclass",
    "raw_smote_flatten_balanced",
    "random_cov_state",
    "pca_cov_state",
    "csta_top1_current",
    "csta_group_template_top",
]


@dataclass(frozen=True)
class MethodInfo:
    source_space: str
    label_mode: str
    uses_external_library: bool
    library_name: str
    budget_matched: bool
    selection_rule: str


METHOD_INFO: Dict[str, MethodInfo] = {
    "no_aug": MethodInfo("none", "hard", False, "", True, "none"),
    "raw_aug_jitter": MethodInfo("raw_time", "hard", True, "tsaug", True, "repeat_train_anchors_addnoise"),
    "raw_aug_scaling": MethodInfo("raw_time", "hard", False, "", True, "repeat_train_anchors_amplitude_uniform"),
    "raw_aug_timewarp": MethodInfo("raw_time", "hard", True, "tsaug", True, "repeat_train_anchors_timewarp"),
    "raw_aug_magnitude_warping": MethodInfo("raw_time", "hard", False, "", True, "repeat_train_anchors_magnitude_warping"),
    "raw_aug_window_warping": MethodInfo("raw_time", "hard", False, "", True, "repeat_train_anchors_window_warping"),
    "raw_aug_window_slicing": MethodInfo("raw_time", "hard", False, "", True, "repeat_train_anchors_window_slicing"),
    "raw_mixup": MethodInfo("raw_mixup", "soft", False, "", True, "train_split_random_pair_beta"),
    "dba_sameclass": MethodInfo("dtw_barycenter", "hard", True, "tslearn", True, "same_class_dba"),
    "wdba_sameclass": MethodInfo("dtw_barycenter", "hard", True, "tslearn", True, "same_class_weighted_dba_anchor_dtw_softmax"),
    "spawner_sameclass_style": MethodInfo("dtw_pattern_mix", "hard", True, "tslearn", True, "spawner_style_same_class_dtw_aligned_average"),
    "jobda_cleanroom": MethodInfo("raw_time_tsw", "joint_hard", False, "", False, "jobda_cleanroom_tsw_joint_label"),
    "rgw_sameclass": MethodInfo("dtw_guided_warp", "hard", False, "", True, "random_guided_warp_same_class_dtw_cleanroom"),
    "dgw_sameclass": MethodInfo("dtw_guided_warp", "hard", False, "", True, "discriminative_guided_warp_same_class_dtw_cleanroom"),
    "raw_smote_flatten_balanced": MethodInfo("flattened_raw", "hard", True, "imbalanced-learn", False, "class_balancing_smote_auto"),
    "random_cov_state": MethodInfo("covariance_state", "hard", False, "", True, "random_unit_z_direction"),
    "pca_cov_state": MethodInfo("covariance_state", "hard", False, "", True, "pca_top_z_direction"),
    "csta_top1_current": MethodInfo("covariance_template", "hard", False, "", True, "anchor_top_response"),
    "csta_group_template_top": MethodInfo("covariance_template", "hard", False, "", True, "group_center_top_response"),
    "csta_topk_softmax_tau_0.05": MethodInfo("covariance_template", "hard", False, "", True, "softmax_tau_0.05_response"),
    "csta_topk_softmax_tau_0.10": MethodInfo("covariance_template", "hard", False, "", True, "softmax_tau_0.10_response"),
    "csta_topk_softmax_tau_0.20": MethodInfo("covariance_template", "hard", False, "", True, "softmax_tau_0.20_response"),
    "csta_topk_uniform_top5": MethodInfo("covariance_template", "hard", False, "", True, "uniform_top5_response"),
    "csta_template_random_within_bank": MethodInfo("covariance_template", "hard", False, "", True, "csta_bank_random"),
    "csta_fv_filter_top5": MethodInfo("covariance_template", "hard", False, "", True, "pre_bridge_fv_filter_top5"),
    "csta_fv_score_top5": MethodInfo("covariance_template", "hard", False, "", True, "pre_bridge_fv_score_top5"),
    "csta_random_feasible_selector": MethodInfo("covariance_template", "hard", False, "", True, "pre_bridge_random_feasible_control"),
    "csta_topk_uniform_top5_ao_fisher": MethodInfo("covariance_template", "hard", False, "", True, "ao_fisher_uniform_top5"),
    "csta_topk_uniform_top5_ao_contrastive": MethodInfo("covariance_template", "hard", False, "", True, "ao_contrastive_uniform_top5"),
    "manifold_mixup": MethodInfo("hidden_state", "soft", False, "", False, "resnet1d_hidden_state_beta_mixup"),
    "timevae_classwise_optional": MethodInfo("generative_model", "hard", False, "", False, "classwise_timevae_style_pytorch_cleanroom"),
    "diffusionts_classwise": MethodInfo("generative_model", "hard", True, "Diffusion-TS", True, "classwise_diffusionts_classifier_guidance"),
    "timevqvae_classwise": MethodInfo("generative_model", "hard", True, "TimeVQVAE", True, "classwise_timevqvae_maskgit"),
}


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def guard_locked_out_root(args) -> None:
    """Prevent smoke/probe runs from overwriting locked reference summaries."""
    out_root = resolved_path(args.out_root)
    locked = {resolved_path(path) for path in LOCKED_RESULT_ROOTS}
    if out_root not in locked:
        return
    if bool(getattr(args, "allow_locked_root_overwrite", False)):
        return
    locked_text = "\n".join(f"  - {path}" for path in sorted(str(path) for path in locked))
    raise RuntimeError(
        "Refusing to write to a locked external-baseline reference root.\n"
        f"Requested out_root: {out_root}\n"
        f"Locked roots:\n{locked_text}\n"
        "Use a smoke/local output root, or pass --allow-locked-root-overwrite "
        "only when intentionally regenerating locked references."
    )


def csta_policy_for_method(method: str) -> str:
    if method == "csta_top1_current":
        return "top1"
    if method == "csta_group_template_top":
        return "group_top"
    clean = method.replace("_ao_fisher", "").replace("_ao_contrastive", "")
    if clean.startswith("csta_topk_"):
        return clean.replace("csta_", "", 1)
    if clean == "csta_template_random_within_bank":
        return "random"
    if clean == "csta_fv_filter_top5":
        return "fv_filter_top5"
    if clean == "csta_fv_score_top5":
        return "fv_score_top5"
    if clean == "csta_random_feasible_selector":
        return "random_feasible_selector"
    return method


def clean_metric_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def extract_csta_extra_metrics(result_row: Dict[str, object], method: str) -> Dict[str, object]:
    extra = pia_operator_metadata(csta_policy_for_method(method))
    for field in CSTA_RESULT_PASSTHROUGH_FIELDS:
        if field in result_row:
            extra[field] = clean_metric_value(result_row[field])
    if "selected_template_entropy" not in extra and "template_usage_entropy" in extra:
        extra["selected_template_entropy"] = extra["template_usage_entropy"]
    return extra
