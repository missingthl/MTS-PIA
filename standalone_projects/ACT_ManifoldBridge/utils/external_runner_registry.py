from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from core.csta.result_schema import CSTA_GENERATION_ENGINE_FIELDS

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
    "raw_aug_timewarp",
    "raw_mixup",
    "dba_sameclass",
    "wdba_sameclass",
    "rgw_sameclass",
    "dgw_sameclass",
    "csta_topk_uniform_top5",
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
    "timegan_classwise",
    "diffusionts_classwise",
    "timevqvae_classwise",
    "random_cov_state",
    "pca_cov_state",
]

CSTA_RESULT_PASSTHROUGH_FIELDS = [
    "transport_error_fro_mean",
    "transport_error_logeuc_mean",
    "bridge_cond_A_mean",
    "downstream_train_sec",
    "aug_cost_sec",
    "cov_state_compute_sec",
    "bridge_realization_sec",
    "peak_gpu_mem_mb",
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
    "bridge_success_rate",
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
    "direction_source",
    "operator_source",
    "ag_target_effective_rank",
    "ag_target_pairwise_cosine_mean",
    "ag_target_norm_mean",
    "ag_target_norm_std",
    "ag_head_pairwise_cosine_mean",
    "ag_head_effective_rank",
    "ag_head_usage_entropy",
    "ag_operator_train_mse_mean",
    "ag_operator_train_cosine_mean",
    "ag_pred_target_cosine_mean",
    "ag_tangent_available_rate",
    "ag_fallback_rate",
    "ag_pos_dist_mean",
    "ag_neg_centroid_dist_mean",
    "ag_pred_norm_mean",
    "ag_hidden_dim",
    "ag_ridge",
    "ag_activation",
    "effective_k_ag_heads",
    "ag_target_norm_mean_audit",
    "ag_pred_norm_mean_audit",
    "ag_pred_target_cosine_mean_audit",
    "ag_tangent_available_rate_audit",
    "ag_fallback_rate_audit",
    "cs_flow_train_mse_mean",
    "cs_flow_train_cosine_mean",
    "cs_flow_pred_target_cosine_mean",
    "cs_flow_target_dist_mean",
    "cs_flow_target_dist_std",
    "cs_flow_velocity_norm_mean",
    "cs_flow_velocity_norm_std",
    "cs_flow_fallback_rate",
    "cs_flow_target_effective_rank",
    "cs_flow_target_pairwise_cosine_mean",
    "cs_flow_velocity_effective_rank",
    "cs_flow_velocity_pairwise_cosine_mean",
    "unique_direction_ratio",
    "generated_direction_pairwise_cosine_mean",
    "effective_aug_multiplier",
    "cs_flow_hidden_width",
    "cs_flow_class_embedding_dim",
    "cs_flow_hidden_layers",
    "cs_flow_t_gen",
    "cs_flow_k_same",
    "cs_flow_epochs",
    "cs_flow_batch_size",
    "cs_flow_lr",
    "cs_flow_weight_decay",
    "cs_flow_target_dist_mean_audit",
    "cs_flow_velocity_norm_mean_audit",
    "cs_flow_pred_target_cosine_mean_audit",
    "cs_flow_fallback_rate_audit",
    "latent_train_cosine_loss_mean",
    "latent_train_mse_mean",
    "latent_train_pred_target_cosine_mean",
    "latent_pred_velocity_norm_mean",
    "latent_pred_velocity_norm_std",
    "latent_target_velocity_norm_mean",
    "latent_target_velocity_norm_std",
    "latent_target_dist_mean",
    "latent_target_dist_std",
    "latent_target_sampling_entropy",
    "latent_target_sampling_entropy_by_class_mean",
    "latent_target_sampling_entropy_by_class_min",
    "latent_fallback_rate",
    "latent_residual_effective_rank",
    "latent_residual_pairwise_cosine_mean",
    "latent_generated_direction_pairwise_cosine_mean",
    "latent_unique_direction_ratio",
    "latent_effective_aug_multiplier",
    "latent_pred_target_cosine_mean",
    "latent_hidden_width",
    "latent_class_embedding_dim",
    "latent_hidden_layers",
    "latent_lambda_cos",
    "latent_flow_epochs",
    "latent_flow_batch_size",
    "latent_flow_lr",
    "latent_flow_weight_decay",
    "latent_rbf_tau_floor",
    "latent_target_dist_mean_audit",
    "latent_target_sampling_prob_mean_audit",
    "latent_residual_norm_mean_audit",
    "latent_pred_velocity_norm_mean_audit",
    "latent_pred_target_cosine_mean_audit",
    "latent_fallback_rate_audit",
    "task_utility_mean",
    "task_utility_std",
    "task_margin_mean",
    "task_margin_std",
    "task_bad_margin_mass",
    "task_wrong_pred_mass",
    "task_sampling_entropy",
    "task_sampling_effective_support",
    "task_kl_task_vs_geo",
    "task_guidance_fallback_rate",
    "task_guidance_fallback_reason",
    "task_invalid_candidate_rate",
    "task_warmup_train_epochs",
    "task_warmup_train_loss_mean",
    "task_guidance_beta",
    "task_guidance_margin_min",
    "task_guidance_lambda_margin",
    "task_guidance_max_candidates",
    "task_utility_mean_audit",
    "task_margin_mean_audit",
    "lc_valid_candidate_rate",
    "lc_no_valid_fallback_rate",
    "lc_bad_margin_mass",
    "lc_wrong_pred_mass",
    "lc_sampling_entropy",
    "lc_sampling_effective_support",
    "lc_kl_lc_vs_geo",
    "lc_margin_mean",
    "lc_margin_std",
    "lc_margin_target_mean",
    "lc_weight_top1_mass",
    "lc_fallback_reason",
    "lc_beta",
    "lc_margin_floor",
    "lc_gamma_eps",
    "lc_warmup_epochs",
    "lc_max_candidates",
    "lc_warmup_train_loss_mean",
    "lc_utility_mean_audit",
    "lc_margin_mean_audit",
    "lc_margin_target_mean_audit",
    "lc_fallback_rate_audit",
    "spg_zhead_train_acc",
    "spg_zhead_train_loss_mean",
    "spg_grad_norm_mean",
    "spg_grad_norm_std",
    "spg_projected_grad_norm_mean",
    "spg_projected_grad_norm_std",
    "spg_projection_energy",
    "spg_projection_energy_std",
    "spg_direction_pairwise_cosine_mean",
    "spg_effective_aug_multiplier",
    "spg_support_rank",
    "spg_support_condition",
    "spg_projection_ridge",
    "spg_noise_sigma",
    "spg_zhead_epochs",
    "spg_zhead_hidden_dim",
    "spg_grad_norm_mean_audit",
    "spg_projected_grad_norm_mean_audit",
    "spg_projection_energy_mean_audit",
    "ecl_projection_energy_mean",
    "ecl_projection_energy_std",
    "ecl_alpha_mean",
    "ecl_alpha_std",
    "ecl_alignment_to_projected_gradient_mean",
    "ecl_direction_pairwise_cosine_mean",
    "ecl_effective_aug_multiplier",
    "ecl_support_rank",
    "ecl_support_noise_norm_mean",
    "ecl_support_noise_norm_std",
    "ecl_fallback_rate",
    "ecl_projection_energy_mean_audit",
    "ecl_alpha_mean_audit",
    "ecl_alignment_to_projected_gradient_mean_audit",
    "ecl_support_noise_norm_mean_audit",
    "ecl_fallback_rate_audit",
    "rn_ecl_projection_energy_mean",
    "rn_ecl_projection_energy_std",
    "rn_ecl_alpha_mean",
    "rn_ecl_alpha_std",
    "rn_ecl_direction_pairwise_cosine_mean",
    "rn_ecl_alignment_to_projected_gradient_mean",
    "rn_ecl_effective_aug_multiplier",
    "rn_ecl_support_rank",
    "rn_ecl_fallback_rate",
    "rn_ecl_projection_energy_mean_audit",
    "rn_ecl_alpha_mean_audit",
    "rn_ecl_alignment_to_projected_gradient_mean_audit",
    "rn_ecl_support_noise_norm_mean_audit",
    "rn_ecl_fallback_rate_audit",
    "gi_spg_operator_train_mse",
    "gi_spg_operator_train_cosine",
    "gi_spg_target_norm_mean",
    "gi_spg_target_norm_std",
    "gi_spg_pred_norm_mean",
    "gi_spg_pred_norm_std",
    "gi_spg_pred_target_cosine_mean",
    "gi_spg_projection_energy_mean",
    "gi_spg_projection_energy_std",
    "gi_spg_zhead_train_acc",
    "gi_spg_direction_pairwise_cosine_mean",
    "gi_spg_effective_aug_multiplier",
    "gi_spg_support_rank",
    "gi_spg_hidden_dim",
    "gi_spg_ridge",
    "gi_spg_activation",
    "spg_cfm_train_mse_mean",
    "spg_cfm_train_cosine_mean",
    "spg_cfm_train_pred_target_cosine_mean",
    "spg_cfm_generation_pred_target_cosine_mean",
    "spg_cfm_generated_direction_pairwise_cosine_mean",
    "spg_cfm_effective_aug_multiplier",
    "spg_cfm_alignment_to_spg_mean",
    "spg_cfm_steps",
    "spg_cfm_projection_energy_mean",
    "spg_cfm_projection_energy_std",
    "spg_cfm_condition_norm_mean",
    "spg_cfm_condition_norm_std",
    "spg_zhead_train_acc",
    "gamma_used_ratio_mean",
    "transport_error_logeuc_mean",
    "augmentation_build_time_sec",
    "spg_cfm_zhead_time_sec",
    "spg_cfm_condition_time_sec",
    "spg_cfm_train_time_sec",
    "spg_cfm_generation_time_sec",
    "generation_time_per_aug_sample_ms",
]

for _field in CSTA_GENERATION_ENGINE_FIELDS:
    if _field not in CSTA_RESULT_PASSTHROUGH_FIELDS:
        CSTA_RESULT_PASSTHROUGH_FIELDS.append(_field)
CSTA_RESULT_PASSTHROUGH_FIELDS = list(dict.fromkeys(CSTA_RESULT_PASSTHROUGH_FIELDS))

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
    "ag_target_direct": MethodInfo("covariance_state_operator", "hard", False, "", True, "ag_target_direct_debug"),
    "ag_pia_single": MethodInfo("covariance_state_operator", "hard", False, "", True, "ag_pia_single_operator"),
    "ag_pia_multihead5": MethodInfo("covariance_state_operator", "hard", False, "", True, "ag_pia_multihead_uniform"),
    "cs_flow_target_direct": MethodInfo("covariance_state_flow", "hard", False, "", True, "cs_flow_target_direct_probe"),
    "cs_flow_single_step": MethodInfo("covariance_state_flow", "hard", False, "", True, "cs_flow_single_step"),
    "latent_residual_direct": MethodInfo(
        "covariance_state_latent_residual",
        "hard",
        False,
        "",
        True,
        "latent_residual_direct_probe",
    ),
    "latent_residual_flow": MethodInfo(
        "covariance_state_latent_residual",
        "hard",
        False,
        "",
        True,
        "latent_residual_sampling",
    ),
    "task_guided_residual_direct": MethodInfo(
        "covariance_state_task_guided_latent_residual",
        "hard",
        False,
        "",
        True,
        "task_guided_residual_direct_probe",
    ),
    "task_guided_latent_residual_flow": MethodInfo(
        "covariance_state_task_guided_latent_residual",
        "hard",
        False,
        "",
        True,
        "task_reweighted_residual_sampling",
    ),
    "lc_residual_direct": MethodInfo(
        "covariance_state_label_consistent_latent_residual",
        "hard",
        False,
        "",
        True,
        "lc_residual_direct_probe",
    ),
    "lc_latent_residual_flow": MethodInfo(
        "covariance_state_label_consistent_latent_residual",
        "hard",
        False,
        "",
        True,
        "label_consistent_boundary_reweighted_sampling",
    ),
    "spg_pia_zhead": MethodInfo(
        "covariance_state_support_projected_gradient",
        "hard",
        False,
        "",
        True,
        "support_projected_gradient_zhead",
    ),
    "spg_pia_zhead_deterministic": MethodInfo(
        "covariance_state_support_projected_gradient",
        "hard",
        False,
        "",
        True,
        "support_projected_gradient_zhead_deterministic_probe",
    ),
    "ecl_spg_pia_zhead": MethodInfo(
        "covariance_state_energy_calibrated_langevin_spg",
        "hard",
        False,
        "",
        True,
        "energy_calibrated_langevin_support_projected_gradient",
    ),
    "ecl_spg_pia_zhead_deterministic": MethodInfo(
        "covariance_state_energy_calibrated_langevin_spg",
        "hard",
        False,
        "",
        True,
        "energy_calibrated_langevin_support_projected_gradient_deterministic_probe",
    ),
    "rn_ecl_spg_pia_zhead": MethodInfo(
        "covariance_state_rank_normalized_ecl_spg",
        "hard",
        False,
        "",
        True,
        "rank_normalized_energy_calibrated_langevin_support_projected_gradient",
    ),
    "rn_ecl_spg_pia_zhead_deterministic": MethodInfo(
        "covariance_state_rank_normalized_ecl_spg",
        "hard",
        False,
        "",
        True,
        "rank_normalized_energy_calibrated_langevin_support_projected_gradient_deterministic_probe",
    ),
    "gi_spg_pia_zhead": MethodInfo(
        "covariance_state_generalized_inverse_spg",
        "hard",
        False,
        "",
        True,
        "generalized_inverse_support_projected_gradient_elm",
    ),
    "spg_cfm_one_step": MethodInfo(
        "covariance_state_spg_conditioned_cfm",
        "hard",
        False,
        "",
        True,
        "spg_conditioned_conditional_flow_matching",
    ),
    "spg_cfm_k3": MethodInfo(
        "covariance_state_spg_conditioned_cfm",
        "hard",
        False,
        "",
        True,
        "spg_conditioned_conditional_flow_matching",
    ),
    "spg_cfm_film_one_step": MethodInfo(
        "covariance_state_spg_conditioned_cfm",
        "hard",
        False,
        "",
        True,
        "spg_conditioned_conditional_flow_matching",
    ),
    "spg_cfm_align_one_step": MethodInfo(
        "covariance_state_spg_conditioned_cfm",
        "hard",
        False,
        "",
        True,
        "spg_conditioned_conditional_flow_matching",
    ),
    "manifold_mixup": MethodInfo("hidden_state", "soft", False, "", False, "resnet1d_hidden_state_beta_mixup"),
    "timevae_classwise_optional": MethodInfo("generative_model", "hard", False, "", False, "classwise_timevae_style_pytorch_cleanroom"),
    "timegan_classwise": MethodInfo("generative_model", "hard", False, "", True, "classwise_timegan_style_pytorch_cleanroom"),
    "diffusionts_classwise": MethodInfo("generative_model", "hard", True, "Diffusion-TS", True, "classwise_diffusionts_classifier_guidance"),
    "timevqvae_classwise": MethodInfo("generative_model", "hard", True, "TimeVQVAE", True, "classwise_timevqvae_maskgit"),
}


E1_MAIN_METHODS = {
    "no_aug",
    "raw_aug_jitter",
    "raw_aug_timewarp",
    "raw_mixup",
    "timegan_classwise",
    "diffusionts_classwise",
    "dba_sameclass",
    "wdba_sameclass",
    "rgw_sameclass",
    "dgw_sameclass",
    "csta_topk_uniform_top5",
}

ACTIVE_CONTROL_METHODS = {
    "random_cov_state",
    "pca_cov_state",
    "csta_top1_current",
    "csta_template_random_within_bank",
}

FUTURE_BRANCH_METHODS = {
    "spg_cfm_one_step",
    "spg_cfm_align_one_step",
}

ACTIVE_METHODS = E1_MAIN_METHODS | ACTIVE_CONTROL_METHODS | FUTURE_BRANCH_METHODS
ARCHIVED_METHODS = set(METHOD_INFO) - ACTIVE_METHODS


def method_visibility(method: str) -> str:
    if method in E1_MAIN_METHODS:
        return "active_e1"
    if method in ACTIVE_CONTROL_METHODS:
        return "active_control"
    if method in FUTURE_BRANCH_METHODS:
        return "future_branch"
    if method in METHOD_INFO:
        return "archived_probe"
    return "unknown"


def active_method_names(*, include_archived: bool = False) -> List[str]:
    names = sorted(METHOD_INFO)
    if include_archived:
        return names
    return [name for name in names if method_visibility(name) != "archived_probe"]


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
    import numpy as np

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def extract_csta_extra_metrics(result_row: Dict[str, object], method: str) -> Dict[str, object]:
    from core.pia_operator import pia_operator_metadata

    if str(method).startswith("rn_ecl_spg_pia_"):
        extra = {
            "operator_name": "RN-ECL-SPG-PIA",
            "dictionary_estimator": "telm2_support_projection",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_rank_normalized_ecl_support_projected_gradient",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("gi_spg_pia_"):
        extra = {
            "operator_name": "GI-SPG-PIA",
            "dictionary_estimator": "telm2_support_projection_elm_generalized_inverse",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_generalized_inverse_support_projected_gradient",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method) in {"spg_cfm_one_step", "spg_cfm_k3", "spg_cfm_film_one_step", "spg_cfm_align_one_step"}:
        extra = {
            "operator_name": "SPG-CFM",
            "dictionary_estimator": "telm2_support_projection",
            "activation_policy": "spg_conditioned_cfm_operator",
            "activation_scope": "anchor_conditioned_spg_conditional_flow_matching",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("ecl_spg_pia_"):
        extra = {
            "operator_name": "ECL-SPG-PIA",
            "dictionary_estimator": "telm2_support_projection",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_energy_calibrated_langevin_support_projected_gradient",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("spg_pia_"):
        extra = {
            "operator_name": "SPG-PIA",
            "dictionary_estimator": "telm2_support_projection",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_support_projected_gradient",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("lc_"):
        extra = {
            "operator_name": "Label-Consistent Latent Residual Flow",
            "dictionary_estimator": "label_consistent_boundary_reweighted_residual_distribution",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_label_consistent_latent_residual",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("task_guided_"):
        extra = {
            "operator_name": "Task-Guided Latent Residual Flow",
            "dictionary_estimator": "task_reweighted_latent_residual_distribution",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_task_guided_latent_residual",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("latent_residual_"):
        extra = {
            "operator_name": "Latent Residual Flow",
            "dictionary_estimator": "latent_residual_distribution",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_latent_residual",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("cs_flow_"):
        extra = {
            "operator_name": "CS-Flow",
            "dictionary_estimator": "one_step_flow_matching",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_flow",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    elif str(method).startswith("ag_"):
        extra = {
            "operator_name": "AG-PIA",
            "dictionary_estimator": "augmentation_field_generalized_inverse",
            "activation_policy": str(method),
            "activation_scope": "anchor_conditioned_operator",
            "activation_topk": "",
            "activation_tau": "",
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    else:
        extra = pia_operator_metadata(csta_policy_for_method(method))
    for field in CSTA_RESULT_PASSTHROUGH_FIELDS:
        if field in result_row:
            extra[field] = clean_metric_value(result_row[field])
    if "selected_template_entropy" not in extra and "template_usage_entropy" in extra:
        extra["selected_template_entropy"] = extra["template_usage_entropy"]
    return extra
