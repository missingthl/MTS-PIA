from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from utils.external_runner_registry import csta_policy_for_method

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AG_PIA_METHODS = {"ag_target_direct", "ag_pia_single", "ag_pia_multihead5"}
CS_FLOW_METHODS = {"cs_flow_target_direct", "cs_flow_single_step"}
LATENT_RESIDUAL_METHODS = {"latent_residual_direct", "latent_residual_flow"}
TASK_GUIDED_LATENT_METHODS = {"task_guided_residual_direct", "task_guided_latent_residual_flow"}
LC_LATENT_METHODS = {"lc_residual_direct", "lc_latent_residual_flow"}
SPG_PIA_METHODS = {
    "spg_pia_zhead",
    "spg_pia_zhead_deterministic",
    "ecl_spg_pia_zhead",
    "ecl_spg_pia_zhead_deterministic",
    "rn_ecl_spg_pia_zhead",
    "rn_ecl_spg_pia_zhead_deterministic",
}
GI_SPG_METHODS = {"gi_spg_pia_zhead"}
SPG_CFM_METHODS = {"spg_cfm_one_step", "spg_cfm_k3", "spg_cfm_film_one_step", "spg_cfm_align_one_step"}
OPERATOR_METHODS = (
    AG_PIA_METHODS
    | CS_FLOW_METHODS
    | LATENT_RESIDUAL_METHODS
    | TASK_GUIDED_LATENT_METHODS
    | LC_LATENT_METHODS
    | SPG_PIA_METHODS
    | GI_SPG_METHODS
    | SPG_CFM_METHODS
)


def resolve_csta_algo_and_selection(method: str) -> Tuple[str, Optional[str]]:
    """Map an external-runner arm to the ACT CLI algo/template policy.

    Operator-generation methods intentionally return ``selection=None`` so
    callers do not accidentally route them through template policies.
    """
    if "ao_fisher" in method:
        return "ao_fisher", "topk_uniform_top5"
    if "ao_contrastive" in method:
        return "ao_contrastive", "topk_uniform_top5"
    if method in OPERATOR_METHODS:
        return method, None
    return "zpia_top1_pool", None


def _extend_ag_pia_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--ag-k-pos",
            str(args.ag_k_pos),
            "--ag-k-neg",
            str(args.ag_k_neg),
            "--ag-lambda-tangent",
            str(args.ag_lambda_tangent),
            "--ag-lambda-inter",
            str(args.ag_lambda_inter),
            "--ag-hidden-dim",
            str(args.ag_hidden_dim),
            "--ag-ridge",
            str(args.ag_ridge),
            "--ag-activation",
            str(args.ag_activation),
        ]
    )


def _extend_cs_flow_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--cs-flow-epochs",
            str(args.cs_flow_epochs),
            "--cs-flow-batch-size",
            str(args.cs_flow_batch_size),
            "--cs-flow-lr",
            str(args.cs_flow_lr),
            "--cs-flow-weight-decay",
            str(args.cs_flow_weight_decay),
            "--cs-flow-k-same",
            str(args.cs_flow_k_same),
            "--cs-flow-hidden-layers",
            str(args.cs_flow_hidden_layers),
            "--cs-flow-hidden-width",
            str(args.cs_flow_hidden_width),
            "--cs-flow-class-embedding-dim",
            str(args.cs_flow_class_embedding_dim),
            "--cs-flow-t-gen",
            str(args.cs_flow_t_gen),
        ]
    )


def _extend_latent_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--latent-flow-epochs",
            str(args.latent_flow_epochs),
            "--latent-flow-batch-size",
            str(args.latent_flow_batch_size),
            "--latent-flow-lr",
            str(args.latent_flow_lr),
            "--latent-flow-weight-decay",
            str(args.latent_flow_weight_decay),
            "--latent-hidden-layers",
            str(args.latent_hidden_layers),
            "--latent-hidden-width",
            str(args.latent_hidden_width),
            "--latent-class-embedding-dim",
            str(args.latent_class_embedding_dim),
            "--latent-lambda-cos",
            str(args.latent_lambda_cos),
            "--latent-rbf-tau-floor",
            str(args.latent_rbf_tau_floor),
        ]
    )


def _extend_task_guidance_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--task-guidance-beta",
            str(args.task_guidance_beta),
            "--task-guidance-margin-min",
            str(args.task_guidance_margin_min),
            "--task-guidance-lambda-margin",
            str(args.task_guidance_lambda_margin),
            "--task-guidance-warmup-epochs",
            str(args.task_guidance_warmup_epochs),
            "--task-guidance-max-candidates",
            str(args.task_guidance_max_candidates),
        ]
    )


def _extend_lc_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--lc-beta",
            str(args.lc_beta),
            "--lc-margin-floor",
            str(args.lc_margin_floor),
            "--lc-gamma-eps",
            str(args.lc_gamma_eps),
            "--lc-warmup-epochs",
            str(args.lc_warmup_epochs),
            "--lc-max-candidates",
            str(args.lc_max_candidates),
        ]
    )


def _extend_spg_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--spg-zhead-epochs",
            str(args.spg_zhead_epochs),
            "--spg-zhead-hidden-dim",
            str(args.spg_zhead_hidden_dim),
            "--spg-zhead-lr",
            str(args.spg_zhead_lr),
            "--spg-zhead-weight-decay",
            str(args.spg_zhead_weight_decay),
            "--spg-zhead-batch-size",
            str(args.spg_zhead_batch_size),
            "--spg-projection-ridge",
            str(args.spg_projection_ridge),
            "--spg-noise-sigma",
            str(args.spg_noise_sigma),
        ]
    )


def _extend_gi_spg_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--gi-spg-hidden-dim",
            str(args.gi_spg_hidden_dim),
            "--gi-spg-ridge",
            str(args.gi_spg_ridge),
            "--gi-spg-activation",
            str(args.gi_spg_activation),
        ]
    )


def _extend_spg_cfm_args(cmd: List[str], args) -> None:
    cmd.extend(
        [
            "--spg-cfm-flow-epochs",
            str(args.spg_cfm_flow_epochs),
            "--spg-cfm-flow-batch-size",
            str(args.spg_cfm_flow_batch_size),
            "--spg-cfm-flow-lr",
            str(args.spg_cfm_flow_lr),
            "--spg-cfm-flow-weight-decay",
            str(args.spg_cfm_flow_weight_decay),
            "--spg-cfm-hidden-layers",
            str(args.spg_cfm_hidden_layers),
            "--spg-cfm-hidden-width",
            str(args.spg_cfm_hidden_width),
            "--spg-cfm-class-embedding-dim",
            str(args.spg_cfm_class_embedding_dim),
            "--spg-cfm-lambda-cos",
            str(getattr(args, "spg_cfm_lambda_cos", 0.5)),
            "--spg-cfm-lambda-align",
            str(getattr(args, "spg_cfm_lambda_align", 0.05)),
        ]
    )


def append_method_family_args(cmd: List[str], method: str, args) -> None:
    if method in AG_PIA_METHODS:
        _extend_ag_pia_args(cmd, args)
    if method in CS_FLOW_METHODS:
        _extend_cs_flow_args(cmd, args)
    if method in LATENT_RESIDUAL_METHODS:
        _extend_latent_args(cmd, args)
    if method in TASK_GUIDED_LATENT_METHODS:
        _extend_latent_args(cmd, args)
        _extend_task_guidance_args(cmd, args)
    if method in LC_LATENT_METHODS:
        _extend_latent_args(cmd, args)
        _extend_lc_args(cmd, args)
    if method in SPG_PIA_METHODS | GI_SPG_METHODS:
        _extend_spg_args(cmd, args)
    if method in GI_SPG_METHODS:
        _extend_gi_spg_args(cmd, args)
    if method in SPG_CFM_METHODS:
        _extend_spg_cfm_args(cmd, args)


def append_template_policy_args(cmd: List[str], method: str, selection: Optional[str], args) -> None:
    if selection is not None:
        cmd.extend(["--template-selection", selection])
    elif method == "csta_group_template_top":
        cmd.extend(["--template-selection", "group_top", "--group-size", str(args.group_size)])
    elif method.startswith("csta_topk_"):
        cmd.extend(["--template-selection", method.replace("csta_", "")])
    elif method in {
        "csta_template_random_within_bank",
        "csta_fv_filter_top5",
        "csta_fv_score_top5",
        "csta_random_feasible_selector",
    }:
        cmd.extend(["--template-selection", csta_policy_for_method(method)])


def build_csta_command(
    *,
    dataset: str,
    seed: int,
    method: str,
    args,
    csta_root: Path,
    python_executable: str | None = None,
) -> List[str]:
    algo_name, selection = resolve_csta_algo_and_selection(method)
    cmd = [
        python_executable or sys.executable,
        str(PROJECT_ROOT / "run_act_pilot.py"),
        "--dataset",
        dataset,
        "--pipeline",
        "act",
        "--algo",
        algo_name,
        "--model",
        args.backbone,
        "--seeds",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--patience",
        str(args.patience),
        "--val-ratio",
        str(args.val_ratio),
        "--k-dir",
        str(args.k_dir),
        "--n-kernels",
        str(args.n_kernels),
        "--pia-gamma",
        str(args.pia_gamma),
        "--eta-safe",
        str(args.eta_safe),
        "--multiplier",
        str(args.multiplier),
        "--device",
        args.device,
        "--out-root",
        str(csta_root),
        "--audit-method-label",
        method,
    ]
    append_method_family_args(cmd, method, args)
    append_template_policy_args(cmd, method, selection, args)
    return cmd
