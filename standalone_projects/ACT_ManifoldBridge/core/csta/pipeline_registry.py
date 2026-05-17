from __future__ import annotations

"""Static pipeline dispatch for ACT/CSTA single-run experiments.

This module keeps method-family routing out of ``experiment.py`` without moving
to a dynamic plugin system.  The static table is intentionally explicit: it is
easy to audit which algos are routed to generation-engine pipelines, while the
experiment loop remains focused on splits, result rows, and optional artifacts.
"""

from typing import Callable, Dict, Iterable, Tuple

from core.csta.pipelines import (
    _run_ag_pia_pipeline,
    _run_act_pipeline,
    _run_act_rc4_multiz_fused_pipeline,
    _run_act_zpia_template_pool_pipeline,
    _run_cs_flow_pipeline,
    _run_gi_spg_pia_pipeline,
    _run_latent_residual_pipeline,
    _run_lc_latent_pipeline,
    _run_spg_cfm_pipeline,
    _run_spg_pia_pipeline,
    _run_task_guided_latent_pipeline,
)


SPG_PIA_ALGOS = {
    "spg_pia_zhead",
    "spg_pia_zhead_deterministic",
    "ecl_spg_pia_zhead",
    "ecl_spg_pia_zhead_deterministic",
    "rn_ecl_spg_pia_zhead",
    "rn_ecl_spg_pia_zhead_deterministic",
}

SPG_CFM_ALGOS = {
    "spg_cfm_one_step",
    "spg_cfm_k3",
    "spg_cfm_film_one_step",
    "spg_cfm_align_one_step",
}

GENERATION_ENGINE_HANDLERS: Tuple[Tuple[Iterable[str], Callable[..., Dict[str, object]]], ...] = (
    (SPG_PIA_ALGOS, _run_spg_pia_pipeline),
    ({"gi_spg_pia_zhead"}, _run_gi_spg_pia_pipeline),
    (SPG_CFM_ALGOS, _run_spg_cfm_pipeline),
    ({"lc_residual_direct", "lc_latent_residual_flow"}, _run_lc_latent_pipeline),
    ({"task_guided_residual_direct", "task_guided_latent_residual_flow"}, _run_task_guided_latent_pipeline),
    ({"latent_residual_direct", "latent_residual_flow"}, _run_latent_residual_pipeline),
    ({"cs_flow_target_direct", "cs_flow_single_step"}, _run_cs_flow_pipeline),
    ({"ag_target_direct", "ag_pia_single", "ag_pia_multihead5"}, _run_ag_pia_pipeline),
)


def run_pipeline_for_algo(args, common_kwargs: Dict[str, object]) -> Dict[str, object]:
    for algos, handler in GENERATION_ENGINE_HANDLERS:
        if args.algo in algos:
            return handler(**common_kwargs, method=args.algo)
    if args.algo == "zpia_top1_pool":
        return _run_act_zpia_template_pool_pipeline(
            **common_kwargs,
            algo_label="zpia_top1_pool",
            top1_only=True,
        )
    if args.algo == "zpia_multidir_pool":
        return _run_act_zpia_template_pool_pipeline(
            **common_kwargs,
            algo_label="zpia_multidir_pool",
            top1_only=False,
        )
    if args.algo == "rc4_multiz_fused":
        return _run_act_rc4_multiz_fused_pipeline(**common_kwargs)
    return _run_act_pipeline(**common_kwargs)
