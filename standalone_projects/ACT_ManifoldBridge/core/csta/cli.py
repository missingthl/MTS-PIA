from __future__ import annotations

import argparse
import os

import pandas as pd

from utils.datasets import AEON_FIXED_SPLIT_SPECS

from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="ACT_ManifoldBridge original ACT runner")
    parser.add_argument("--dataset", type=str, default="natops")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--pipeline", type=str, choices=["act", "mba", "mba_feedback"], default="act")
    parser.add_argument(
        "--algo",
        type=str,
        choices=[
            "pia",
            "lraes",
            "zpia",
            "rc4_fused",
            "zpia_top1_pool",
            "zpia_multidir_pool",
            "rc4_multiz_fused",
            "ao_fisher",
            "ao_contrastive",
            "ag_target_direct",
            "ag_pia_single",
            "ag_pia_multihead5",
            "cs_flow_target_direct",
            "cs_flow_single_step",
            "latent_residual_direct",
            "latent_residual_flow",
            "task_guided_residual_direct",
            "task_guided_latent_residual_flow",
            "lc_residual_direct",
            "lc_latent_residual_flow",
            "spg_pia_zhead",
            "spg_pia_zhead_deterministic",
            "ecl_spg_pia_zhead",
            "ecl_spg_pia_zhead_deterministic",
            "rn_ecl_spg_pia_zhead",
            "rn_ecl_spg_pia_zhead_deterministic",
            "gi_spg_pia_zhead",
            "spg_cfm_one_step",
            "spg_cfm_k3",
            "spg_cfm_film_one_step",
            "spg_cfm_align_one_step",
        ],
        default="lraes",
    )
    parser.add_argument("--model", type=str, choices=["minirocket", "resnet1d", "patchtst", "timesnet", "mptsnet", "moderntcn"], default="resnet1d")
    parser.add_argument("--host-config", type=str, choices=["none", "resnet1d_default", "patchtst_default", "timesnet_default"], default="none")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--k-dir", type=int, default=10)
    parser.add_argument("--pia-gamma", type=float, default=0.1)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--n-kernels", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-root", type=str, default="standalone_projects/ACT_ManifoldBridge/results/act_core")
    parser.add_argument("--theory-diagnostics", action="store_true")
    parser.add_argument("--save-viz-samples", action="store_true")
    parser.add_argument("--disable-safe-step", action="store_true")
    parser.add_argument("--osf-alpha", type=float, default=0.5)
    parser.add_argument("--osf-beta", type=float, default=0.5)
    parser.add_argument("--osf-kappa", type=float, default=1.0)
    parser.add_argument("--multi-template-pairs", type=int, default=0)
    parser.add_argument("--telm2-c-repr", type=float, default=10.0)
    parser.add_argument("--telm2-n-iters", type=int, default=50)
    parser.add_argument("--telm2-activation", type=str, choices=["sine", "sigmoid", "none"], default="sine")
    parser.add_argument("--telm2-bias-update-mode", type=str, choices=["none", "mean", "ema"], default="none")
    parser.add_argument("--feedback-margin-temperature", type=float, default=1.0)
    parser.add_argument("--aug-loss-weight", type=float, default=0.5)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--aug-weight-mode", type=str, default="sigmoid")
    parser.add_argument(
        "--template-selection",
        type=str,
        choices=[
            "top_response",
            "random",
            "fixed",
            "group_random",
            "group_top",
            "group_avg_response",
            "group_top_random_sameclass",
            "sameclass_zmix",
            "topk_softmax_tau_0.05",
            "topk_softmax_tau_0.10",
            "topk_softmax_tau_0.20",
            "topk_uniform_top5",
            "fv_filter_top5",
            "fv_score_top5",
            "random_feasible_selector",
        ],
        default="top_response",
    )
    parser.add_argument("--template-source", type=str, choices=["zpia", "pca", "random_orth"], default="zpia")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--eta-safe", type=float, default=0.75)
    parser.add_argument("--ag-k-pos", type=int, default=5)
    parser.add_argument("--ag-k-neg", type=int, default=5)
    parser.add_argument("--ag-lambda-tangent", type=float, default=0.5)
    parser.add_argument("--ag-lambda-inter", type=float, default=0.25)
    parser.add_argument("--ag-hidden-dim", type=int, default=0)
    parser.add_argument("--ag-ridge", type=float, default=1e-3)
    parser.add_argument("--ag-activation", type=str, choices=["tanh", "sigmoid", "none"], default="tanh")
    parser.add_argument("--cs-flow-epochs", type=int, default=50)
    parser.add_argument("--cs-flow-batch-size", type=int, default=128)
    parser.add_argument("--cs-flow-lr", type=float, default=1e-3)
    parser.add_argument("--cs-flow-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cs-flow-k-same", type=int, default=5)
    parser.add_argument("--cs-flow-hidden-layers", type=int, default=2)
    parser.add_argument("--cs-flow-hidden-width", type=int, default=0)
    parser.add_argument("--cs-flow-class-embedding-dim", type=int, default=0)
    parser.add_argument("--cs-flow-t-gen", type=float, default=0.0)
    parser.add_argument("--latent-flow-epochs", type=int, default=50)
    parser.add_argument("--latent-flow-batch-size", type=int, default=128)
    parser.add_argument("--latent-flow-lr", type=float, default=1e-3)
    parser.add_argument("--latent-flow-weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-hidden-layers", type=int, default=2)
    parser.add_argument("--latent-hidden-width", type=int, default=0)
    parser.add_argument("--latent-class-embedding-dim", type=int, default=0)
    parser.add_argument("--latent-lambda-cos", type=float, default=0.5)
    parser.add_argument("--latent-rbf-tau-floor", type=float, default=1e-12)
    parser.add_argument("--task-guidance-beta", type=float, default=1.0)
    parser.add_argument("--task-guidance-margin-min", type=float, default=0.0)
    parser.add_argument("--task-guidance-lambda-margin", type=float, default=1.0)
    parser.add_argument("--task-guidance-warmup-epochs", type=int, default=10)
    parser.add_argument("--task-guidance-max-candidates", type=int, default=0)
    parser.add_argument("--lc-beta", type=float, default=1.0)
    parser.add_argument("--lc-margin-floor", type=float, default=0.0)
    parser.add_argument("--lc-gamma-eps", type=float, default=1e-12)
    parser.add_argument("--lc-warmup-epochs", type=int, default=10)
    parser.add_argument("--lc-max-candidates", type=int, default=0)
    parser.add_argument("--spg-zhead-epochs", type=int, default=50)
    parser.add_argument("--spg-zhead-hidden-dim", type=int, default=0)
    parser.add_argument("--spg-zhead-lr", type=float, default=1e-3)
    parser.add_argument("--spg-zhead-weight-decay", type=float, default=1e-4)
    parser.add_argument("--spg-zhead-batch-size", type=int, default=128)
    parser.add_argument("--spg-projection-ridge", type=float, default=1e-6)
    parser.add_argument("--spg-noise-sigma", type=float, default=0.1)
    parser.add_argument("--gi-spg-hidden-dim", type=int, default=0)
    parser.add_argument("--gi-spg-ridge", type=float, default=1e-3)
    parser.add_argument("--gi-spg-activation", type=str, choices=["tanh", "sigmoid", "none"], default="tanh")
    parser.add_argument("--spg-cfm-flow-epochs", type=int, default=50)
    parser.add_argument("--spg-cfm-flow-batch-size", type=int, default=128)
    parser.add_argument("--spg-cfm-flow-lr", type=float, default=1e-3)
    parser.add_argument("--spg-cfm-flow-weight-decay", type=float, default=1e-4)
    parser.add_argument("--spg-cfm-hidden-layers", type=int, default=2)
    parser.add_argument("--spg-cfm-hidden-width", type=int, default=0)
    parser.add_argument("--spg-cfm-class-embedding-dim", type=int, default=0)
    parser.add_argument("--spg-cfm-lambda-cos", type=float, default=0.5)
    parser.add_argument("--spg-cfm-lambda-align", type=float, default=0.05)
    parser.add_argument("--audit-method-label", type=str, default="")
    args = parser.parse_args()

    if args.pipeline == "mba":
        print("Using legacy pipeline alias 'mba' -> 'act'.")
    if args.feedback_margin_temperature <= 0.0:
        raise ValueError("--feedback-margin-temperature must be positive.")
    if args.aug_loss_weight < 0.0:
        raise ValueError("--aug-loss-weight must be non-negative.")
    if args.pipeline == "mba_feedback" and args.model == "minirocket":
        raise ValueError("--pipeline mba_feedback supports resnet1d, patchtst, and timesnet only.")

    os.makedirs(args.out_root, exist_ok=True)
    datasets = [args.dataset]
    if args.all_datasets:
        datasets = sorted(list(AEON_FIXED_SPLIT_SPECS.keys()))

    all_results = []
    for dataset_name in datasets:
        try:
            result_rows = run_experiment(dataset_name, args)
            all_results.extend(result_rows)
            pd.DataFrame(all_results).to_csv(os.path.join(args.out_root, "sweep_results.csv"), index=False)
        except Exception as exc:
            print(f"Failed {dataset_name}: {exc}")

    final_df = pd.DataFrame(all_results)
    out_csv = os.path.join(args.out_root, f"{datasets[0]}_results.csv")
    final_df.to_csv(out_csv, index=False)
    print(f"\nSweep complete. Results saved to {out_csv}")
