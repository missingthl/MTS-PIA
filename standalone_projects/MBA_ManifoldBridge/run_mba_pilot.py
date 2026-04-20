import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import numpy as np
import torch
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Local imports
from core.bridge import bridge_single, logvec_to_spd
from core.pia import (
    FisherPIAConfig, LRAESConfig, 
    compute_fisher_pia_terms, build_lraes_direction_bank, build_pia_direction_bank
)
from core.curriculum import (
    active_direction_probs, 
    build_curriculum_aug_candidates,
    apply_safe_step_constraint
)
from host_alignment_probe import compute_gradient_alignment, compute_entropy_shift
from utils.datasets import load_trials_for_dataset, make_trial_split, AEON_FIXED_SPLIT_SPECS
from utils.evaluators import (
    build_model, fit_eval_minirocket, fit_eval_resnet1d, 
    fit_eval_patchtst, fit_eval_timesnet, _get_dev
)


@dataclass
class TrialRecord:
    tid: str
    y: int
    x_raw: np.ndarray
    sigma_orig: np.ndarray
    z: np.ndarray


def _build_trial_records(trials, spd_eps=1e-4):
    if not trials: return [], None
    records = []; log_covs = []
    for t in trials:
        x = torch.from_numpy(t.x).double()
        x = x - x.mean(dim=-1, keepdim=True)
        cov = (x @ x.transpose(-1, -2)) / (x.shape[-1] - 1)
        cov = cov + spd_eps * torch.eye(cov.shape[0])
        vals, vecs = torch.linalg.eigh(cov)
        log_cov = vecs @ torch.diag_embed(torch.log(torch.clamp(vals, min=spd_eps))) @ vecs.transpose(-1, -2)
        log_covs.append(log_cov.numpy())
        records.append({"tid": t.tid, "y": t.y, "x_raw": t.x, "sigma_orig": cov.numpy(), "log_cov": log_cov.numpy()})
    
    mean_log = np.mean(log_covs, axis=0)
    final_records = []
    idx = np.triu_indices(mean_log.shape[0])
    for r in records:
        z = (r["log_cov"] - mean_log)[idx]
        final_records.append(TrialRecord(tid=r["tid"], y=r["y"], x_raw=r["x_raw"], sigma_orig=r["sigma_orig"], z=z))
    return final_records, mean_log


def run_experiment(dataset_name, args):
    print(f"\n>>>> Dataset: {dataset_name} | Model: {args.model} <<<<")
    try:
        all_trials = load_trials_for_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return [{
            "dataset": dataset_name, "seed": -1, "status": "failed", "fail_reason": str(e),
            "requested_k_dir": args.k_dir, "effective_k_dir": 0, "algo": args.algo, "model": args.model
        }]

    # Load host defaults if requested
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    patience = args.patience
    
    if args.host_config != "none":
        if args.host_config == "resnet1d_default":
            epochs, lr, batch_size, patience = 30, 1e-3, 64, 10
        elif args.host_config == "patchtst_default":
            epochs, lr, batch_size, patience = 100, 5e-4, 64, 15
        elif args.host_config == "timesnet_default":
            epochs, lr, batch_size, patience = 100, 5e-4, 32, 15

    results = []
    seeds = [int(s) for s in args.seeds.split(",")]
    
    for seed in seeds:
        print(f"Seed {seed}...")
        try:
            train_trials, test_trials, val_trials = make_trial_split(all_trials, seed=seed, val_ratio=args.val_ratio)
            train_recs, mean_log = _build_trial_records(train_trials)
            test_recs, _ = _build_trial_records(test_trials)
            val_recs, _ = _build_trial_records(val_trials)
            
            X_train_raw = np.stack([r.x_raw for r in train_recs])
            y_train = np.array([r.y for r in train_recs])
            X_test_raw = np.stack([r.x_raw for r in test_recs])
            y_test = np.array([r.y for r in test_recs])
            
            X_val_raw, y_val = None, None
            if val_recs:
                X_val_raw = np.stack([r.x_raw for r in val_recs])
                y_val = np.array([r.y for r in val_recs])

            X_train_z = np.stack([r.z for r in train_recs])
            num_classes = len(np.unique(y_train))
            latent_dim = X_train_z.shape[1]

            # 1. Bank Generation
            if args.algo == "lraes":
                W, _ = build_lraes_direction_bank(X_train_z, y_train, k_dir=args.k_dir, 
                                                 fisher_cfg=FisherPIAConfig(), lraes_cfg=LRAESConfig())
            else:
                W, _ = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)

            effective_k = W.shape[0]
            print(f"Requested K: {args.k_dir} | Effective K: {effective_k} | Classes: {num_classes}")

            # Define training dispatcher
            def _fit(X_tr, y_tr, is_baseline=False):
                # Only return heavy model object if theory-diagnostics is on and it is the baseline fit
                return_model = args.theory_diagnostics and is_baseline
                
                kwargs = {
                    "epochs": epochs, "lr": lr, "batch_size": batch_size, 
                    "patience": patience, "device": args.device,
                    "return_model_obj": return_model
                }
                
                if args.model == "resnet1d":
                    return fit_eval_resnet1d(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
                elif args.model == "patchtst":
                    return fit_eval_patchtst(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
                elif args.model == "timesnet":
                    return fit_eval_timesnet(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
                else:
                    m = build_model(n_kernels=args.n_kernels, random_state=seed)
                    return fit_eval_minirocket(m, X_tr, y_tr, X_test_raw, y_test)

            # 2. Baseline
            print("Fitting Baseline...")
            res_base = _fit(X_train_raw, y_train, is_baseline=True)
            
            # 3. MBA Augmentation (Proposition 2: Safe Region)
            gamma_budget = np.full((effective_k,), args.pia_gamma)
            probs = active_direction_probs(gamma_budget, freeze_eps=0.01)
            
            # Use safety coefficient eta=0.5 (Proposition 2)
            eta_val = 0.5 if not args.disable_safe_step else None
            z_aug, y_aug, tid_aug, z_src, dir_ids, aug_meta = build_curriculum_aug_candidates(
                X_train_z, y_train, np.array([r.tid for r in train_recs]),
                direction_bank=W, direction_probs=probs, gamma_by_dir=gamma_budget, 
                multiplier=args.multiplier, seed=seed + 42,
                eta_safe=eta_val
            )
            
            aug_trials = []
            bridge_metrics = []
            tid_to_rec = {r.tid: r for r in train_recs}
            for i in range(len(z_aug)):
                src = tid_to_rec[tid_aug[i]]
                sigma_aug = logvec_to_spd(z_aug[i], mean_log)
                x_aug, meta_b = bridge_single(torch.from_numpy(src.x_raw), torch.from_numpy(src.sigma_orig), torch.from_numpy(sigma_aug))
                aug_trials.append({"x": x_aug.numpy(), "y": int(y_aug[i])})
                bridge_metrics.append(meta_b)
            
            if len(aug_trials) > 0:
                X_mix = np.concatenate([X_train_raw, np.stack([t["x"] for t in aug_trials])])
                y_mix = np.concatenate([y_train, np.array([t["y"] for t in aug_trials])])
            else:
                X_mix, y_mix = X_train_raw, y_train
            
            # --- Proposition 3: Host Alignment Probing (Gated) ---
            alignment_metrics = {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}
            if args.theory_diagnostics and args.model != "minirocket" and "model_obj" in res_base:
                print("Running Theory Diagnostics (Host Alignment Probe)...")
                with torch.enable_grad():
                    aligns = []
                    probe_idx = np.random.choice(len(aug_trials), min(20, len(aug_trials)), replace=False)
                    for i in probe_idx:
                        src = tid_to_rec[tid_aug[i]]
                        x_o = torch.from_numpy(src.x_raw).unsqueeze(0).float()
                        y_o = torch.tensor([src.y]).long()
                        x_a = torch.from_numpy(aug_trials[i]["x"]).unsqueeze(0).float()
                        
                        probe = compute_gradient_alignment(res_base["model_obj"], x_o, y_o, x_a, device=args.device)
                        aligns.append(probe)
                    
                    alignment_metrics["host_geom_cosine_mean"] = float(np.mean([p["alignment_cosine"] for p in aligns]))
                    alignment_metrics["host_conflict_rate"] = float(np.mean([p["is_conflict"] for p in aligns]))

            print(f"Fitting MBA Model ({len(X_mix)} samples)...")
            res_mba = _fit(X_mix, y_mix, is_baseline=False)
            
            # Aggregate theoretical metrics
            avg_bridge = pd.DataFrame(bridge_metrics).mean().to_dict() if bridge_metrics else {}

            summary = {
                "dataset": dataset_name, "seed": seed, "status": "success", 
                "algo": args.algo, "model": args.model,
                "base_f1": res_base["macro_f1"], "mba_f1": res_mba["macro_f1"],
                "gain": res_mba["macro_f1"] - res_base["macro_f1"],
                
                # Prop 1: Transport Fidelity
                "transport_error_fro_mean": avg_bridge.get("transport_error_fro", 0),
                "transport_error_logeuc_mean": avg_bridge.get("transport_error_logeuc", 0),
                "bridge_cond_A_mean": avg_bridge.get("bridge_cond_A", 0),
                "metric_preservation_error_mean": avg_bridge.get("metric_preservation_error", 0),
                
                # Prop 2: Safe Region
                "safe_radius_ratio_mean": aug_meta.get("safe_radius_ratio_mean", 1.0),
                "manifold_margin_mean": aug_meta.get("manifold_margin_mean", 0),
                
                # Prop 3: Host Alignment
                "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
                "host_conflict_rate": alignment_metrics["host_conflict_rate"],
                
                "base_stop_epoch": res_base.get("stop_epoch", 0),
                "mba_stop_epoch": res_mba.get("stop_epoch", 0),
                "f1_gain_pct": (res_mba["macro_f1"] - res_base["macro_f1"]) / (res_base["macro_f1"] + 1e-7) * 100,
                "base_best_val_f1": res_base.get("best_val_f1", 0),
                "mba_best_val_f1": res_mba.get("best_val_f1", 0)
            }
            print(f"Base: {summary['base_f1']:.4f} | MBA: {summary['mba_f1']:.4f} | Gain: {summary['gain']:.4f} ({summary['f1_gain_pct']:.1f}%)")
            results.append(summary)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in {dataset_name} Seed {seed}: {e}")
            results.append({
                "dataset": dataset_name, "seed": seed, "status": "failed", "fail_reason": str(e),
                "requested_k_dir": args.k_dir, "effective_k_dir": 0, "algo": args.algo, "model": args.model
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MBA Full-Scale sweep")
    parser.add_argument("--dataset", type=str, default="natops")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--algo", type=str, choices=["pia", "lraes"], default="lraes")
    parser.add_argument("--model", type=str, choices=["minirocket", "resnet1d", "patchtst", "timesnet"], default="resnet1d")
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
    parser.add_argument("--theory-diagnostics", action="store_true", help="Enable heavy theoretical metrics (host alignment, etc)")
    parser.add_argument("--disable-safe-step", action="store_true", help="Ablation: Disable Safe-Step constraint")
    parser.add_argument("--out-root", type=str, default="results/full_sweep_v1")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    
    datasets = [args.dataset]
    if args.all_datasets:
        datasets = sorted(list(AEON_FIXED_SPLIT_SPECS.keys()))
    
    all_results = []
    for ds in datasets:
        try:
            res = run_experiment(ds, args)
            all_results.extend(res)
            pd.DataFrame(all_results).to_csv(f"{args.out_root}/sweep_results.csv", index=False)
        except Exception as e:
            print(f"Failed {ds}: {e}")

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(f"{args.out_root}/final_results.csv", index=False)
    print(f"\nSweep Complete! Results saved to {args.out_root}/final_results.csv")


if __name__ == "__main__":
    main()
