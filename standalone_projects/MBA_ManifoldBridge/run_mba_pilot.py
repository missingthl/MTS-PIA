import os
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
    build_curriculum_aug_candidates
)
from utils.datasets import load_trials_for_dataset, make_trial_split, AEON_FIXED_SPLIT_SPECS
from utils.evaluators import build_model, fit_eval_minirocket, fit_eval_resnet1d


@dataclass
class TrialRecord:
    tid: str
    y: int
    x_raw: np.ndarray
    sigma_orig: np.ndarray
    z: np.ndarray


def _build_trial_records(trials, spd_eps=1e-4):
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
    all_trials = load_trials_for_dataset(dataset_name)
    results = []

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        print(f"Seed {seed}...")
        train_trials, test_trials, _ = make_trial_split(all_trials, seed=seed)
        train_recs, mean_log = _build_trial_records(train_trials)
        test_recs, _ = _build_trial_records(test_trials)
        
        X_train_z = np.stack([r.z for r in train_recs])
        y_train = np.array([r.y for r in train_recs])
        X_test_z = np.stack([r.z for r in test_recs])
        y_test = np.array([r.y for r in test_recs])
        X_train_raw = np.stack([r.x_raw for r in train_recs])
        X_test_raw = np.stack([r.x_raw for r in test_recs])

        # 1. Bank Generation
        if args.algo == "lraes":
            W, _ = build_lraes_direction_bank(X_train_z, y_train, k_dir=args.k_dir, 
                                             fisher_cfg=FisherPIAConfig(), lraes_cfg=LRAESConfig())
        else:
            W, _ = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)

        # 2. Baseline
        print("Fitting Baseline...")
        if args.model == "resnet1d":
            res_base = fit_eval_resnet1d(X_train_raw, y_train, X_test_raw, y_test, 
                                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device)
        else:
            model_base = build_model(n_kernels=args.n_kernels, random_state=seed)
            res_base = fit_eval_minirocket(model_base, X_train_raw, y_train, X_test_raw, y_test)
        
        # 3. MBA Augmentation
        gamma_budget = np.full((args.k_dir,), args.pia_gamma)
        probs = active_direction_probs(gamma_budget, freeze_eps=0.01)
        z_aug, y_aug, tid_aug, _, _, _ = build_curriculum_aug_candidates(
            X_train_z, y_train, np.array([r.tid for r in train_recs]),
            direction_bank=W, direction_probs=probs, gamma_by_dir=gamma_budget, 
            multiplier=args.multiplier, seed=seed + 42
        )
        
        aug_trials = []
        tid_to_rec = {r.tid: r for r in train_recs}
        for i in range(len(z_aug)):
            src = tid_to_rec[tid_aug[i]]
            sigma_aug = logvec_to_spd(z_aug[i], mean_log)
            x_aug, _ = bridge_single(torch.from_numpy(src.x_raw), torch.from_numpy(src.sigma_orig), torch.from_numpy(sigma_aug))
            aug_trials.append({"x": x_aug.numpy(), "y": int(y_aug[i])})
        
        X_mix = np.concatenate([X_train_raw, np.stack([t["x"] for t in aug_trials])])
        y_mix = np.concatenate([y_train, np.array([t["y"] for t in aug_trials])])
        
        print(f"Fitting MBA Model ({len(X_mix)} samples)...")
        if args.model == "resnet1d":
            res_mba = fit_eval_resnet1d(X_mix, y_mix, X_test_raw, y_test, 
                                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device)
        else:
            model_mba = build_model(n_kernels=args.n_kernels, random_state=seed)
            res_mba = fit_eval_minirocket(model_mba, X_mix, y_mix, X_test_raw, y_test)
        
        summary = {
            "dataset": dataset_name, "seed": seed, "algo": args.algo, "model": args.model,
            "base_f1": res_base["macro_f1"], "mba_f1": res_mba["macro_f1"],
            "gain": res_mba["macro_f1"] - res_base["macro_f1"]
        }
        print(f"Base: {summary['base_f1']:.4f} | MBA: {summary['mba_f1']:.4f} | Gain: {summary['gain']:.4f}")
        results.append(summary)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MBA Full-Scale sweep")
    parser.add_argument("--dataset", type=str, default="natops")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--algo", type=str, choices=["pia", "lraes"], default="lraes")
    parser.add_argument("--model", type=str, choices=["minirocket", "resnet1d"], default="minirocket")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--k-dir", type=int, default=10)
    parser.add_argument("--pia-gamma", type=float, default=0.1)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--n-kernels", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-root", type=str, default="results/full_sweep_v1")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    
    datasets = [args.dataset]
    if args.all_datasets:
        datasets = ["natops", "har"] + list(AEON_FIXED_SPLIT_SPECS.keys())
    
    all_results = []
    for ds in datasets:
        try:
            res = run_experiment(ds, args)
            all_results.extend(res)
            # Save intermediate
            pd.DataFrame(all_results).to_csv(f"{args.out_root}/sweep_results.csv", index=False)
        except Exception as e:
            print(f"Failed {ds}: {e}")

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(f"{args.out_root}/final_results.csv", index=False)
    print(f"\nSweep Complete! Results saved to {args.out_root}/final_results.csv")


if __name__ == "__main__":
    main()
