import os

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from core.bridge import bridge_single, logvec_to_spd
from core.curriculum import active_direction_probs, build_curriculum_aug_candidates
from core.pia import (
    FisherPIAConfig,
    LRAESConfig,
    build_lraes_direction_bank,
    build_pia_direction_bank,
)
from host_alignment_probe import compute_gradient_alignment
from utils.datasets import AEON_FIXED_SPLIT_SPECS, load_trials_for_dataset, make_trial_split
from utils.evaluators import (
    build_model,
    fit_eval_minirocket,
    fit_eval_patchtst,
    fit_eval_resnet1d,
    fit_eval_timesnet,
)


@dataclass
class TrialRecord:
    tid: str
    y: int
    x_raw: np.ndarray
    sigma_orig: np.ndarray
    z: np.ndarray


def _build_trial_records(trials, spd_eps: float = 1e-4):
    if not trials:
        return [], None

    records = []
    log_covs = []
    for t in trials:
        x = torch.from_numpy(t.x).double()
        x = x - x.mean(dim=-1, keepdim=True)
        cov = (x @ x.transpose(-1, -2)) / (x.shape[-1] - 1)
        cov = cov + spd_eps * torch.eye(cov.shape[0], dtype=cov.dtype)
        vals, vecs = torch.linalg.eigh(cov)
        log_cov = vecs @ torch.diag_embed(torch.log(torch.clamp(vals, min=spd_eps))) @ vecs.transpose(-1, -2)
        log_covs.append(log_cov.numpy())
        records.append(
            {
                "tid": t.tid,
                "y": t.y,
                "x_raw": t.x,
                "sigma_orig": cov.numpy(),
                "log_cov": log_cov.numpy(),
            }
        )

    mean_log = np.mean(log_covs, axis=0)
    idx = np.triu_indices(mean_log.shape[0])
    final_records = []
    for record in records:
        z = (record["log_cov"] - mean_log)[idx]
        final_records.append(
            TrialRecord(
                tid=record["tid"],
                y=record["y"],
                x_raw=record["x_raw"],
                sigma_orig=record["sigma_orig"],
                z=z,
            )
        )
    return final_records, mean_log


def _fit_host_model(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    return_model_obj: bool = False,
    loader_seed: Optional[int] = None,
) -> Dict[str, object]:
    kwargs = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "device": args.device,
        "return_model_obj": return_model_obj,
    }
    if args.model == "resnet1d":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_resnet1d(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "patchtst":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_patchtst(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
    if args.model == "timesnet":
        kwargs["loader_seed"] = loader_seed
        return fit_eval_timesnet(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)

    model = build_model(n_kernels=args.n_kernels, random_state=loader_seed or 42)
    return fit_eval_minirocket(model, X_tr, y_tr, X_test_raw, y_test)


def _build_act_realized_augmentations(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
) -> Dict[str, object]:
    if args.algo == "lraes":
        direction_bank, _ = build_lraes_direction_bank(
            X_train_z,
            y_train,
            k_dir=args.k_dir,
            fisher_cfg=FisherPIAConfig(),
            lraes_cfg=LRAESConfig(),
        )
    else:
        direction_bank, _ = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)

    effective_k = int(direction_bank.shape[0])
    print(f"Requested K: {args.k_dir} | Effective K: {effective_k} | Classes: {len(np.unique(y_train))}")

    gamma_budget = np.full((effective_k,), float(args.pia_gamma), dtype=np.float64)
    direction_probs = active_direction_probs(gamma_budget, freeze_eps=0.01)
    eta_safe = None if args.disable_safe_step else 0.5
    tid_train = np.asarray([record.tid for record in train_recs], dtype=object)

    z_aug, y_aug, tid_aug, _, _, aug_meta = build_curriculum_aug_candidates(
        X_train_z,
        y_train,
        tid_train,
        direction_bank=direction_bank,
        direction_probs=direction_probs,
        gamma_by_dir=gamma_budget,
        multiplier=args.multiplier,
        seed=seed + 42,
        eta_safe=eta_safe,
    )

    tid_to_rec = {record.tid: record for record in train_recs}
    aug_trials: List[Dict[str, object]] = []
    bridge_metrics: List[Dict[str, object]] = []
    for i in range(len(z_aug)):
        src = tid_to_rec[tid_aug[i]]
        sigma_aug = logvec_to_spd(z_aug[i], mean_log)
        x_aug, bridge_meta = bridge_single(
            torch.from_numpy(src.x_raw),
            torch.from_numpy(src.sigma_orig),
            torch.from_numpy(sigma_aug),
        )
        aug_trials.append({"x": x_aug.numpy(), "y": int(y_aug[i]), "tid": tid_aug[i]})
        bridge_metrics.append(bridge_meta)

    X_aug_raw = np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None
    y_aug_np = np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None
    avg_bridge = pd.DataFrame(bridge_metrics).mean().to_dict() if bridge_metrics else {}

    return {
        "effective_k": effective_k,
        "z_aug": z_aug,
        "y_aug": y_aug,
        "tid_aug": tid_aug,
        "aug_trials": aug_trials,
        "X_aug_raw": X_aug_raw,
        "y_aug_np": y_aug_np,
        "tid_to_rec": tid_to_rec,
        "avg_bridge": avg_bridge,
        "safe_radius_ratio_mean": aug_meta.get("safe_radius_ratio_mean", 1.0),
        "manifold_margin_mean": aug_meta.get("manifold_margin_mean", 0.0),
        "candidate_total_count": int(aug_meta.get("aug_total_count", len(aug_trials))),
        "aug_total_count": int(aug_meta.get("aug_total_count", len(aug_trials))),
    }


def _run_analysis_probe(
    *,
    args,
    model_obj,
    tid_aug: np.ndarray,
    aug_trials: List[Dict[str, object]],
    tid_to_rec: Dict[str, TrialRecord],
) -> Dict[str, float]:
    alignment_metrics = {"host_geom_cosine_mean": 0.0, "host_conflict_rate": 0.0}
    if not args.theory_diagnostics or args.model == "minirocket" or model_obj is None or not aug_trials:
        return alignment_metrics

    print("Running theory diagnostics...")
    with torch.enable_grad():
        aligns = []
        probe_idx = np.random.choice(len(aug_trials), min(20, len(aug_trials)), replace=False)
        for idx in probe_idx:
            src = tid_to_rec[tid_aug[idx]]
            x_orig = torch.from_numpy(src.x_raw).unsqueeze(0).float()
            y_orig = torch.tensor([src.y]).long()
            x_aug = torch.from_numpy(aug_trials[idx]["x"]).unsqueeze(0).float()
            aligns.append(compute_gradient_alignment(model_obj, x_orig, y_orig, x_aug, device=args.device))

        if aligns:
            alignment_metrics["host_geom_cosine_mean"] = float(np.mean([probe["alignment_cosine"] for probe in aligns]))
            alignment_metrics["host_conflict_rate"] = float(np.mean([probe["is_conflict"] for probe in aligns]))
    return alignment_metrics


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
        aug_trials=aug_out["aug_trials"],
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
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "viz_payload": {
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]]) if aug_out["aug_trials"] else None,
        },
    }


def run_experiment(dataset_name, args):
    print(f"\n>>>> Dataset: {dataset_name} | Model: {args.model} <<<<")
    try:
        all_trials = load_trials_for_dataset(dataset_name)
    except Exception as exc:
        print(f"Failed to load {dataset_name}: {exc}")
        return [
            {
                "dataset": dataset_name,
                "seed": -1,
                "status": "failed",
                "fail_reason": str(exc),
                "requested_k_dir": args.k_dir,
                "effective_k_dir": 0,
                "algo": args.algo,
                "model": args.model,
                "pipeline": "act",
            }
        ]

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
    seeds = [int(seed) for seed in args.seeds.split(",")]
    for seed in seeds:
        print(f"Seed {seed}...")
        try:
            train_trials, test_trials, val_trials = make_trial_split(all_trials, seed=seed, val_ratio=args.val_ratio)
            train_recs, mean_log = _build_trial_records(train_trials)
            test_recs, _ = _build_trial_records(test_trials)
            val_recs, _ = _build_trial_records(val_trials)

            X_train_raw = np.stack([record.x_raw for record in train_recs])
            y_train = np.asarray([record.y for record in train_recs], dtype=np.int64)
            X_test_raw = np.stack([record.x_raw for record in test_recs])
            y_test = np.asarray([record.y for record in test_recs], dtype=np.int64)

            X_val_raw, y_val = None, None
            if val_recs:
                X_val_raw = np.stack([record.x_raw for record in val_recs])
                y_val = np.asarray([record.y for record in val_recs], dtype=np.int64)

            X_train_z = np.stack([record.z for record in train_recs])
            pipeline_out = _run_act_pipeline(
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
            )

            res_base = pipeline_out["res_base"]
            res_act = pipeline_out["res_act"]
            avg_bridge = pipeline_out.get("avg_bridge", {})
            gain = float(res_act["macro_f1"] - res_base["macro_f1"])
            summary = {
                "dataset": dataset_name,
                "seed": seed,
                "status": "success",
                "algo": args.algo,
                "model": args.model,
                "pipeline": "act",
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
                "host_geom_cosine_mean": float(pipeline_out.get("host_geom_cosine_mean", 0.0)),
                "host_conflict_rate": float(pipeline_out.get("host_conflict_rate", 0.0)),
                "candidate_total_count": int(pipeline_out.get("candidate_total_count", 0)),
                "aug_total_count": int(pipeline_out.get("aug_total_count", 0)),
                "requested_k_dir": int(args.k_dir),
                "effective_k_dir": int(pipeline_out.get("effective_k", 0)),
            }
            print(
                f"Base: {summary['base_f1']:.4f} | "
                f"ACT: {summary['act_f1']:.4f} | "
                f"Gain: {summary['gain']:.4f} ({summary['f1_gain_pct']:.1f}%)"
            )
            results.append(summary)

            if args.save_viz_samples:
                viz_dir = os.path.join(args.out_root, "viz_data")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{dataset_name}_s{seed}_viz.npz")
                np.savez(
                    save_path,
                    Z_orig=X_train_z,
                    y_orig=y_train,
                    Z_aug=pipeline_out["viz_payload"]["Z_aug"],
                    y_aug=pipeline_out["viz_payload"]["y_aug"],
                    X_orig_raw=X_train_raw[:20],
                    X_aug_raw=pipeline_out["viz_payload"]["X_aug_raw"],
                    mean_log=mean_log,
                )
                print(f"Visualization samples saved to {save_path}")

        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"Error in {dataset_name} Seed {seed}: {exc}")
            results.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "status": "failed",
                    "fail_reason": str(exc),
                    "requested_k_dir": args.k_dir,
                    "effective_k_dir": 0,
                    "algo": args.algo,
                    "model": args.model,
                    "pipeline": "act",
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="ACT_ManifoldBridge original ACT runner")
    parser.add_argument("--dataset", type=str, default="natops")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--pipeline", type=str, choices=["act", "mba"], default="act")
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
    parser.add_argument("--theory-diagnostics", action="store_true", help="Enable host-alignment diagnostics for sampled augmented trials")
    parser.add_argument("--disable-safe-step", action="store_true", help="Disable Safe-Step constraint")
    parser.add_argument("--save-viz-samples", action="store_true", help="Save latent and raw samples for visualization")
    parser.add_argument("--out-root", type=str, default="standalone_projects/ACT_ManifoldBridge/results/act_core")
    args = parser.parse_args()

    if args.pipeline == "mba":
        print("Using legacy pipeline alias 'mba' -> 'act'.")

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
    final_df.to_csv(os.path.join(args.out_root, "final_results.csv"), index=False)
    print(f"\nSweep complete. Results saved to {os.path.join(args.out_root, 'final_results.csv')}")


if __name__ == "__main__":
    main()
