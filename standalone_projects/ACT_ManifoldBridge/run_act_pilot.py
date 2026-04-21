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
    compute_fisher_pia_terms, build_lraes_direction_bank, build_lraes_class_basis_bank, build_pia_direction_bank
)
from core.curriculum import (
    active_direction_probs, 
    build_curriculum_aug_candidates,
    build_acl_candidate_pool,
    apply_safe_step_constraint
)
from host_alignment_probe import (
    compute_candidate_usefulness_batch,
    compute_gradient_alignment,
    compute_entropy_shift,
    score_hard_positive_candidates,
)
from utils.datasets import load_trials_for_dataset, make_trial_split, AEON_FIXED_SPLIT_SPECS
from utils.evaluators import (
    build_model, fit_eval_minirocket, fit_eval_resnet1d, fit_eval_resnet1d_acl,
    fit_eval_resnet1d_continue_ce,
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


def _save_acl_audits(out_root: str, dataset_name: str, seed: int, candidate_rows: List[Dict[str, object]], selected_rows: List[Dict[str, object]]):
    audit_dir = os.path.join(out_root, "audit")
    os.makedirs(audit_dir, exist_ok=True)

    candidate_df = pd.DataFrame([_materialize_audit_row(r) for r in candidate_rows])
    selected_df = pd.DataFrame([_materialize_audit_row(r) for r in selected_rows])
    candidate_path = os.path.join(audit_dir, f"{dataset_name}_s{seed}_candidate_scores.csv")
    selected_path = os.path.join(audit_dir, f"{dataset_name}_s{seed}_selected_positives.csv")
    candidate_df.to_csv(candidate_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    return candidate_path, selected_path


def _materialize_audit_row(row: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in row.items():
        if key in {"z_src", "z_cand", "x_cand"}:
            continue
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                out[key] = float(value.item())
            else:
                out[key] = json.dumps(np.asarray(value).tolist())
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _build_selected_positive_map(selected_rows: List[Dict[str, object]]) -> Dict[int, List[np.ndarray]]:
    positive_map: Dict[int, List[np.ndarray]] = {}
    for row in selected_rows:
        anchor_idx = int(row["anchor_index"])
        x_cand = np.asarray(row["x_cand"], dtype=np.float32)
        positive_map.setdefault(anchor_idx, []).append(x_cand)
    return positive_map


def _run_mba_pipeline(
    *,
    args,
    seed: int,
    dataset_name: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray | None,
    y_val: np.ndarray | None,
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
    if args.algo == "lraes":
        W, _ = build_lraes_direction_bank(
            X_train_z,
            y_train,
            k_dir=args.k_dir,
            fisher_cfg=FisherPIAConfig(),
            lraes_cfg=LRAESConfig(),
        )
    else:
        W, _ = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)

    effective_k = W.shape[0]
    print(f"Requested K: {args.k_dir} | Effective K: {effective_k} | Classes: {len(np.unique(y_train))}")

    def _fit(X_tr, y_tr, is_baseline=False):
        return_model = args.theory_diagnostics and is_baseline
        kwargs = {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "patience": patience,
            "device": args.device,
            "return_model_obj": return_model,
        }
        if args.model == "resnet1d":
            return fit_eval_resnet1d(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
        if args.model == "patchtst":
            return fit_eval_patchtst(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)
        if args.model == "timesnet":
            return fit_eval_timesnet(X_tr, y_tr, X_val_raw, y_val, X_test_raw, y_test, **kwargs)

        m = build_model(n_kernels=args.n_kernels, random_state=seed)
        return fit_eval_minirocket(m, X_tr, y_tr, X_test_raw, y_test)

    print("Fitting Baseline...")
    res_base = _fit(X_train_raw, y_train, is_baseline=True)

    gamma_budget = np.full((effective_k,), args.pia_gamma)
    probs = active_direction_probs(gamma_budget, freeze_eps=0.01)
    eta_val = 0.5 if not args.disable_safe_step else None
    z_aug, y_aug, tid_aug, z_src, dir_ids, aug_meta = build_curriculum_aug_candidates(
        X_train_z,
        y_train,
        np.array([r.tid for r in train_recs]),
        direction_bank=W,
        direction_probs=probs,
        gamma_by_dir=gamma_budget,
        multiplier=args.multiplier,
        seed=seed + 42,
        eta_safe=eta_val,
    )

    aug_trials = []
    bridge_metrics = []
    tid_to_rec = {r.tid: r for r in train_recs}
    for i in range(len(z_aug)):
        src = tid_to_rec[tid_aug[i]]
        sigma_aug = logvec_to_spd(z_aug[i], mean_log)
        x_aug, meta_b = bridge_single(
            torch.from_numpy(src.x_raw),
            torch.from_numpy(src.sigma_orig),
            torch.from_numpy(sigma_aug),
        )
        aug_trials.append({"x": x_aug.numpy(), "y": int(y_aug[i])})
        bridge_metrics.append(meta_b)

    if len(aug_trials) > 0:
        X_mix = np.concatenate([X_train_raw, np.stack([t["x"] for t in aug_trials])])
        y_mix = np.concatenate([y_train, np.array([t["y"] for t in aug_trials])])
    else:
        X_mix, y_mix = X_train_raw, y_train

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

            alignment_metrics["host_geom_cosine_mean"] = float(np.mean([p["alignment_cosine"] for p in aligns])) if aligns else 0.0
            alignment_metrics["host_conflict_rate"] = float(np.mean([p["is_conflict"] for p in aligns])) if aligns else 0.0

    print(f"Fitting ACT Model ({len(X_mix)} samples)...")
    res_act = _fit(X_mix, y_mix, is_baseline=False)
    avg_bridge = pd.DataFrame(bridge_metrics).mean().to_dict() if bridge_metrics else {}

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": avg_bridge,
        "safe_radius_ratio_mean": aug_meta.get("safe_radius_ratio_mean", 1.0),
        "manifold_margin_mean": aug_meta.get("manifold_margin_mean", 0.0),
        "host_geom_cosine_mean": alignment_metrics["host_geom_cosine_mean"],
        "host_conflict_rate": alignment_metrics["host_conflict_rate"],
        "viz_payload": {
            "Z_aug": z_aug,
            "y_aug": y_aug,
            "X_aug_raw": np.stack([t["x"] for t in aug_trials[:20]]) if aug_trials else None,
        },
    }


def _run_gcg_acl_pipeline(
    *,
    args,
    seed: int,
    dataset_name: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray | None,
    y_val: np.ndarray | None,
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
    if args.model != "resnet1d":
        raise ValueError("gcg_acl pipeline currently supports only model=resnet1d")
    if args.algo != "lraes":
        raise ValueError("gcg_acl pipeline currently expects algo=lraes")

    warmup_epochs = int(args.acl_warmup_epochs)
    if epochs > 1:
        warmup_epochs = min(max(1, warmup_epochs), epochs - 1)
    else:
        warmup_epochs = 1
    acl_epochs = max(0, int(epochs) - warmup_epochs)
    if acl_epochs <= 0:
        print("Warning: total epochs leave no ACL fine-alignment epochs after warm-up.")

    print(f"Phase A Warm-up: epochs={warmup_epochs}")
    res_warmup = fit_eval_resnet1d(
        X_train_raw,
        y_train,
        X_val_raw,
        y_val,
        X_test_raw,
        y_test,
        epochs=warmup_epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=args.device,
        return_model_obj=True,
    )
    warmup_model = res_warmup["model_obj"]
    warmup_state = {k: v.detach().cpu().clone() for k, v in warmup_model.state_dict().items()}

    if acl_epochs > 0:
        print(f"Matched-Budget CE Baseline: continue CE for epochs={acl_epochs}")
        res_base = fit_eval_resnet1d_continue_ce(
            X_train_raw,
            y_train,
            X_val_raw,
            y_val,
            X_test_raw,
            y_test,
            init_state_dict=warmup_state,
            epochs=acl_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=args.device,
            return_model_obj=False,
        )
    else:
        res_base = {
            "accuracy": res_warmup["accuracy"],
            "macro_f1": res_warmup["macro_f1"],
            "best_val_f1": res_warmup.get("best_val_f1", 0.0),
            "best_val_loss": res_warmup.get("best_val_loss", float("inf")),
            "stop_epoch": 0,
        }

    print("Phase B Candidate Build: generating class-conditioned basis...")
    class_basis_bank, basis_meta = build_lraes_class_basis_bank(
        X_train_z,
        y_train,
        k_dir=args.k_dir,
        fisher_cfg=FisherPIAConfig(),
        lraes_cfg=LRAESConfig(top_k_per_class=int(args.k_dir)),
    )
    candidates, candidate_meta = build_acl_candidate_pool(
        X_train_z,
        y_train,
        np.array([r.tid for r in train_recs]),
        class_basis_bank=class_basis_bank,
        candidates_per_anchor=args.acl_candidates_per_anchor,
        gamma_scale=args.pia_gamma,
        seed=seed + 42,
        eta_safe=None if args.disable_safe_step else 0.5,
    )

    tid_to_rec = {r.tid: r for r in train_recs}
    dev = _get_dev(args.device)
    warmup_model = warmup_model.to(dev)

    candidate_rows: List[Dict[str, object]] = []
    probe_batch_size = 32
    for start in range(0, len(candidates), probe_batch_size):
        chunk = candidates[start : start + probe_batch_size]
        x_orig_list = []
        x_cand_list = []
        y_list = []
        bridged_chunk = []
        for item in chunk:
            src = tid_to_rec[item["tid"]]
            sigma_cand = logvec_to_spd(item["z_cand"], mean_log)
            x_cand, bridge_meta = bridge_single(
                torch.from_numpy(src.x_raw),
                torch.from_numpy(src.sigma_orig),
                torch.from_numpy(sigma_cand),
            )
            enriched = dict(item)
            enriched["x_cand"] = x_cand.numpy()
            enriched.update({
                "transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A": float(bridge_meta.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(bridge_meta.get("metric_preservation_error", 0.0)),
            })
            bridged_chunk.append(enriched)
            x_orig_list.append(src.x_raw)
            x_cand_list.append(enriched["x_cand"])
            y_list.append(src.y)

        if not bridged_chunk:
            continue

        x_orig_batch = torch.from_numpy(np.stack(x_orig_list)).float()
        x_cand_batch = torch.from_numpy(np.stack(x_cand_list)).float()
        y_batch = torch.from_numpy(np.asarray(y_list, dtype=np.int64))
        usefulness = compute_candidate_usefulness_batch(
            warmup_model,
            x_orig_batch,
            y_batch,
            x_cand_batch,
            device=args.device,
        )
        for enriched, probe_metrics in zip(bridged_chunk, usefulness):
            enriched.update(probe_metrics)
            candidate_rows.append(enriched)

    scored_rows, selected_rows = score_hard_positive_candidates(
        candidate_rows,
        alignment_weight=args.acl_alignment_weight,
        positives_per_anchor=args.acl_positives_per_anchor,
    )
    candidate_csv, selected_csv = _save_acl_audits(args.out_root, dataset_name, seed, scored_rows, selected_rows)
    selected_positive_map = _build_selected_positive_map(selected_rows)
    
    # Build alignment map for soft gating
    selected_alignment_map: Dict[int, List[float]] = {}
    for row in selected_rows:
        anchor_idx = int(row["anchor_index"])
        align_score = float(row.get("alignment_cosine", 1.0))
        selected_alignment_map.setdefault(anchor_idx, []).append(align_score)

    if acl_epochs > 0:
        print(f"Phase C ACL Fine Alignment: epochs={acl_epochs} | selected anchors={len(selected_positive_map)} | soft-gating={args.acl_soft_gating}")
        res_acl = fit_eval_resnet1d_acl(
            X_train_raw,
            y_train,
            X_val_raw,
            y_val,
            X_test_raw,
            y_test,
            init_state_dict=warmup_state,
            selected_positive_map=selected_positive_map,
            epochs=acl_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=args.device,
            acl_temperature=args.acl_temperature,
            acl_loss_weight=args.acl_loss_weight,
            aug_ce_mode=args.acl_aug_ce_mode,
            selected_alignment_map=selected_alignment_map,
            soft_gating=args.acl_soft_gating,
            gating_tau=args.acl_gating_tau,
            return_model_obj=False,
        )
    else:
        res_acl = {
            "accuracy": res_warmup["accuracy"],
            "macro_f1": res_warmup["macro_f1"],
            "best_val_f1": res_warmup.get("best_val_f1", 0.0),
            "best_val_loss": res_warmup.get("best_val_loss", float("inf")),
            "stop_epoch": 0,
            "last_ce_loss": 0.0,
            "last_supcon_loss": 0.0,
            "selected_anchor_count": int(sum(1 for v in selected_positive_map.values() if v)),
            "selected_positive_count": int(sum(len(v) for v in selected_positive_map.values())),
        }

    avg_bridge = pd.DataFrame(
        [
            {
                "transport_error_fro": float(r.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": float(r.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A": float(r.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(r.get("metric_preservation_error", 0.0)),
                "safe_radius_ratio": float(r.get("safe_radius_ratio", 1.0)),
            }
            for r in scored_rows
        ]
    ).mean().to_dict() if scored_rows else {}

    return {
        "res_base": res_base,
        "res_act": res_acl,
        "res_warmup": res_warmup,
        "avg_bridge": avg_bridge,
        "safe_radius_ratio_mean": candidate_meta.get("safe_radius_ratio_mean", 1.0),
        "manifold_margin_mean": candidate_meta.get("manifold_margin_mean", 0.0),
        "host_geom_cosine_mean": float(np.mean([float(r.get("alignment_cosine", 0.0)) for r in scored_rows])) if scored_rows else 0.0,
        "host_conflict_rate": float(np.mean([1.0 if float(r.get("alignment_cosine", 0.0)) < 0 else 0.0 for r in scored_rows])) if scored_rows else 0.0,
        "selected_anchor_count": int(sum(1 for v in selected_positive_map.values() if v)),
        "selected_positive_count": int(sum(len(v) for v in selected_positive_map.values())),
        "candidate_total_count": int(len(scored_rows)),
        "hard_positive_score_mean": float(np.mean([float(r.get("hard_positive_score", 0.0)) for r in selected_rows])) if selected_rows else 0.0,
        "fidelity_score_mean": float(np.mean([float(r.get("fidelity_score", 0.0)) for r in selected_rows])) if selected_rows else 0.0,
        "candidate_csv": candidate_csv,
        "selected_csv": selected_csv,
        "basis_meta": basis_meta,
        "mean_aug_ce_weight": res_acl.get("mean_aug_ce_weight", 1.0),
        "zero_weight_fraction": res_acl.get("zero_weight_fraction", 0.0),
    }


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
            if args.pipeline == "gcg_acl":
                pipeline_out = _run_gcg_acl_pipeline(
                    args=args,
                    seed=seed,
                    dataset_name=dataset_name,
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
            else:
                pipeline_out = _run_mba_pipeline(
                    args=args,
                    seed=seed,
                    dataset_name=dataset_name,
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
            res_warmup = pipeline_out.get("res_warmup")
            avg_bridge = pipeline_out.get("avg_bridge", {})

            summary = {
                "dataset": dataset_name, "seed": seed, "status": "success", 
                "algo": args.algo, "model": args.model, "pipeline": args.pipeline,
                "acl_aug_ce_mode": args.acl_aug_ce_mode if args.pipeline == "gcg_acl" else "n/a",
                "base_f1": res_base["macro_f1"], "act_f1": res_act["macro_f1"],
                "gain": res_act["macro_f1"] - res_base["macro_f1"],
                "warmup_f1": res_warmup["macro_f1"] if res_warmup is not None else np.nan,
                
                # Prop 1: Transport Fidelity
                "transport_error_fro_mean": avg_bridge.get("transport_error_fro", 0),
                "transport_error_logeuc_mean": avg_bridge.get("transport_error_logeuc", 0),
                "bridge_cond_A_mean": avg_bridge.get("bridge_cond_A", 0),
                "metric_preservation_error_mean": avg_bridge.get("metric_preservation_error", 0),
                
                # Prop 2: Safe Region
                "safe_radius_ratio_mean": pipeline_out.get("safe_radius_ratio_mean", 1.0),
                "manifold_margin_mean": pipeline_out.get("manifold_margin_mean", 0),
                
                # Prop 3: Host Alignment
                "host_geom_cosine_mean": pipeline_out.get("host_geom_cosine_mean", 0.0),
                "host_conflict_rate": pipeline_out.get("host_conflict_rate", 0.0),
                
                "base_stop_epoch": res_base.get("stop_epoch", 0),
                "act_stop_epoch": res_act.get("stop_epoch", 0),
                "warmup_stop_epoch": res_warmup.get("stop_epoch", 0) if res_warmup is not None else np.nan,
                "f1_gain_pct": (res_act["macro_f1"] - res_base["macro_f1"]) / (res_base["macro_f1"] + 1e-7) * 100,
                "base_best_val_f1": res_base.get("best_val_f1", 0),
                "act_best_val_f1": res_act.get("best_val_f1", 0),
                "warmup_best_val_f1": res_warmup.get("best_val_f1", 0) if res_warmup is not None else np.nan,
                "selected_anchor_count": pipeline_out.get("selected_anchor_count", 0),
                "selected_positive_count": pipeline_out.get("selected_positive_count", 0),
                "candidate_total_count": pipeline_out.get("candidate_total_count", 0),
                "hard_positive_score_mean": pipeline_out.get("hard_positive_score_mean", 0.0),
                "fidelity_score_mean": pipeline_out.get("fidelity_score_mean", 0.0),
                "mean_aug_ce_weight": pipeline_out.get("mean_aug_ce_weight", 1.0),
                "zero_weight_fraction": pipeline_out.get("zero_weight_fraction", 0.0),
                "acl_last_ce_loss": res_act.get("last_ce_loss", 0.0),
                "acl_last_supcon_loss": res_act.get("last_supcon_loss", 0.0),
            }
            print(f"Base: {summary['base_f1']:.4f} | ACT: {summary['act_f1']:.4f} | Gain: {summary['gain']:.4f} ({summary['f1_gain_pct']:.1f}%)")
            if res_warmup is not None:
                print(f"Warm-up reference: {summary['warmup_f1']:.4f}")
            results.append(summary)

            # Optional: Save visualization data
            if args.save_viz_samples and args.pipeline == "mba":
                viz_dir = os.path.join(args.out_root, "viz_data")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{dataset_name}_s{seed}_viz.npz")
                np.savez(save_path,
                    Z_orig=X_train_z, 
                    y_orig=y_train,
                    Z_aug=pipeline_out["viz_payload"]["Z_aug"],
                    y_aug=pipeline_out["viz_payload"]["y_aug"],
                    X_orig_raw=X_train_raw[:20], # Sample 20 for waveform plotting
                    X_aug_raw=pipeline_out["viz_payload"]["X_aug_raw"],
                    mean_log=mean_log
                )
                print(f"Visualization samples saved to {save_path}")

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
    parser = argparse.ArgumentParser(description="ACT Full-Scale sweep")
    parser.add_argument("--dataset", type=str, default="natops")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--pipeline", type=str, choices=["mba", "gcg_acl"], default="mba")
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
    parser.add_argument("--save-viz-samples", action="store_true", help="Save latent and raw samples for paper visualizations")
    parser.add_argument("--acl-warmup-epochs", type=int, default=10)
    parser.add_argument("--acl-candidates-per-anchor", type=int, default=4)
    parser.add_argument("--acl-temperature", type=float, default=0.07)
    parser.add_argument("--acl-loss-weight", type=float, default=0.2)
    parser.add_argument("--acl-alignment-weight", type=float, default=0.7)
    parser.add_argument("--acl-positives-per-anchor", type=int, choices=[1, 2], default=1)
    parser.add_argument("--acl-aug-ce-mode", type=str, choices=["none", "selected"], default="selected", 
                        help="Pure ACL (none) vs Hybrid ACL (selected)")
    parser.add_argument("--acl-soft-gating", action="store_true", help="Enable Alignment-Guided Soft CE Gating")
    parser.add_argument("--acl-gating-tau", type=float, default=0.0, help="Tau threshold for soft gating")
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
