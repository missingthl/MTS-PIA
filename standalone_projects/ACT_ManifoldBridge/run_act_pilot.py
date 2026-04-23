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
    build_zpia_direction_bank,
)
from core.wavelet_mba import (
    WaveletTrialRecord,
    build_step_tier_candidates,
    build_wavelet_trial_records,
    compute_identity_checks,
    parse_step_tier_ratios,
    realize_wavelet_candidates,
)
from core.whitened_edit import white_edit_single, white_identity_error
from host_alignment_probe import compute_gradient_alignment
from utils.datasets import AEON_FIXED_SPLIT_SPECS, load_trials_for_dataset, make_trial_split
from utils.evaluators import (
    build_model,
    fit_eval_minirocket,
    fit_eval_patchtst,
    fit_eval_patchtst_weighted_aug_ce,
    fit_eval_resnet1d_adaptive_aug_ce,
    fit_eval_resnet1d,
    fit_eval_resnet1d_weighted_aug_ce,
    fit_eval_timesnet,
    fit_eval_timesnet_weighted_aug_ce,
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


def _fit_host_model_weighted_aug_ce(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_aug: Optional[np.ndarray],
    y_aug: Optional[np.ndarray],
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    loader_seed: Optional[int] = None,
) -> Dict[str, object]:
    kwargs = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "device": args.device,
        "feedback_margin_temperature": args.feedback_margin_temperature,
        "aug_loss_weight": args.aug_loss_weight,
        "loader_seed": loader_seed,
    }
    if args.model == "resnet1d":
        return fit_eval_resnet1d_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    if args.model == "patchtst":
        return fit_eval_patchtst_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    if args.model == "timesnet":
        return fit_eval_timesnet_weighted_aug_ce(
            X_tr, y_tr, X_aug, y_aug, X_val_raw, y_val, X_test_raw, y_test, **kwargs
        )
    raise ValueError("Weighted aug-CE training supports resnet1d, patchtst, and timesnet only.")


def _fit_host_model_adaptive_aug_ce(
    *,
    args,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_aug_lraes: Optional[np.ndarray],
    y_aug_lraes: Optional[np.ndarray],
    X_aug_zpia: Optional[np.ndarray],
    y_aug_zpia: Optional[np.ndarray],
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    loader_seed: Optional[int] = None,
) -> Dict[str, object]:
    if args.model != "resnet1d":
        raise ValueError("Adaptive router v1 currently supports resnet1d only.")
    return fit_eval_resnet1d_adaptive_aug_ce(
        X_tr,
        y_tr,
        X_aug_lraes,
        y_aug_lraes,
        X_aug_zpia,
        y_aug_zpia,
        X_val_raw,
        y_val,
        X_test_raw,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=args.device,
        feedback_margin_temperature=args.feedback_margin_temperature,
        aug_loss_weight=args.aug_loss_weight,
        router_temperature=args.router_temperature,
        router_min_prob=args.router_min_prob,
        router_smoothing=args.router_smoothing,
        loader_seed=loader_seed,
    )


def _score_aug_margins(
    *,
    model_obj,
    X_aug: Optional[np.ndarray],
    y_aug: Optional[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    if model_obj is None or X_aug is None or y_aug is None or len(y_aug) == 0:
        return np.empty((0,), dtype=np.float64)
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    dev = torch.device(device if use_cuda else "cpu")
    model_obj.to(dev)
    model_obj.eval()
    margins: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(y_aug), int(batch_size)):
            bx = torch.from_numpy(X_aug[start : start + int(batch_size)]).float().to(dev)
            by = torch.from_numpy(y_aug[start : start + int(batch_size)]).long().to(dev)
            logits = model_obj(bx)
            true_logits = logits.gather(1, by.view(-1, 1)).squeeze(1)
            if logits.shape[1] <= 1:
                margin = true_logits
            else:
                masked = logits.clone()
                masked.scatter_(1, by.view(-1, 1), -torch.inf)
                other_logits = torch.max(masked, dim=1).values
                margin = true_logits - other_logits
            margins.append(margin.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(margins) if margins else np.empty((0,), dtype=np.float64)


def _clone_args_with_updates(args, **updates):
    cloned = argparse.Namespace(**vars(args))
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _build_direction_bank_for_args(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    algo_override: Optional[str] = None,
) -> Dict[str, object]:
    algo_name = str(algo_override or args.algo)
    if algo_name == "lraes":
        direction_bank, direction_meta = build_lraes_direction_bank(
            X_train_z,
            y_train,
            k_dir=args.k_dir,
            fisher_cfg=FisherPIAConfig(),
            lraes_cfg=LRAESConfig(),
        )
    elif algo_name == "zpia":
        direction_bank, direction_meta = build_zpia_direction_bank(
            X_train_z,
            k_dir=args.k_dir,
            seed=seed,
            telm2_n_iters=args.telm2_n_iters,
            telm2_c_repr=args.telm2_c_repr,
            telm2_activation=args.telm2_activation,
            telm2_bias_update_mode=args.telm2_bias_update_mode,
        )
    else:
        direction_bank, direction_meta = build_pia_direction_bank(X_train_z, k_dir=args.k_dir, seed=seed)
    return {"bank": direction_bank, "meta": direction_meta}


def _build_act_realized_augmentations(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    algo_override: Optional[str] = None,
    engine_id: Optional[str] = None,
) -> Dict[str, object]:
    algo_name = str(algo_override or args.algo)
    bank_out = _build_direction_bank_for_args(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        algo_override=algo_name,
    )
    direction_bank = bank_out["bank"]
    direction_meta = bank_out["meta"]

    effective_k = int(direction_bank.shape[0])
    print(
        f"Requested K: {args.k_dir} | Effective K: {effective_k} | "
        f"Source: {direction_meta.get('bank_source', algo_name)} | Classes: {len(np.unique(y_train))}"
    )

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
    audit_rows: List[Dict[str, object]] = []
    candidate_rows = list(aug_meta.get("candidate_rows", []))
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
        audit = candidate_rows[i].copy() if i < len(candidate_rows) else {
            "anchor_index": -1,
            "tid": tid_aug[i],
            "class_id": int(y_aug[i]),
            "candidate_order": int(i),
            "direction_id": -1,
            "sign": 0.0,
            "gamma_used": 0.0,
            "safe_radius_ratio": 0.0,
        }
        audit.update(
            {
                "algo": algo_name,
                "engine_id": str(engine_id or algo_name),
                "direction_bank_source": direction_meta.get("bank_source", algo_name),
                "transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
                "transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "bridge_cond_A": float(bridge_meta.get("bridge_cond_A", 0.0)),
                "metric_preservation_error": float(bridge_meta.get("metric_preservation_error", 0.0)),
            }
        )
        audit_rows.append(audit)

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
        "audit_rows": audit_rows,
        "direction_bank_meta": direction_meta,
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
        "direction_bank_meta": aug_out.get("direction_bank_meta", {}),
        "audit_rows": aug_out.get("audit_rows", []),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]]) if aug_out["aug_trials"] else None,
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
    margins = _score_aug_margins(
        model_obj=res_base.get("model_obj"),
        X_aug=aug_out["X_aug_raw"],
        y_aug=aug_out["y_aug_np"],
        device=args.device,
        batch_size=batch_size,
    )
    scaled_margins = np.clip(margins / max(float(args.feedback_margin_temperature), 1e-6), -60.0, 60.0)
    weights = 1.0 / (1.0 + np.exp(-scaled_margins))
    for idx, row in enumerate(aug_out.get("audit_rows", [])):
        row["margin_aug"] = float(margins[idx]) if idx < len(margins) else 0.0
        row["feedback_weight"] = float(weights[idx]) if idx < len(weights) else 0.0

    print(f"Fitting MBA feedback model ({len(y_train)} orig + {len(aug_out['y_aug_np']) if aug_out['y_aug_np'] is not None else 0} aug stream)...")
    res_act = _fit_host_model_weighted_aug_ce(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_aug=aug_out["X_aug_raw"],
        y_aug=aug_out["y_aug_np"],
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "audit_rows": aug_out.get("audit_rows", []),
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


def _run_mba_feedback_adaptive_pipeline(
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

    aug_out_lraes = _build_act_realized_augmentations(
        args=_clone_args_with_updates(args, algo="lraes"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_override="lraes",
        engine_id="lraes",
    )
    aug_out_zpia = _build_act_realized_augmentations(
        args=_clone_args_with_updates(args, algo="zpia"),
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
        algo_override="zpia",
        engine_id="zpia",
    )

    for engine_name, aug_out in [("lraes", aug_out_lraes), ("zpia", aug_out_zpia)]:
        margins = _score_aug_margins(
            model_obj=res_base.get("model_obj"),
            X_aug=aug_out["X_aug_raw"],
            y_aug=aug_out["y_aug_np"],
            device=args.device,
            batch_size=batch_size,
        )
        scaled_margins = np.clip(margins / max(float(args.feedback_margin_temperature), 1e-6), -60.0, 60.0)
        weights = 1.0 / (1.0 + np.exp(-scaled_margins))
        for idx, row in enumerate(aug_out.get("audit_rows", [])):
            row["engine_id"] = engine_name
            row["margin_aug"] = float(margins[idx]) if idx < len(margins) else 0.0
            row["feedback_weight"] = float(weights[idx]) if idx < len(weights) else 0.0

    print(
        f"Fitting adaptive MBA feedback model "
        f"({len(y_train)} orig + {int(aug_out_lraes['aug_total_count'])} lraes + {int(aug_out_zpia['aug_total_count'])} zpia aug)..."
    )
    res_act = _fit_host_model_adaptive_aug_ce(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_aug_lraes=aug_out_lraes["X_aug_raw"],
        y_aug_lraes=aug_out_lraes["y_aug_np"],
        X_aug_zpia=aug_out_zpia["X_aug_raw"],
        y_aug_zpia=aug_out_zpia["y_aug_np"],
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        loader_seed=seed,
    )

    def _weighted_bridge_mean(key: str) -> float:
        count_l = float(aug_out_lraes.get("aug_total_count", 0))
        count_z = float(aug_out_zpia.get("aug_total_count", 0))
        total = count_l + count_z
        if total <= 0.0:
            return 0.0
        return (
            count_l * float(aug_out_lraes.get("avg_bridge", {}).get(key, 0.0))
            + count_z * float(aug_out_zpia.get("avg_bridge", {}).get(key, 0.0))
        ) / total

    avg_bridge = {
        "transport_error_fro": _weighted_bridge_mean("transport_error_fro"),
        "transport_error_logeuc": _weighted_bridge_mean("transport_error_logeuc"),
        "bridge_cond_A": _weighted_bridge_mean("bridge_cond_A"),
        "metric_preservation_error": _weighted_bridge_mean("metric_preservation_error"),
    }
    audit_rows = list(aug_out_lraes.get("audit_rows", [])) + list(aug_out_zpia.get("audit_rows", []))

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": avg_bridge,
        "audit_rows": audit_rows,
        "direction_bank_meta": {
            "bank_source": "adaptive_dual",
            "engine_sources": [
                aug_out_lraes.get("direction_bank_meta", {}).get("bank_source", "lraes"),
                aug_out_zpia.get("direction_bank_meta", {}).get("bank_source", "zpia_telm2"),
            ],
            "lraes_meta": aug_out_lraes.get("direction_bank_meta", {}),
            "zpia_meta": aug_out_zpia.get("direction_bank_meta", {}),
        },
        "safe_radius_ratio_mean": float(
            np.mean(
                [
                    float(aug_out_lraes.get("safe_radius_ratio_mean", 1.0)),
                    float(aug_out_zpia.get("safe_radius_ratio_mean", 1.0)),
                ]
            )
        ),
        "manifold_margin_mean": float(
            np.mean(
                [
                    float(aug_out_lraes.get("manifold_margin_mean", 0.0)),
                    float(aug_out_zpia.get("manifold_margin_mean", 0.0)),
                ]
            )
        ),
        "host_geom_cosine_mean": 0.0,
        "host_conflict_rate": 0.0,
        "candidate_total_count": int(aug_out_lraes.get("candidate_total_count", 0)) + int(
            aug_out_zpia.get("candidate_total_count", 0)
        ),
        "aug_total_count": int(aug_out_lraes.get("aug_total_count", 0)) + int(aug_out_zpia.get("aug_total_count", 0)),
        "effective_k": int(max(aug_out_lraes.get("effective_k", 0), aug_out_zpia.get("effective_k", 0))),
        "effective_k_lraes": int(aug_out_lraes.get("effective_k", 0)),
        "effective_k_zpia": int(aug_out_zpia.get("effective_k", 0)),
        "router_trace": list(res_act.get("router_trace", [])),
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": np.concatenate([aug_out_lraes["z_aug"], aug_out_zpia["z_aug"]], axis=0),
            "y_aug": np.concatenate([aug_out_lraes["y_aug"], aug_out_zpia["y_aug"]], axis=0),
            "X_aug_raw": np.concatenate(
                [
                    aug_out_lraes["X_aug_raw"][:10] if aug_out_lraes["X_aug_raw"] is not None else np.empty((0,) + X_train_raw.shape[1:], dtype=np.float32),
                    aug_out_zpia["X_aug_raw"][:10] if aug_out_zpia["X_aug_raw"] is not None else np.empty((0,) + X_train_raw.shape[1:], dtype=np.float32),
                ],
                axis=0,
            ),
        },
    }


def _build_mba_white_edit_realized_augmentations(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
) -> Dict[str, object]:
    direction_bank, _ = build_lraes_direction_bank(
        X_train_z,
        y_train,
        k_dir=args.k_dir,
        fisher_cfg=FisherPIAConfig(),
        lraes_cfg=LRAESConfig(),
    )
    effective_k = int(direction_bank.shape[0])
    print(f"Requested K: {args.k_dir} | Effective K: {effective_k} | Classes: {len(np.unique(y_train))}")
    identity_errors = [
        white_identity_error(
            torch.from_numpy(record.x_raw),
            torch.from_numpy(record.sigma_orig),
        )
        for record in train_recs
    ]
    if effective_k == 0:
        return {
            "effective_k": 0,
            "z_aug": np.empty((0, X_train_z.shape[1]), dtype=np.float32),
            "y_aug": np.empty((0,), dtype=np.int64),
            "tid_aug": np.empty((0,), dtype=object),
            "aug_trials": [],
            "X_aug_raw": None,
            "y_aug_np": None,
            "tid_to_rec": {record.tid: record for record in train_recs},
            "avg_bridge": {},
            "audit_rows": [],
            "white_identity_error_mean": float(np.mean(identity_errors)) if identity_errors else 0.0,
            "safe_radius_ratio_mean": 1.0,
            "manifold_margin_mean": 0.0,
            "candidate_total_count": 0,
            "aug_total_count": 0,
        }

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
    candidate_rows = list(aug_meta.get("candidate_rows", []))
    if len(candidate_rows) != len(z_aug):
        raise AssertionError("mba_white_edit requires one candidate metadata row per augmented sample.")

    tid_to_rec = {record.tid: record for record in train_recs}
    aug_trials: List[Dict[str, object]] = []
    edit_metrics: List[Dict[str, float]] = []
    audit_rows: List[Dict[str, object]] = []

    for i in range(len(z_aug)):
        row = candidate_rows[i]
        src = tid_to_rec[tid_aug[i]]
        sigma_aug = logvec_to_spd(z_aug[i], mean_log)
        x_aug, edit_meta = white_edit_single(
            torch.from_numpy(src.x_raw),
            torch.from_numpy(src.sigma_orig),
            torch.from_numpy(sigma_aug),
            sign=float(row.get("sign", 1.0)),
            edit_alpha_scale=float(args.edit_alpha_scale),
        )
        aug_trials.append({"x": x_aug.numpy(), "y": int(y_aug[i]), "tid": tid_aug[i]})
        edit_metrics.append(edit_meta)
        audit = {
            **row,
            "edit_mode": args.edit_mode,
            "edit_basis": args.edit_basis,
            "edit_alpha_scale": float(args.edit_alpha_scale),
            "edit_alpha": float(edit_meta.get("edit_alpha", 0.0)),
            "edit_norm": float(edit_meta.get("edit_norm", 0.0)),
            "edit_energy": float(edit_meta.get("edit_energy", 0.0)),
            "edit_basis_fro_norm": float(edit_meta.get("edit_basis_fro_norm", 0.0)),
            "edit_status_code": float(edit_meta.get("edit_status_code", 0.0)),
            "transport_error_fro": float(edit_meta.get("transport_error_fro", 0.0)),
            "transport_error_logeuc": float(edit_meta.get("transport_error_logeuc", 0.0)),
            "recolor_transport_error_fro": float(edit_meta.get("recolor_transport_error_fro", 0.0)),
            "recolor_transport_error_logeuc": float(edit_meta.get("recolor_transport_error_logeuc", 0.0)),
            "bridge_cond_A": float(edit_meta.get("bridge_cond_A", 0.0)),
        }
        audit_rows.append(audit)

    X_aug_raw = np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None
    y_aug_np = np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None
    avg_bridge = pd.DataFrame(edit_metrics).mean().to_dict() if edit_metrics else {}

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
        "audit_rows": audit_rows,
        "white_identity_error_mean": float(np.mean(identity_errors)) if identity_errors else 0.0,
        "safe_radius_ratio_mean": aug_meta.get("safe_radius_ratio_mean", 1.0),
        "manifold_margin_mean": aug_meta.get("manifold_margin_mean", 0.0),
        "candidate_total_count": int(aug_meta.get("aug_total_count", len(aug_trials))),
        "aug_total_count": int(aug_meta.get("aug_total_count", len(aug_trials))),
    }


def _run_mba_white_edit_pipeline(
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

    aug_out = _build_mba_white_edit_realized_augmentations(
        args=args,
        seed=seed,
        X_train_z=X_train_z,
        y_train=y_train,
        train_recs=train_recs,
        mean_log=mean_log,
    )
    margins = _score_aug_margins(
        model_obj=res_base.get("model_obj"),
        X_aug=aug_out["X_aug_raw"],
        y_aug=aug_out["y_aug_np"],
        device=args.device,
        batch_size=batch_size,
    )
    if len(margins) == len(aug_out["audit_rows"]):
        for row, margin in zip(aug_out["audit_rows"], margins):
            row["margin_aug"] = float(margin)
    else:
        for row in aug_out["audit_rows"]:
            row["margin_aug"] = 0.0

    print(f"Fitting mba_white_edit weighted CE model ({len(X_train_raw)} orig + {aug_out['aug_total_count']} aug)...")
    res_act = _fit_host_model_weighted_aug_ce(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_aug=aug_out["X_aug_raw"],
        y_aug=aug_out["y_aug_np"],
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "effective_k": aug_out["effective_k"],
        "white_identity_error_mean": aug_out["white_identity_error_mean"],
        "audit_rows": aug_out["audit_rows"],
        "viz_payload": {
            "Z_orig": X_train_z,
            "Z_aug": aug_out["z_aug"],
            "y_aug": aug_out["y_aug"],
            "X_aug_raw": np.stack([trial["x"] for trial in aug_out["aug_trials"][:20]]) if aug_out["aug_trials"] else None,
        },
    }


def _build_wavelet_mba_realized_augmentations(
    *,
    args,
    seed: int,
    y_train: np.ndarray,
    train_recs: List[WaveletTrialRecord],
    mean_log_a: np.ndarray,
    mean_log_dm: Optional[np.ndarray],
) -> Dict[str, object]:
    object_mode = str(args.wavelet_object_mode)
    X_train_z = np.stack([record.z_a for record in train_recs])
    tid_train = np.asarray([record.tid for record in train_recs], dtype=object)
    direction_bank, _ = build_lraes_direction_bank(
        X_train_z,
        y_train,
        k_dir=args.k_dir,
        fisher_cfg=FisherPIAConfig(),
        lraes_cfg=LRAESConfig(),
    )
    effective_k = int(direction_bank.shape[0])
    gamma_budget = np.full((effective_k,), float(args.pia_gamma), dtype=np.float64)
    tier_ratios = parse_step_tier_ratios(args.wavelet_step_tier_ratios)
    eta_safe = None if args.disable_safe_step else 0.5

    z_aug, y_aug, tid_aug, dir_aug, aug_meta = build_step_tier_candidates(
        X_train_z,
        y_train,
        tid_train,
        direction_bank=direction_bank,
        gamma_by_dir=gamma_budget,
        tier_ratios=tier_ratios,
        seed=seed + 811,
        eta_safe=eta_safe,
    )
    z_dm_aug = None
    effective_k_dm = 0
    if object_mode == "dual_a_dm":
        if mean_log_dm is None:
            raise ValueError("dual_a_dm requires a valid cD_m mean log covariance.")
        if any(record.z_dm is None for record in train_recs):
            raise ValueError("dual_a_dm requires z_dm for every wavelet trial record.")
        X_train_z_dm = np.stack([np.asarray(record.z_dm) for record in train_recs])
        direction_bank_dm, _ = build_lraes_direction_bank(
            X_train_z_dm,
            y_train,
            k_dir=args.k_dir,
            fisher_cfg=FisherPIAConfig(),
            lraes_cfg=LRAESConfig(),
        )
        effective_k_dm = int(direction_bank_dm.shape[0])
        gamma_budget_dm = np.full(
            (effective_k_dm,),
            float(args.pia_gamma) * float(args.wavelet_detail_gamma_scale),
            dtype=np.float64,
        )
        z_dm_aug, y_dm_aug, tid_dm_aug, _, aug_meta_dm = build_step_tier_candidates(
            X_train_z_dm,
            y_train,
            tid_train,
            direction_bank=direction_bank_dm,
            gamma_by_dir=gamma_budget_dm,
            tier_ratios=tier_ratios,
            seed=seed + 1811,
            eta_safe=eta_safe,
        )
        if len(z_dm_aug) != len(z_aug) or not np.array_equal(y_dm_aug, y_aug) or not np.array_equal(tid_dm_aug, tid_aug):
            raise AssertionError("dual_a_dm generated misaligned cA/cD_m candidate streams.")
        combined_rows = []
        for row_a, row_dm in zip(aug_meta.get("candidate_rows", []), aug_meta_dm.get("candidate_rows", [])):
            combined = {
                "anchor_index": int(row_a["anchor_index"]),
                "tid": row_a["tid"],
                "tier_ratio": float(row_a["tier_ratio"]),
                "tier_label": row_a["tier_label"],
                "cA_direction_id": int(row_a["direction_id"]),
                "cA_sign": float(row_a["sign"]),
                "cA_safe_upper_bound": float(row_a["safe_upper_bound"]),
                "cA_gamma_used": float(row_a["gamma_used"]),
                "safe_radius_ratio_A": float(row_a["safe_radius_ratio"]),
                "cDm_direction_id": int(row_dm["direction_id"]),
                "cDm_sign": float(row_dm["sign"]),
                "cDm_safe_upper_bound": float(row_dm["safe_upper_bound"]),
                "cDm_gamma_used": float(row_dm["gamma_used"]),
                "safe_radius_ratio_Dm": float(row_dm["safe_radius_ratio"]),
                "safe_radius_ratio": 0.5
                * (float(row_a["safe_radius_ratio"]) + float(row_dm["safe_radius_ratio"])),
            }
            combined_rows.append(combined)
        aug_meta["candidate_rows"] = combined_rows
        aug_meta["safe_radius_ratio_mean"] = 0.5 * (
            float(aug_meta.get("safe_radius_ratio_mean", 1.0))
            + float(aug_meta_dm.get("safe_radius_ratio_mean", 1.0))
        )
    tid_to_rec = {record.tid: record for record in train_recs}
    realized = realize_wavelet_candidates(
        z_aug=z_aug,
        y_aug=y_aug,
        tid_aug=tid_aug,
        candidate_rows=aug_meta.get("candidate_rows", []),
        tid_to_rec=tid_to_rec,
        mean_log=mean_log_a,
        wavelet_name=args.wavelet_name,
        wavelet_mode=args.wavelet_mode,
        object_mode=object_mode,
        z_dm_aug=z_dm_aug,
        mean_log_dm=mean_log_dm,
    )
    identity_meta = compute_identity_checks(
        train_recs,
        wavelet_name=args.wavelet_name,
        wavelet_mode=args.wavelet_mode,
        object_mode=object_mode,
    )
    return {
        "effective_k": effective_k,
        "effective_k_dm": effective_k_dm,
        "z_aug": z_aug,
        "y_aug": y_aug,
        "tid_aug": tid_aug,
        "dir_aug": dir_aug,
        "tid_to_rec": tid_to_rec,
        "aug_trials": realized["aug_trials"],
        "X_aug_raw": realized["X_aug_raw"],
        "y_aug_np": realized["y_aug_np"],
        "avg_bridge": realized["avg_bridge"],
        "audit_rows": realized["audit_rows"],
        "safe_radius_ratio_mean": aug_meta.get("safe_radius_ratio_mean", 1.0),
        "safe_radius_ratio_min": aug_meta.get("safe_radius_ratio_min", 1.0),
        "manifold_margin_mean": aug_meta.get("manifold_margin_mean", 0.0),
        "candidate_total_count": int(aug_meta.get("aug_total_count", len(realized["aug_trials"]))),
        "aug_total_count": int(aug_meta.get("aug_total_count", len(realized["aug_trials"]))),
        "step_tier_count": int(aug_meta.get("step_tier_count", len(tier_ratios))),
        **identity_meta,
    }


def _run_wavelet_mba_pipeline(
    *,
    args,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    train_recs: List[WaveletTrialRecord],
    wavelet_meta: Dict[str, object],
    mean_log_a: np.ndarray,
    mean_log_dm: Optional[np.ndarray],
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> Dict[str, object]:
    aug_out = _build_wavelet_mba_realized_augmentations(
        args=args,
        seed=seed,
        y_train=y_train,
        train_recs=train_recs,
        mean_log_a=mean_log_a,
        mean_log_dm=mean_log_dm,
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
        return_model_obj=False,
        loader_seed=seed,
    )

    print(f"Fitting wavelet_mba weighted CE model ({len(X_train_raw)} orig + {aug_out['aug_total_count']} aug)...")
    res_act = _fit_host_model_weighted_aug_ce(
        args=args,
        X_tr=X_train_raw,
        y_tr=y_train,
        X_aug=aug_out["X_aug_raw"],
        y_aug=aug_out["y_aug_np"],
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        loader_seed=seed,
    )

    return {
        "res_base": res_base,
        "res_act": res_act,
        "avg_bridge": aug_out["avg_bridge"],
        "safe_radius_ratio_mean": aug_out["safe_radius_ratio_mean"],
        "manifold_margin_mean": aug_out["manifold_margin_mean"],
        "candidate_total_count": aug_out["candidate_total_count"],
        "aug_total_count": aug_out["aug_total_count"],
        "step_tier_count": aug_out["step_tier_count"],
        "effective_k": aug_out["effective_k"],
        "effective_k_dm": aug_out["effective_k_dm"],
        "wavelet_meta": wavelet_meta,
        "cA_identity_bridge_error_mean": aug_out["cA_identity_bridge_error_mean"],
        "idwt_identity_recon_error_mean": aug_out["idwt_identity_recon_error_mean"],
        "audit_rows": aug_out["audit_rows"],
        "viz_payload": {
            "Z_orig": np.stack([record.z_a for record in train_recs]),
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
                "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
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
            wavelet_meta: Dict[str, object] = {}
            if args.pipeline == "wavelet_mba":
                wavelet_train_recs, mean_log_a, wavelet_meta = build_wavelet_trial_records(
                    train_trials,
                    wavelet_name=args.wavelet_name,
                    wavelet_level=args.wavelet_level,
                    wavelet_mode=args.wavelet_mode,
                    secondary_detail_level=args.wavelet_secondary_detail_level,
                )
                if mean_log_a is None:
                    raise ValueError("wavelet_mba could not build a cA mean log covariance.")
                mean_log_dm = wavelet_meta.get("mean_log_dm")
                if args.wavelet_object_mode == "dual_a_dm":
                    if int(wavelet_meta.get("wavelet_level_eff", 0)) != 2:
                        raise ValueError("dual_a_dm V2 requires wavelet_level_eff == 2.")
                    if int(args.wavelet_secondary_detail_level) != 2:
                        raise ValueError("dual_a_dm V2 keeps cD_1 frozen and only supports cD_2 as the secondary object.")
                    if float(wavelet_meta.get("cDm_energy_mean", 0.0)) <= 1e-10:
                        raise ValueError("dual_a_dm skipped because cD_2 energy is near zero.")
                pipeline_out = _run_wavelet_mba_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    train_recs=wavelet_train_recs,
                    wavelet_meta=wavelet_meta,
                    mean_log_a=mean_log_a,
                    mean_log_dm=mean_log_dm,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            elif args.pipeline == "mba_white_edit":
                pipeline_out = _run_mba_white_edit_pipeline(
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
            elif args.pipeline == "mba_feedback" and args.algo == "adaptive":
                pipeline_out = _run_mba_feedback_adaptive_pipeline(
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
            elif args.pipeline == "mba_feedback":
                pipeline_out = _run_mba_feedback_pipeline(
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
            else:
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
                "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
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
            direction_meta = dict(pipeline_out.get("direction_bank_meta", {}))
            summary["direction_bank_source"] = str(direction_meta.get("bank_source", args.algo))
            if direction_meta.get("bank_source") == "zpia_telm2" or args.algo == "zpia":
                summary.update(
                    {
                        "zpia_z_dim": int(direction_meta.get("z_dim", 0)),
                        "zpia_n_train": int(direction_meta.get("n_train", 0)),
                        "zpia_n_train_lt_z_dim": bool(direction_meta.get("n_train_lt_z_dim", False)),
                        "zpia_row_norm_min": float(direction_meta.get("row_norm_min", 0.0)),
                        "zpia_row_norm_max": float(direction_meta.get("row_norm_max", 0.0)),
                        "zpia_row_norm_mean": float(direction_meta.get("row_norm_mean", 0.0)),
                        "zpia_fallback_row_count": int(direction_meta.get("fallback_row_count", 0)),
                        "telm2_recon_last": float(direction_meta.get("telm2_recon_last", 0.0)),
                        "telm2_recon_mean": float(direction_meta.get("telm2_recon_mean", 0.0)),
                        "telm2_recon_std": float(direction_meta.get("telm2_recon_std", 0.0)),
                        "telm2_n_iters": int(direction_meta.get("telm2_n_iters", args.telm2_n_iters)),
                        "telm2_c_repr": float(direction_meta.get("telm2_c_repr", args.telm2_c_repr)),
                        "telm2_activation": str(direction_meta.get("telm2_activation", args.telm2_activation)),
                        "telm2_bias_update_mode": str(
                            direction_meta.get("telm2_bias_update_mode", args.telm2_bias_update_mode)
                        ),
                    }
                )
            if args.pipeline == "mba_feedback":
                summary.update(
                    {
                        "feedback_margin_temperature": float(args.feedback_margin_temperature),
                        "aug_loss_weight": float(args.aug_loss_weight),
                        "feedback_weight_mean": float(res_act.get("feedback_weight_mean", 0.0)),
                        "feedback_weight_std": float(res_act.get("feedback_weight_std", 0.0)),
                        "last_orig_ce_loss": float(res_act.get("last_orig_ce_loss", 0.0)),
                        "last_weighted_aug_ce_loss": float(res_act.get("last_weighted_aug_ce_loss", 0.0)),
                        "last_aug_margin_mean": float(res_act.get("last_aug_margin_mean", 0.0)),
                    }
                )
                if args.algo == "adaptive":
                    summary.update(
                        {
                            "router_temperature": float(args.router_temperature),
                            "router_min_prob": float(args.router_min_prob),
                            "router_smoothing": float(args.router_smoothing),
                            "router_reward": str(args.router_reward),
                            "router_p_lraes_final": float(res_act.get("router_p_lraes_final", 0.0)),
                            "router_p_zpia_final": float(res_act.get("router_p_zpia_final", 0.0)),
                            "router_reward_lraes_last": float(res_act.get("router_reward_lraes_last", 0.0)),
                            "router_reward_zpia_last": float(res_act.get("router_reward_zpia_last", 0.0)),
                            "adaptive_best_engine_final": str(res_act.get("adaptive_best_engine_final", "")),
                            "effective_k_dir_lraes": int(pipeline_out.get("effective_k_lraes", 0)),
                            "effective_k_dir_zpia": int(pipeline_out.get("effective_k_zpia", 0)),
                            "feedback_weight_mean_lraes": float(res_act.get("feedback_weight_mean_lraes", 0.0)),
                            "feedback_weight_mean_zpia": float(res_act.get("feedback_weight_mean_zpia", 0.0)),
                            "last_weighted_aug_ce_loss_lraes": float(
                                res_act.get("last_weighted_aug_ce_loss_lraes", 0.0)
                            ),
                            "last_weighted_aug_ce_loss_zpia": float(
                                res_act.get("last_weighted_aug_ce_loss_zpia", 0.0)
                            ),
                            "last_aug_margin_mean_lraes": float(res_act.get("last_aug_margin_mean_lraes", 0.0)),
                            "last_aug_margin_mean_zpia": float(res_act.get("last_aug_margin_mean_zpia", 0.0)),
                            "adaptive_engine_sources": ",".join(
                                [str(x) for x in direction_meta.get("engine_sources", [])]
                            ),
                        }
                    )
                audit_rows = pipeline_out.get("audit_rows", [])
                if audit_rows:
                    audit_dir = os.path.join(args.out_root, "audit")
                    os.makedirs(audit_dir, exist_ok=True)
                    pd.DataFrame(audit_rows).to_csv(
                        os.path.join(audit_dir, f"{dataset_name}_s{seed}_{args.algo}_candidates.csv"),
                        index=False,
                    )
                if args.algo == "adaptive":
                    router_trace = pipeline_out.get("router_trace", [])
                    if router_trace:
                        trace_dir = os.path.join(args.out_root, "router")
                        os.makedirs(trace_dir, exist_ok=True)
                        pd.DataFrame(router_trace).to_csv(
                            os.path.join(trace_dir, f"{dataset_name}_s{seed}_router_trace.csv"),
                            index=False,
                        )
            if args.pipeline == "wavelet_mba":
                cD_frozen_count = int(wavelet_meta.get("cD_frozen_count", 0))
                if args.wavelet_object_mode == "dual_a_dm":
                    cD_frozen_count = max(0, cD_frozen_count - 1)
                summary.update(
                    {
                        "wavelet_object_mode": args.wavelet_object_mode,
                        "wavelet_name": args.wavelet_name,
                        "wavelet_level": args.wavelet_level,
                        "wavelet_level_eff": int(wavelet_meta.get("wavelet_level_eff", 0)),
                        "wavelet_secondary_detail_level": int(args.wavelet_secondary_detail_level),
                        "wavelet_detail_gamma_scale": float(args.wavelet_detail_gamma_scale),
                        "cA_length": int(wavelet_meta.get("cA_length", 0)),
                        "cDm_length": int(wavelet_meta.get("cDm_length", 0)),
                        "cDm_energy_mean": float(wavelet_meta.get("cDm_energy_mean", 0.0)),
                        "cD_frozen_count": cD_frozen_count,
                        "wavelet_recon_error_mean": float(wavelet_meta.get("wavelet_recon_error_mean", 0.0)),
                        "cA_identity_bridge_error_mean": float(pipeline_out.get("cA_identity_bridge_error_mean", 0.0)),
                        "cDm_identity_bridge_error_mean": float(pipeline_out.get("cDm_identity_bridge_error_mean", 0.0)),
                        "idwt_identity_recon_error_mean": float(pipeline_out.get("idwt_identity_recon_error_mean", 0.0)),
                        "cA_transport_error_fro_mean": float(avg_bridge.get("cA_transport_error_fro", 0.0)),
                        "cA_transport_error_logeuc_mean": float(avg_bridge.get("cA_transport_error_logeuc", 0.0)),
                        "cDm_transport_error_fro_mean": float(avg_bridge.get("cDm_transport_error_fro", 0.0)),
                        "cDm_transport_error_logeuc_mean": float(avg_bridge.get("cDm_transport_error_logeuc", 0.0)),
                        "dual_transport_error_logeuc_mean": float(avg_bridge.get("dual_transport_error_logeuc", 0.0)),
                        "step_tier_count": int(pipeline_out.get("step_tier_count", 0)),
                        "effective_k_dir_dm": int(pipeline_out.get("effective_k_dm", 0)),
                        "feedback_weight_mean": float(res_act.get("feedback_weight_mean", 0.0)),
                        "feedback_weight_std": float(res_act.get("feedback_weight_std", 0.0)),
                        "last_orig_ce_loss": float(res_act.get("last_orig_ce_loss", 0.0)),
                        "last_weighted_aug_ce_loss": float(res_act.get("last_weighted_aug_ce_loss", 0.0)),
                        "last_aug_margin_mean": float(res_act.get("last_aug_margin_mean", 0.0)),
                    }
                )
                audit_rows = pipeline_out.get("audit_rows", [])
                if audit_rows:
                    audit_dir = os.path.join(args.out_root, "audit")
                    os.makedirs(audit_dir, exist_ok=True)
                    pd.DataFrame(audit_rows).to_csv(
                        os.path.join(audit_dir, f"{dataset_name}_s{seed}_wavelet_candidates.csv"),
                        index=False,
                    )
            if args.pipeline == "mba_white_edit":
                summary.update(
                    {
                        "edit_mode": args.edit_mode,
                        "edit_basis": args.edit_basis,
                        "edit_alpha_scale": float(args.edit_alpha_scale),
                        "white_identity_error_mean": float(pipeline_out.get("white_identity_error_mean", 0.0)),
                        "recolor_transport_error_logeuc_mean": float(
                            avg_bridge.get("recolor_transport_error_logeuc", 0.0)
                        ),
                        "recolor_transport_error_fro_mean": float(avg_bridge.get("recolor_transport_error_fro", 0.0)),
                        "edit_energy_mean": float(avg_bridge.get("edit_energy", 0.0)),
                        "edit_norm_mean": float(avg_bridge.get("edit_norm", 0.0)),
                        "edit_alpha_mean": float(avg_bridge.get("edit_alpha", 0.0)),
                        "feedback_weight_mean": float(res_act.get("feedback_weight_mean", 0.0)),
                        "feedback_weight_std": float(res_act.get("feedback_weight_std", 0.0)),
                        "last_orig_ce_loss": float(res_act.get("last_orig_ce_loss", 0.0)),
                        "last_weighted_aug_ce_loss": float(res_act.get("last_weighted_aug_ce_loss", 0.0)),
                        "last_aug_margin_mean": float(res_act.get("last_aug_margin_mean", 0.0)),
                    }
                )
                audit_rows = pipeline_out.get("audit_rows", [])
                if audit_rows:
                    audit_dir = os.path.join(args.out_root, "audit")
                    os.makedirs(audit_dir, exist_ok=True)
                    pd.DataFrame(audit_rows).to_csv(
                        os.path.join(audit_dir, f"{dataset_name}_s{seed}_white_edit_candidates.csv"),
                        index=False,
                    )
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
                    Z_orig=pipeline_out["viz_payload"].get("Z_orig", X_train_z),
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
                    "pipeline": "act" if args.pipeline == "mba" else args.pipeline,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="ACT_ManifoldBridge original ACT runner")
    parser.add_argument("--dataset", type=str, default="natops")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--pipeline", type=str, choices=["act", "mba", "mba_feedback", "wavelet_mba", "mba_white_edit"], default="act")
    parser.add_argument("--algo", type=str, choices=["pia", "lraes", "zpia", "adaptive"], default="lraes")
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
    parser.add_argument("--wavelet-name", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=str, default="auto")
    parser.add_argument("--wavelet-mode", type=str, default="symmetric")
    parser.add_argument("--wavelet-object-mode", type=str, choices=["ca_only", "dual_a_dm"], default="ca_only")
    parser.add_argument("--wavelet-secondary-detail-level", type=int, default=2)
    parser.add_argument("--wavelet-detail-gamma-scale", type=float, default=0.5)
    parser.add_argument("--wavelet-step-tier-ratios", type=str, default="0.25,0.5,0.9")
    parser.add_argument("--edit-mode", type=str, choices=["line"], default="line")
    parser.add_argument("--edit-basis", type=str, choices=["whitened_pca"], default="whitened_pca")
    parser.add_argument("--edit-alpha-scale", type=float, default=0.25)
    parser.add_argument("--feedback-margin-temperature", type=float, default=1.0)
    parser.add_argument("--aug-loss-weight", type=float, default=1.0)
    parser.add_argument("--telm2-n-iters", type=int, default=3)
    parser.add_argument("--telm2-c-repr", type=float, default=1.0)
    parser.add_argument("--telm2-activation", type=str, choices=["sine", "sigmoid"], default="sine")
    parser.add_argument("--telm2-bias-update-mode", type=str, choices=["off", "act_mean", "residual"], default="residual")
    parser.add_argument("--router-temperature", type=float, default=0.05)
    parser.add_argument("--router-min-prob", type=float, default=0.10)
    parser.add_argument("--router-smoothing", type=float, default=0.5)
    parser.add_argument("--router-reward", type=str, choices=["feedback_weight"], default="feedback_weight")
    parser.add_argument("--out-root", type=str, default="standalone_projects/ACT_ManifoldBridge/results/act_core")
    args = parser.parse_args()

    if args.pipeline == "mba":
        print("Using legacy pipeline alias 'mba' -> 'act'.")
    if args.feedback_margin_temperature <= 0.0:
        raise ValueError("--feedback-margin-temperature must be positive.")
    if args.aug_loss_weight < 0.0:
        raise ValueError("--aug-loss-weight must be non-negative.")
    if args.algo == "zpia":
        if args.k_dir <= 0:
            raise ValueError("--algo zpia requires --k-dir > 0.")
        if args.telm2_c_repr <= 0.0:
            raise ValueError("--telm2-c-repr must be positive.")
        if args.telm2_n_iters < 0:
            raise ValueError("--telm2-n-iters must be non-negative.")
    if args.algo == "adaptive":
        if args.pipeline != "mba_feedback":
            raise ValueError("--algo adaptive currently supports --pipeline mba_feedback only.")
        if args.model != "resnet1d":
            raise ValueError("--algo adaptive v1 supports --model resnet1d only.")
        if args.router_temperature <= 0.0:
            raise ValueError("--router-temperature must be positive.")
        if not (0.0 <= args.router_min_prob < 0.5):
            raise ValueError("--router-min-prob must satisfy 0 <= value < 0.5.")
        if not (0.0 <= args.router_smoothing <= 1.0):
            raise ValueError("--router-smoothing must satisfy 0 <= value <= 1.")
    if args.pipeline == "mba_feedback" and args.model == "minirocket":
        raise ValueError("--pipeline mba_feedback supports resnet1d, patchtst, and timesnet only.")
    if args.pipeline == "wavelet_mba":
        if args.algo != "lraes":
            raise ValueError("--pipeline wavelet_mba currently supports --algo lraes only.")
        if args.model == "minirocket":
            raise ValueError("--pipeline wavelet_mba supports resnet1d, patchtst, and timesnet only.")
        if args.wavelet_detail_gamma_scale <= 0.0:
            raise ValueError("--wavelet-detail-gamma-scale must be positive.")
        if args.wavelet_object_mode == "dual_a_dm" and args.wavelet_secondary_detail_level != 2:
            raise ValueError("--wavelet-object-mode dual_a_dm V2 only supports --wavelet-secondary-detail-level 2.")
    if args.pipeline == "mba_white_edit":
        if args.algo != "lraes":
            raise ValueError("--pipeline mba_white_edit currently supports --algo lraes only.")
        if args.model != "resnet1d":
            raise ValueError("--pipeline mba_white_edit v1 supports --model resnet1d only.")
        if args.edit_alpha_scale < 0.0:
            raise ValueError("--edit-alpha-scale must be non-negative.")

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
