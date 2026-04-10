#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_ATRIALFIBRILLATION_ROOT,
    DEFAULT_BASICMOTIONS_ROOT,
    DEFAULT_EPILEPSY_ROOT,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_MITBIH_NPZ,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_PENDIGITS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    DEFAULT_SEEDIV_ROOT,
    DEFAULT_SEEDV_ROOT,
    DEFAULT_UWAVEGESTURELIBRARY_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
)
from run_bridge_curriculum_pilot import (  # noqa: E402
    _bridge_aug_trials,
    _dataset_title,
    _ensure_dir,
    _fit_raw_minirocket,
    _format_mean_std,
    _records_to_trial_dicts,
    _risk_comment,
    _select_best_curriculum_round,
    _write_json,
)
from run_phase15_multiround_curriculum_probe import (  # noqa: E402
    _active_direction_probs,
    _build_curriculum_aug_candidates,
    _compute_direction_intrusion,
    _mech_dir_maps,
    _update_direction_budget,
)
from run_phase15_step1a_maxplane import _fit_eval_linearsvc  # noqa: E402
from run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
)
from run_raw_bridge_probe import (  # noqa: E402
    TrialRecord,
    _apply_mean_log,
    _build_trial_records,
)
from scripts.protocol_split_utils import (  # noqa: E402
    MITBIH_DATASETS,
    OFFICIAL_FIXED_SPLIT_DATASETS,
    SEED_NATIVE_DATASETS,
    resolve_inner_train_val_split,
    resolve_protocol_split,
)
from scripts.run_phase15_mainline_freeze import _summarize_dir_profile  # noqa: E402


MODE_PREFIX = {
    "train_replace": "augmentation_train_replace",
    "best_round_pool": "augmentation_best_round_pool",
    "filtered_pool": "augmentation_filtered_pool",
    "seed_light_best_round": "seed_family_bridge_train_replace",
}


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _summary_stats(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _stack_feature(records: Sequence[TrialRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.asarray([], dtype=object),
        )
    return (
        np.stack([np.asarray(r.z, dtype=np.float32) for r in records], axis=0),
        np.asarray([int(r.y) for r in records], dtype=np.int64),
        np.asarray([r.tid for r in records], dtype=object),
    )


def _dataset_group(dataset: str) -> str:
    ds = normalize_dataset_name(dataset)
    if ds in OFFICIAL_FIXED_SPLIT_DATASETS:
        return "fixed_split_main"
    if ds in MITBIH_DATASETS:
        return "mitbih_extension"
    if ds in SEED_NATIVE_DATASETS:
        return "seed_family_light"
    return "other"


def _protocol_label(split_meta: Dict[str, object]) -> str:
    ptype = str(split_meta.get("protocol_type", "")).strip()
    if ptype:
        return ptype
    return "unknown_protocol"


def _make_raw_eval_args(args: argparse.Namespace, seed_out_root: str) -> SimpleNamespace:
    return SimpleNamespace(
        out_root=str(seed_out_root),
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        prop_win_ratio=float(args.prop_win_ratio),
        prop_hop_ratio=float(args.prop_hop_ratio),
        min_window_len_samples=int(args.min_window_len_samples),
        min_hop_len_samples=int(args.min_hop_len_samples),
        nominal_cap_k=int(args.nominal_cap_k),
        cap_sampling_policy=str(args.cap_sampling_policy),
        aggregation_mode=str(args.aggregation_mode),
        n_kernels=int(args.n_kernels),
        n_jobs=int(args.n_jobs),
        memmap_threshold_gb=float(args.memmap_threshold_gb),
    )


def _build_round_rows(
    *,
    dataset: str,
    seed: int,
    direction_bank: np.ndarray,
    X_train_base: np.ndarray,
    y_train_base: np.ndarray,
    tid_train: np.ndarray,
    X_eval: np.ndarray | None,
    y_eval: np.ndarray | None,
    tid_eval: np.ndarray | None,
    args: argparse.Namespace,
    seed_offset: int,
    progress_tag: str,
) -> List[Dict[str, object]]:
    round_rows: List[Dict[str, object]] = []
    gamma_by_dir = np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64)
    has_eval = X_eval is not None and y_eval is not None and tid_eval is not None and len(y_eval) > 0
    for round_id in range(1, int(args.curriculum_rounds) + 1):
        direction_probs = _active_direction_probs(gamma_by_dir, freeze_eps=float(args.curriculum_freeze_eps))
        gamma_before = gamma_by_dir.copy()
        X_curr, y_curr, tid_curr, src_curr, dir_curr, curr_aug_meta = _build_curriculum_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            direction_bank=direction_bank,
            direction_probs=direction_probs,
            gamma_by_dir=gamma_before,
            multiplier=int(args.pia_multiplier),
            seed=int(seed + seed_offset + round_id * 1009),
        )
        mech_curr = _compute_mech_metrics(
            X_train_real=X_train_base,
            y_train_real=y_train_base,
            X_aug_generated=X_curr,
            y_aug_generated=y_curr,
            X_aug_accepted=X_curr,
            y_aug_accepted=y_curr,
            X_src_accepted=src_curr,
            dir_generated=dir_curr,
            dir_accepted=dir_curr,
            seed=int(seed),
            linear_c=float(args.linear_c),
            class_weight=str(args.linear_class_weight),
            linear_max_iter=int(args.linear_max_iter),
            knn_k=int(args.mech_knn_k),
            max_aug_for_mech=int(args.mech_max_aug_for_metrics),
            max_real_knn_ref=int(args.mech_max_real_knn_ref),
            max_real_knn_query=int(args.mech_max_real_knn_query),
            progress_prefix=f"[route-b-aug][{dataset}][seed={seed}][{progress_tag}-r{round_id}-mech]",
        )
        intrusion_by_dir = _compute_direction_intrusion(
            X_anchor=X_train_base,
            y_anchor=y_train_base,
            X_aug_accepted=X_curr,
            y_aug_accepted=y_curr,
            dir_accepted=dir_curr,
            seed=int(seed),
            knn_k=int(args.mech_knn_k),
            max_eval=int(args.mech_max_aug_for_metrics),
        )
        maps = _mech_dir_maps(mech_curr, intrusion_by_dir=intrusion_by_dir)
        gamma_after, state_by_dir, score_by_dir = _update_direction_budget(
            gamma_before=gamma_before,
            margin_by_dir=maps["margin_drop_median"],
            flip_by_dir=maps["flip_rate"],
            intrusion_by_dir=maps["intrusion"],
            expand_factor=float(args.curriculum_expand_factor),
            shrink_factor=float(args.curriculum_shrink_factor),
            gamma_max=float(args.curriculum_gamma_max),
            freeze_eps=float(args.curriculum_freeze_eps),
        )
        z_trial_macro_f1 = np.nan
        z_window_macro_f1 = np.nan
        if has_eval:
            X_train_curr = np.vstack([X_train_base, X_curr]) if len(y_curr) else X_train_base.copy()
            y_train_curr = np.concatenate([y_train_base, y_curr]) if len(y_curr) else y_train_base.copy()
            tid_train_curr = np.concatenate([tid_train, tid_curr]) if len(y_curr) else tid_train.copy()
            is_aug_curr = (
                np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_curr),), dtype=bool)])
                if len(y_curr)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            z_metrics, _ = _fit_eval_linearsvc(
                X_train_curr,
                y_train_curr,
                tid_train_curr,
                X_eval,
                y_eval,
                tid_eval,
                seed=int(seed),
                cap_k=int(args.nominal_cap_k),
                cap_seed=int(seed + 41),
                cap_sampling_policy="balanced_real_aug",
                linear_c=float(args.linear_c),
                class_weight=str(args.linear_class_weight),
                max_iter=int(args.linear_max_iter),
                agg_mode="majority",
                is_aug_train=is_aug_curr,
                progress_prefix=f"[route-b-aug][{dataset}][seed={seed}][{progress_tag}-r{round_id}-fit]",
            )
            z_trial_macro_f1 = float(z_metrics["trial_macro_f1"])
            z_window_macro_f1 = float(z_metrics["window_macro_f1"])
        round_rows.append(
            {
                "round_id": int(round_id),
                "z_aug": X_curr,
                "y_aug": y_curr,
                "tid_aug": tid_curr,
                "mech": mech_curr,
                "dir_summary": _summarize_dir_profile(mech_curr.get("dir_profile", {})),
                "aug_meta": curr_aug_meta,
                "direction_usage_entropy": float(curr_aug_meta.get("direction_usage_entropy", 0.0)),
                "direction_probs": curr_aug_meta.get("direction_probs", {}),
                "gamma_before": {str(i): float(gamma_before[i]) for i in range(len(gamma_before))},
                "gamma_after": {str(i): float(gamma_after[i]) for i in range(len(gamma_after))},
                "direction_state": {str(k): str(v) for k, v in state_by_dir.items()},
                "direction_score": {str(k): float(v) for k, v in score_by_dir.items()},
                "z_trial_macro_f1": float(z_trial_macro_f1) if np.isfinite(z_trial_macro_f1) else np.nan,
                "z_window_macro_f1": float(z_window_macro_f1) if np.isfinite(z_window_macro_f1) else np.nan,
            }
        )
        gamma_by_dir = gamma_after.copy()
    return round_rows


def _attach_bridge_payload(
    *,
    train_records: Sequence[TrialRecord],
    mean_log_train: np.ndarray,
    round_rows: Sequence[Dict[str, object]],
    bridge_eps: float,
    variant_prefix: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in round_rows:
        aug_records, bridge_meta = _bridge_aug_trials(
            train_records=train_records,
            mean_log_train=mean_log_train,
            z_aug=np.asarray(row["z_aug"], dtype=np.float32),
            y_aug=np.asarray(row["y_aug"], dtype=np.int64),
            tid_aug=np.asarray(row["tid_aug"]),
            variant_tag=f"{variant_prefix}_r{int(row['round_id'])}",
            bridge_eps=float(bridge_eps),
        )
        out.append(
            {
                **row,
                "aug_records": aug_records,
                "aug_trials": _records_to_trial_dicts(aug_records),
                "bridge_meta": bridge_meta,
            }
        )
    return out


def _sample_pool_trials(
    *,
    round_payloads: Sequence[Dict[str, object]],
    total_aug_target: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    if total_aug_target <= 0 or not round_payloads:
        return [], {}
    rs = np.random.RandomState(int(seed))
    counts = np.asarray([len(p["aug_trials"]) for p in round_payloads], dtype=np.int64)
    total_available = int(np.sum(counts))
    if total_available <= 0:
        return [], {str(int(p["round_id"])): 0 for p in round_payloads}
    target = min(int(total_aug_target), total_available)
    shares = counts.astype(np.float64) / float(np.sum(counts))
    raw_take = shares * float(target)
    take = np.floor(raw_take).astype(np.int64)
    remainder = int(target - int(np.sum(take)))
    if remainder > 0:
        fractional = raw_take - take.astype(np.float64)
        order = np.argsort(-fractional)
        for idx in order.tolist():
            if remainder <= 0:
                break
            if take[idx] < counts[idx]:
                take[idx] += 1
                remainder -= 1
    sampled: List[Dict[str, object]] = []
    sampled_counts: Dict[str, int] = {}
    for payload, n_take in zip(round_payloads, take.tolist()):
        rid = str(int(payload["round_id"]))
        aug_trials = list(payload["aug_trials"])
        if n_take <= 0:
            sampled_counts[rid] = 0
            continue
        if n_take >= len(aug_trials):
            sampled.extend(aug_trials)
            sampled_counts[rid] = int(len(aug_trials))
            continue
        sel_idx = rs.choice(len(aug_trials), size=int(n_take), replace=False)
        sampled.extend([aug_trials[int(i)] for i in np.asarray(sel_idx, dtype=np.int64).tolist()])
        sampled_counts[rid] = int(n_take)
    return sampled, sampled_counts


def _build_train_variant_trials(
    *,
    train_raw_trials: Sequence[Dict[str, object]],
    single_payload: Dict[str, object] | None,
    final_payload: Dict[str, object] | None,
    best_payload: Dict[str, object] | None,
    pool_payloads: Sequence[Dict[str, object]],
    filtered_payloads: Sequence[Dict[str, object]],
    seed: int,
) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, object]]:
    raw_trials = list(train_raw_trials)
    out: Dict[str, List[Dict[str, object]]] = {"raw": raw_trials}
    meta: Dict[str, object] = {
        "best_round": None if best_payload is None else int(best_payload["round_id"]),
        "final_round": None if final_payload is None else int(final_payload["round_id"]),
        "pool_rounds": [int(p["round_id"]) for p in pool_payloads],
        "filtered_rounds": [int(p["round_id"]) for p in filtered_payloads],
    }
    if single_payload is not None:
        out["bridge_single_round"] = raw_trials + list(single_payload["aug_trials"])
    if final_payload is not None:
        out["bridge_multiround_final"] = raw_trials + list(final_payload["aug_trials"])
    if best_payload is not None:
        out["bridge_multiround_best_round"] = raw_trials + list(best_payload["aug_trials"])
        out["seed_light_best_round"] = raw_trials + list(best_payload["aug_trials"])
    if pool_payloads:
        sampled, sampled_counts = _sample_pool_trials(
            round_payloads=pool_payloads,
            total_aug_target=int(len(raw_trials)),
            seed=int(seed + 5001),
        )
        out["bridge_best_round_pool"] = raw_trials + sampled
        meta["pool_sample_counts"] = sampled_counts
        meta["pool_size_ratio"] = float(len(sampled) / max(1, len(raw_trials)))
    if filtered_payloads:
        sampled, sampled_counts = _sample_pool_trials(
            round_payloads=filtered_payloads,
            total_aug_target=int(len(raw_trials)),
            seed=int(seed + 7001),
        )
        out["bridge_filtered_pool"] = raw_trials + sampled
        meta["filtered_sample_counts"] = sampled_counts
        meta["filtered_pool_size_ratio"] = float(len(sampled) / max(1, len(raw_trials)))
    return out, meta


def _choose_best_round(selection_rows: Sequence[Dict[str, object]], has_val: bool) -> Tuple[int, str, float]:
    if not selection_rows:
        raise RuntimeError("No multiround rows produced.")
    if not has_val:
        return int(selection_rows[0]["round_id"]), "no_val_default_first_round", float("nan")
    best = _select_best_curriculum_round(list(selection_rows))
    return int(best["round_id"]), "inner_val_best_round", float(best["z_trial_macro_f1"])


def _fit_variant(
    *,
    dataset: str,
    train_trials: Sequence[Dict[str, object]],
    test_trials: Sequence[Dict[str, object]],
    seed: int,
    args: argparse.Namespace,
    stage_seed_dir: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    raw_args = _make_raw_eval_args(args, stage_seed_dir)
    return _fit_raw_minirocket(
        dataset=str(dataset),
        train_trials=train_trials,
        test_trials=test_trials,
        seed=int(seed),
        args=raw_args,
    )


def _variant_rows_to_summary(
    *,
    dataset: str,
    dataset_group: str,
    protocol: str,
    mode: str,
    by_seed_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    df = pd.DataFrame(by_seed_rows)
    if df.empty:
        raise RuntimeError("No by-seed rows to summarize.")
    raw_df = df[df["train_variant"] == "raw"]
    single_df = df[df["train_variant"] == "bridge_single_round"]
    final_df = df[df["train_variant"] == "bridge_multiround_final"]
    best_df = df[df["train_variant"] == "bridge_multiround_best_round"]
    pool_df = df[df["train_variant"] == "bridge_best_round_pool"]
    filt_df = df[df["train_variant"] == "bridge_filtered_pool"]
    seed_best_df = df[df["train_variant"] == "seed_light_best_round"]

    summary = {
        "dataset": dataset,
        "dataset_group": dataset_group,
        "protocol": protocol,
        "mode": mode,
        "seed_count": int(df["seed"].nunique()),
        "raw_minirocket_acc": _format_mean_std(raw_df["test_acc"].tolist()) if len(raw_df) else "",
        "raw_minirocket_macro_f1": _format_mean_std(raw_df["test_macro_f1"].tolist()) if len(raw_df) else "",
        "bridge_single_train_acc": _format_mean_std(single_df["test_acc"].tolist()) if len(single_df) else "",
        "bridge_single_train_macro_f1": _format_mean_std(single_df["test_macro_f1"].tolist()) if len(single_df) else "",
        "bridge_final_train_acc": _format_mean_std(final_df["test_acc"].tolist()) if len(final_df) else "",
        "bridge_final_train_macro_f1": _format_mean_std(final_df["test_macro_f1"].tolist()) if len(final_df) else "",
        "bridge_best_round_train_acc": _format_mean_std(best_df["test_acc"].tolist()) if len(best_df) else _format_mean_std(seed_best_df["test_acc"].tolist()) if len(seed_best_df) else "",
        "bridge_best_round_train_macro_f1": _format_mean_std(best_df["test_macro_f1"].tolist()) if len(best_df) else _format_mean_std(seed_best_df["test_macro_f1"].tolist()) if len(seed_best_df) else "",
        "bridge_best_round_pool_acc": _format_mean_std(pool_df["test_acc"].tolist()) if len(pool_df) else "",
        "bridge_best_round_pool_macro_f1": _format_mean_std(pool_df["test_macro_f1"].tolist()) if len(pool_df) else "",
        "bridge_filtered_pool_acc": _format_mean_std(filt_df["test_acc"].tolist()) if len(filt_df) else "",
        "bridge_filtered_pool_macro_f1": _format_mean_std(filt_df["test_macro_f1"].tolist()) if len(filt_df) else "",
    }
    raw_mean_f1 = float(raw_df["test_macro_f1"].mean()) if len(raw_df) else float("nan")
    variant_priority = [
        "bridge_filtered_pool",
        "bridge_best_round_pool",
        "bridge_multiround_best_round",
        "seed_light_best_round",
        "bridge_multiround_final",
        "bridge_single_round",
        "raw",
    ]
    variant_scores = {
        variant: float(group["test_macro_f1"].mean())
        for variant, group in df.groupby("train_variant", sort=False)
    }
    best_variant = max(
        variant_scores.items(),
        key=lambda kv: (
            float("-inf") if not np.isfinite(kv[1]) else kv[1],
            -variant_priority.index(kv[0]) if kv[0] in variant_priority else -999,
        ),
    )[0]
    best_variant_f1 = float(variant_scores[best_variant])
    delta_vs_raw = float(best_variant_f1 - raw_mean_f1) if np.isfinite(raw_mean_f1) else float("nan")
    if delta_vs_raw >= 0.002:
        label = "positive"
    elif delta_vs_raw <= -0.002:
        label = "negative"
    else:
        label = "flat"
    best_note = ""
    best_note_df = df[df["train_variant"] == best_variant]
    if len(best_note_df):
        best_note = str(best_note_df["note"].iloc[0])
    if best_variant == "raw" or best_note == "raw_reference" or not best_note:
        non_raw_df = df[df["train_variant"] != "raw"]
        if len(non_raw_df):
            best_note = str(non_raw_df["note"].iloc[0])
    summary.update(
        {
            "best_variant": str(best_variant),
            "delta_vs_raw": float(delta_vs_raw) if np.isfinite(delta_vs_raw) else np.nan,
            "augmentation_label": str(label),
            "best_rounds": "|".join(sorted({str(int(v)) for v in df["best_round"].dropna().astype(int).tolist()}))
            if "best_round" in df
            else "",
            "note": str(best_note),
        }
    )
    return summary


def _summary_to_markdown(summary_row: Dict[str, object], *, title: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- `dataset`: `{summary_row['dataset']}`",
            f"- `protocol`: `{summary_row['protocol']}`",
            f"- `mode`: `{summary_row['mode']}`",
            f"- `raw`: `{summary_row['raw_minirocket_macro_f1']}`",
            f"- `single`: `{summary_row.get('bridge_single_train_macro_f1', '')}`",
            f"- `final`: `{summary_row.get('bridge_final_train_macro_f1', '')}`",
            f"- `best_round`: `{summary_row.get('bridge_best_round_train_macro_f1', '')}`",
            f"- `best_round_pool`: `{summary_row.get('bridge_best_round_pool_macro_f1', '')}`",
            f"- `filtered_pool`: `{summary_row.get('bridge_filtered_pool_macro_f1', '')}`",
            f"- `best_variant`: `{summary_row['best_variant']}`",
            f"- `delta_vs_raw`: `{float(summary_row['delta_vs_raw']):+.6f}`",
            f"- `augmentation_label`: `{summary_row['augmentation_label']}`",
            f"- `note`: `{summary_row['note']}`",
            "",
        ]
    )


def _load_trials_for_runner(dataset_name: str, args: argparse.Namespace) -> List[Dict]:
    return load_trials_for_dataset(
        dataset=dataset_name,
        har_root=args.har_root,
        mitbih_npz=args.mitbih_npz,
        natops_root=args.natops_root,
        fingermovements_root=args.fingermovements_root,
        selfregulationscp1_root=args.selfregulationscp1_root,
        basicmotions_root=args.basicmotions_root,
        handmovementdirection_root=args.handmovementdirection_root,
        uwavegesturelibrary_root=args.uwavegesturelibrary_root,
        epilepsy_root=args.epilepsy_root,
        atrialfibrillation_root=args.atrialfibrillation_root,
        pendigits_root=args.pendigits_root,
        processed_root=args.processed_root,
        stim_xlsx=args.stim_xlsx,
        seediv_root=args.seediv_root,
        seedv_root=args.seedv_root,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Route B augmentation-style MiniROCKET training runner.")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "har",
            "mitbih",
            "natops",
            "selfregulationscp1",
            "fingermovements",
            "basicmotions",
            "handmovementdirection",
            "uwavegesturelibrary",
            "epilepsy",
            "atrialfibrillation",
            "pendigits",
            "seed1",
            "seediv",
            "seedv",
        ],
    )
    p.add_argument(
        "--mode",
        type=str,
        default="train_replace",
        choices=["train_replace", "best_round_pool", "filtered_pool", "seed_light_best_round"],
    )
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-fallback-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bridge-eps", type=float, default=1e-4)
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--prop-win-ratio", type=float, default=0.5)
    p.add_argument("--prop-hop-ratio", type=float, default=0.25)
    p.add_argument("--min-window-len-samples", type=int, default=16)
    p.add_argument("--min-hop-len-samples", type=int, default=8)
    p.add_argument("--nominal-cap-k", type=int, default=120)
    p.add_argument("--cap-sampling-policy", type=str, default="random")
    p.add_argument("--aggregation-mode", type=str, default="majority")
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--memmap-threshold-gb", type=float, default=1.0)
    p.add_argument("--k-dir", type=int, default=5)
    p.add_argument("--subset-size", type=int, default=1)
    p.add_argument("--pia-multiplier", type=int, default=1)
    p.add_argument("--pia-gamma", type=float, default=0.10)
    p.add_argument("--pia-n-iters", type=int, default=2)
    p.add_argument("--pia-activation", type=str, default="sine")
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--mech-knn-k", type=int, default=20)
    p.add_argument("--mech-max-aug-for-metrics", type=int, default=2000)
    p.add_argument("--mech-max-real-knn-ref", type=int, default=10000)
    p.add_argument("--mech-max-real-knn-query", type=int, default=1000)
    p.add_argument("--linear-c", type=float, default=1.0)
    p.add_argument("--linear-class-weight", type=str, default="none")
    p.add_argument("--linear-max-iter", type=int, default=1000)
    p.add_argument("--curriculum-rounds", type=int, default=3)
    p.add_argument("--curriculum-init-gamma", type=float, default=0.06)
    p.add_argument("--curriculum-expand-factor", type=float, default=1.25)
    p.add_argument("--curriculum-shrink-factor", type=float, default=0.70)
    p.add_argument("--curriculum-gamma-max", type=float, default=0.16)
    p.add_argument("--curriculum-freeze-eps", type=float, default=0.02)
    p.add_argument("--pool-target-ratio", type=float, default=1.0)
    p.add_argument("--filter-z-gap", type=float, default=0.003)
    p.add_argument("--filter-bridge-dist-mult", type=float, default=1.10)
    p.add_argument("--filter-cond-mult", type=float, default=1.10)
    p.add_argument("--filter-mean-shift-mult", type=float, default=1.25)
    p.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    p.add_argument("--mitbih-npz", type=str, default=DEFAULT_MITBIH_NPZ)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    p.add_argument("--basicmotions-root", type=str, default=DEFAULT_BASICMOTIONS_ROOT)
    p.add_argument("--handmovementdirection-root", type=str, default=DEFAULT_HANDMOVEMENTDIRECTION_ROOT)
    p.add_argument("--uwavegesturelibrary-root", type=str, default=DEFAULT_UWAVEGESTURELIBRARY_ROOT)
    p.add_argument("--epilepsy-root", type=str, default=DEFAULT_EPILEPSY_ROOT)
    p.add_argument("--atrialfibrillation-root", type=str, default=DEFAULT_ATRIALFIBRILLATION_ROOT)
    p.add_argument("--pendigits-root", type=str, default=DEFAULT_PENDIGITS_ROOT)
    p.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG")
    p.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    p.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)
    p.add_argument("--seedv-root", type=str, default=DEFAULT_SEEDV_ROOT)
    args = p.parse_args()

    dataset_name = normalize_dataset_name(args.dataset)
    mode = str(args.mode).strip().lower()
    if mode != "seed_light_best_round" and dataset_name in SEED_NATIVE_DATASETS:
        raise ValueError("SEED family only supports --mode seed_light_best_round in this runner.")
    if mode == "seed_light_best_round" and dataset_name not in SEED_NATIVE_DATASETS:
        raise ValueError("seed_light_best_round mode only supports seed1/seediv/seedv.")

    seeds = _parse_seed_list(args.seeds)
    stage_prefix = MODE_PREFIX[mode]
    dataset_out_dir = os.path.join(args.out_root, dataset_name)
    _ensure_dir(dataset_out_dir)

    all_trials = _load_trials_for_runner(dataset_name, args)
    by_seed_rows: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"[route-b-aug][{dataset_name}][mode={mode}][seed={seed}] split_start", flush=True)
        seed_dir = os.path.join(dataset_out_dir, f"seed{seed}")
        _ensure_dir(seed_dir)
        train_full_trials, test_trials, split_meta = resolve_protocol_split(
            dataset=dataset_name,
            all_trials=all_trials,
            seed=int(seed),
            allow_random_fallback=False,
        )
        train_core_trials, val_trials, inner_meta = resolve_inner_train_val_split(
            train_trials=train_full_trials,
            seed=int(seed) + 1701,
            val_fraction=float(args.val_fraction),
            fallback_fraction=float(args.val_fallback_fraction),
        )

        print(
            f"[route-b-aug][{dataset_name}][mode={mode}][seed={seed}] "
            f"protocol={split_meta.get('protocol_type')} inner_split={inner_meta.get('inner_split_status')}",
            flush=True,
        )

        raw_metrics, raw_meta = _fit_variant(
            dataset=dataset_name,
            train_trials=train_full_trials,
            test_trials=test_trials,
            seed=int(seed),
            args=args,
            stage_seed_dir=seed_dir,
        )
        by_seed_rows.append(
            {
                "dataset": dataset_name,
                "dataset_group": _dataset_group(dataset_name),
                "protocol": _protocol_label(split_meta),
                "seed": int(seed),
                "mode": mode,
                "train_variant": "raw",
                "test_acc": float(raw_metrics["trial_acc"]),
                "test_macro_f1": float(raw_metrics["trial_macro_f1"]),
                "best_round": np.nan,
                "final_round": np.nan,
                "pool_rounds": "",
                "selected_rounds": "",
                "pool_size_ratio": 0.0,
                "inner_split_status": str(inner_meta["inner_split_status"]),
                "inner_val_fraction": float(inner_meta["inner_val_fraction"]),
                "note": "raw_reference",
            }
        )

        train_core_tmp, mean_log_core = _build_trial_records(train_core_trials, spd_eps=float(args.spd_eps))
        train_core_records = _apply_mean_log(train_core_tmp, mean_log_core)
        X_core, y_core, tid_core = _stack_feature(train_core_records)
        if len(val_trials) > 0:
            val_tmp, _ = _build_trial_records(val_trials, spd_eps=float(args.spd_eps))
            val_records = _apply_mean_log(val_tmp, mean_log_core)
            X_val, y_val, tid_val = _stack_feature(val_records)
        else:
            X_val = y_val = tid_val = None

        direction_bank_core, _ = _build_direction_bank_d1(
            X_train=X_core,
            k_dir=int(args.k_dir),
            seed=int(seed * 10000 + int(args.k_dir) * 113 + 17),
            n_iters=int(args.pia_n_iters),
            activation=str(args.pia_activation),
            bias_update_mode=str(args.pia_bias_update_mode),
            c_repr=float(args.pia_c_repr),
        )
        selection_round_rows = _build_round_rows(
            dataset=dataset_name,
            seed=int(seed),
            direction_bank=direction_bank_core,
            X_train_base=X_core,
            y_train_base=y_core,
            tid_train=tid_core,
            X_eval=X_val,
            y_eval=y_val,
            tid_eval=tid_val,
            args=args,
            seed_offset=400000,
            progress_tag="select",
        )
        best_round_id, best_round_note, best_round_val_f1 = _choose_best_round(
            selection_round_rows,
            has_val=bool(len(val_trials) > 0),
        )

        full_tmp, mean_log_full = _build_trial_records(train_full_trials, spd_eps=float(args.spd_eps))
        full_records = _apply_mean_log(full_tmp, mean_log_full)
        X_full, y_full, tid_full = _stack_feature(full_records)
        direction_bank_full, _ = _build_direction_bank_d1(
            X_train=X_full,
            k_dir=int(args.k_dir),
            seed=int(seed * 10000 + int(args.k_dir) * 113 + 17),
            n_iters=int(args.pia_n_iters),
            activation=str(args.pia_activation),
            bias_update_mode=str(args.pia_bias_update_mode),
            c_repr=float(args.pia_c_repr),
        )

        X_single, y_single, tid_single, _src_single, _dir_single, single_aug_meta = _build_multidir_aug_candidates(
            X_train=X_full,
            y_train=y_full,
            tid_train=tid_full,
            direction_bank=direction_bank_full,
            subset_size=int(args.subset_size),
            gamma=float(args.pia_gamma),
            multiplier=int(args.pia_multiplier),
            seed=int(seed + 100000 + int(args.k_dir) * 101 + int(args.subset_size) * 7),
        )
        single_records, single_bridge_meta = _bridge_aug_trials(
            train_records=full_records,
            mean_log_train=mean_log_full,
            z_aug=np.asarray(X_single, dtype=np.float32),
            y_aug=np.asarray(y_single, dtype=np.int64),
            tid_aug=np.asarray(tid_single),
            variant_tag="single_round",
            bridge_eps=float(args.bridge_eps),
        )
        single_payload = {
            "round_id": 1,
            "aug_records": single_records,
            "aug_trials": _records_to_trial_dicts(single_records),
            "bridge_meta": single_bridge_meta,
            "aug_meta": single_aug_meta,
        }

        full_round_rows = _build_round_rows(
            dataset=dataset_name,
            seed=int(seed),
            direction_bank=direction_bank_full,
            X_train_base=X_full,
            y_train_base=y_full,
            tid_train=tid_full,
            X_eval=None,
            y_eval=None,
            tid_eval=None,
            args=args,
            seed_offset=500000,
            progress_tag="full",
        )
        full_round_rows = _attach_bridge_payload(
            train_records=full_records,
            mean_log_train=mean_log_full,
            round_rows=full_round_rows,
            bridge_eps=float(args.bridge_eps),
            variant_prefix="multiround",
        )
        best_full = next((r for r in full_round_rows if int(r["round_id"]) == int(best_round_id)), None)
        if best_full is None:
            best_full = full_round_rows[0]
        final_full = full_round_rows[-1]

        pool_candidate_rows: List[Dict[str, object]] = []
        for rid in [int(best_round_id) - 1, int(best_round_id), int(best_round_id) + 1]:
            found = next((r for r in full_round_rows if int(r["round_id"]) == rid), None)
            if found is not None:
                pool_candidate_rows.append(found)

        selection_map = {int(r["round_id"]): r for r in selection_round_rows}
        filtered_payloads: List[Dict[str, object]] = []
        if pool_candidate_rows:
            neigh_bridge_dist = np.asarray(
                [float(r["bridge_meta"]["bridge_cov_to_orig_distance_logeuc_mean"]) for r in pool_candidate_rows],
                dtype=np.float64,
            )
            neigh_cond = np.asarray([float(r["bridge_meta"]["cond_A_mean"]) for r in pool_candidate_rows], dtype=np.float64)
            neigh_shift = np.asarray(
                [float(r["bridge_meta"]["raw_mean_shift_abs_mean"]) for r in pool_candidate_rows],
                dtype=np.float64,
            )
            med_dist = float(np.median(neigh_bridge_dist)) if neigh_bridge_dist.size else 0.0
            med_cond = float(np.median(neigh_cond)) if neigh_cond.size else 0.0
            med_shift = float(np.median(neigh_shift)) if neigh_shift.size else 0.0
            best_z_val = float(best_round_val_f1) if np.isfinite(best_round_val_f1) else float("nan")
            for payload in pool_candidate_rows:
                sel_row = selection_map.get(int(payload["round_id"]))
                z_ok = True
                if sel_row is not None and np.isfinite(best_z_val):
                    z_ok = float(sel_row["z_trial_macro_f1"]) >= float(best_z_val - float(args.filter_z_gap))
                dist_ok = float(payload["bridge_meta"]["bridge_cov_to_orig_distance_logeuc_mean"]) <= med_dist * float(args.filter_bridge_dist_mult) + 1e-12
                cond_ok = float(payload["bridge_meta"]["cond_A_mean"]) <= med_cond * float(args.filter_cond_mult) + 1e-12
                shift_ok = float(payload["bridge_meta"]["raw_mean_shift_abs_mean"]) <= med_shift * float(args.filter_mean_shift_mult) + 1e-8
                risk_ok = str(_risk_comment(payload["bridge_meta"])) != "bridge_margin_shrink_risk"
                if z_ok and dist_ok and cond_ok and shift_ok and risk_ok:
                    filtered_payloads.append(payload)
        if not filtered_payloads and mode == "filtered_pool":
            filtered_payloads = list(pool_candidate_rows)

        train_variant_trials, build_meta = _build_train_variant_trials(
            train_raw_trials=train_full_trials,
            single_payload=single_payload if mode != "seed_light_best_round" else None,
            final_payload=final_full if mode in {"train_replace", "best_round_pool", "filtered_pool"} else None,
            best_payload=best_full,
            pool_payloads=pool_candidate_rows if mode in {"best_round_pool", "filtered_pool"} else [],
            filtered_payloads=filtered_payloads if mode == "filtered_pool" else [],
            seed=int(seed),
        )

        variant_order = {
            "train_replace": ["raw", "bridge_single_round", "bridge_multiround_final", "bridge_multiround_best_round"],
            "best_round_pool": ["raw", "bridge_multiround_final", "bridge_multiround_best_round", "bridge_best_round_pool"],
            "filtered_pool": ["raw", "bridge_multiround_final", "bridge_multiround_best_round", "bridge_best_round_pool", "bridge_filtered_pool"],
            "seed_light_best_round": ["raw", "seed_light_best_round"],
        }[mode]

        if len(val_trials) == 0:
            if mode in {"best_round_pool", "filtered_pool"}:
                variant_order = ["raw", "bridge_multiround_best_round"] if mode != "seed_light_best_round" else ["raw", "seed_light_best_round"]
            elif mode == "train_replace":
                variant_order = ["raw", "bridge_multiround_best_round"]

        for train_variant in variant_order:
            if train_variant not in train_variant_trials:
                continue
            metrics, _run_meta = _fit_variant(
                dataset=dataset_name,
                train_trials=train_variant_trials[train_variant],
                test_trials=test_trials,
                seed=int(seed),
                args=args,
                stage_seed_dir=os.path.join(seed_dir, train_variant),
            )
            note = best_round_note
            if mode in {"best_round_pool", "filtered_pool"} and train_variant in {"bridge_best_round_pool", "bridge_filtered_pool"}:
                note = f"{best_round_note}|pool_rounds={build_meta.get('pool_rounds', [])}"
            by_seed_rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_group": _dataset_group(dataset_name),
                    "protocol": _protocol_label(split_meta),
                    "seed": int(seed),
                    "mode": mode,
                    "train_variant": train_variant,
                    "test_acc": float(metrics["trial_acc"]),
                    "test_macro_f1": float(metrics["trial_macro_f1"]),
                    "best_round": int(build_meta["best_round"]) if build_meta.get("best_round") is not None else np.nan,
                    "final_round": int(build_meta["final_round"]) if build_meta.get("final_round") is not None else np.nan,
                    "pool_rounds": "|".join(str(int(v)) for v in build_meta.get("pool_rounds", [])),
                    "selected_rounds": "|".join(str(int(v)) for v in build_meta.get("filtered_rounds", [])),
                    "pool_size_ratio": float(
                        build_meta.get("filtered_pool_size_ratio", build_meta.get("pool_size_ratio", 0.0))
                        if train_variant == "bridge_filtered_pool"
                        else build_meta.get("pool_size_ratio", 0.0)
                    ),
                    "inner_split_status": str(inner_meta["inner_split_status"]),
                    "inner_val_fraction": float(inner_meta["inner_val_fraction"]),
                    "note": str(note),
                }
            )

        _write_json(
            os.path.join(seed_dir, "selection_meta.json"),
            {
                "split_meta": split_meta,
                "inner_meta": inner_meta,
                "best_round_id": int(best_round_id),
                "best_round_note": str(best_round_note),
                "best_round_val_f1": None if not np.isfinite(best_round_val_f1) else float(best_round_val_f1),
                "pool_rounds": build_meta.get("pool_rounds", []),
                "filtered_rounds": build_meta.get("filtered_rounds", []),
                "pool_sample_counts": build_meta.get("pool_sample_counts", {}),
                "filtered_sample_counts": build_meta.get("filtered_sample_counts", {}),
                "single_bridge_meta": single_bridge_meta,
                "best_bridge_meta": best_full["bridge_meta"],
                "final_bridge_meta": final_full["bridge_meta"],
                "selection_round_rows": [
                    {
                        "round_id": int(r["round_id"]),
                        "z_trial_macro_f1": None if not np.isfinite(r["z_trial_macro_f1"]) else float(r["z_trial_macro_f1"]),
                        "z_window_macro_f1": None if not np.isfinite(r["z_window_macro_f1"]) else float(r["z_window_macro_f1"]),
                        "direction_usage_entropy": float(r["direction_usage_entropy"]),
                        "dir_summary": r["dir_summary"],
                    }
                    for r in selection_round_rows
                ],
            },
        )

    by_seed_df = pd.DataFrame(by_seed_rows)
    dataset_summary = _variant_rows_to_summary(
        dataset=dataset_name,
        dataset_group=_dataset_group(dataset_name),
        protocol=str(by_seed_df["protocol"].iloc[0]),
        mode=mode,
        by_seed_rows=by_seed_rows,
    )
    by_seed_path = os.path.join(dataset_out_dir, f"{stage_prefix}_by_seed.csv")
    summary_path = os.path.join(dataset_out_dir, f"{stage_prefix}_summary.csv")
    conclusion_path = os.path.join(dataset_out_dir, f"{stage_prefix}_conclusion.md")
    by_seed_df.to_csv(by_seed_path, index=False)
    pd.DataFrame([dataset_summary]).to_csv(summary_path, index=False)
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write(_summary_to_markdown(dataset_summary, title=f"{_dataset_title(dataset_name)} {stage_prefix}") + "\n")


if __name__ == "__main__":
    main()
