#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_ATRIALFIBRILLATION_ROOT,
    DEFAULT_BASICMOTIONS_ROOT,
    DEFAULT_EPILEPSY_ROOT,
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_HANDMOVEMENTDIRECTION_ROOT,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_PENDIGITS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    DEFAULT_UWAVEGESTURELIBRARY_ROOT,
    load_trials_for_dataset,
)
from scripts.legacy_phase.run_phase15_mainline_freeze import _summarize_dir_profile  # noqa: E402
from scripts.legacy_phase.run_phase15_multiround_curriculum_probe import (  # noqa: E402
    _active_direction_probs,
    _build_curriculum_aug_candidates,
    _compute_direction_intrusion,
    _mech_dir_maps,
    _update_direction_budget,
)
from scripts.legacy_phase.run_phase15_fisher_curriculum_probe import _score_to_probs  # noqa: E402
from scripts.legacy_phase.run_phase15_step0a_paired_lock import _aggregate_trials, _make_trial_split  # noqa: E402
from scripts.legacy_phase.run_phase15_step1a_maxplane import _fit_eval_linearsvc  # noqa: E402
from scripts.legacy_phase.run_phase15_step1b_multidir_matrix import (  # noqa: E402
    _build_direction_bank_d1,
    _build_multidir_aug_candidates,
    _compute_mech_metrics,
)
from scripts.support.fisher_pia_utils import (  # noqa: E402
    FisherPIAConfig,
    compute_fisher_pia_terms,
    compute_safe_axis_scores,
)
from scripts.raw_baselines.run_raw_bridge_probe import (  # noqa: E402
    TrialRecord,
    _apply_mean_log,
    _build_trial_records,
)
from scripts.raw_baselines.run_raw_minirocket_baseline import (  # noqa: E402
    _build_capped_train_windows,
    _build_model,
    _iter_test_windows,
    _resolve_window_policy,
    _to_scores,
)
from transforms.whiten_color_bridge import bridge_single, logvec_to_spd  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, ensure_ascii=False, indent=2)


def _compact_json(obj) -> str:
    return json.dumps(_json_sanitize(obj), ensure_ascii=False, sort_keys=True)


def _format_mean_std(values: Sequence[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "0.0000 +/- 0.0000"
    return f"{float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f}"


def _minmax_norm(x: np.ndarray, *, constant_fill: float = 0.5) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0:
        return np.asarray([], dtype=np.float64)
    xmin = float(np.min(xx))
    xmax = float(np.max(xx))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) <= 1e-12:
        return np.full(xx.shape, float(constant_fill), dtype=np.float64)
    return (xx - xmin) / (xmax - xmin)


def _build_direction_probs_with_prior(
    *,
    fisher_probs: np.ndarray,
    gamma_by_dir: np.ndarray,
    freeze_eps: float,
) -> np.ndarray:
    probs = np.asarray(fisher_probs, dtype=np.float64).ravel().copy()
    gamma = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    active = gamma > float(freeze_eps)
    probs = probs * active.astype(np.float64)
    total = float(np.sum(probs))
    if total <= 0:
        if np.any(active):
            probs = active.astype(np.float64) / float(np.sum(active))
        else:
            probs = np.full((len(gamma),), 1.0 / float(max(1, len(gamma))), dtype=np.float64)
    else:
        probs = probs / total
    return probs


def _entropy_from_pick_fraction(pick_fraction: Dict[str, object]) -> float:
    if not isinstance(pick_fraction, dict) or not pick_fraction:
        return 0.0
    vals = np.asarray([float(v) for v in pick_fraction.values()], dtype=np.float64)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    return float(-np.sum(vals * np.log(vals)))


def _direction_usage_entropy_from_aug_meta(aug_meta: Dict[str, object]) -> float:
    if "direction_usage_entropy" in aug_meta:
        return float(aug_meta.get("direction_usage_entropy", 0.0))
    mixing = aug_meta.get("mixing_stats", {})
    if isinstance(mixing, dict):
        return _entropy_from_pick_fraction(mixing.get("direction_pick_fraction", {}))
    return 0.0


def _trial_record_to_dict(rec: TrialRecord) -> Dict[str, object]:
    return {
        "trial_id_str": str(rec.tid),
        "label": int(rec.y),
        "x_trial": np.asarray(rec.x_raw, dtype=np.float32),
    }


def _records_to_trial_dicts(records: Sequence[TrialRecord]) -> List[Dict[str, object]]:
    return [_trial_record_to_dict(r) for r in records]


def _fit_raw_minirocket(
    *,
    dataset: str,
    train_trials: Sequence[Dict[str, object]],
    test_trials: Sequence[Dict[str, object]],
    seed: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] "
        f"resolve_policy_start train_trials={len(train_trials)} test_trials={len(test_trials)}",
        flush=True,
    )
    policy = _resolve_window_policy(
        dataset=dataset,
        train_trials=train_trials,
        fixed_window_sec=float(args.window_sec),
        fixed_hop_sec=float(args.hop_sec),
        prop_win_ratio=float(args.prop_win_ratio),
        prop_hop_ratio=float(args.prop_hop_ratio),
        min_win_len=int(args.min_window_len_samples),
        min_hop_len=int(args.min_hop_len_samples),
    )
    win_len = int(policy["window_len_samples"])
    hop_len = int(policy["hop_len_samples"])
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] "
        f"resolve_policy_done window_len={win_len} hop_len={hop_len}",
        flush=True,
    )

    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] build_train_windows_start",
        flush=True,
    )
    x_train, y_train, _tid_train, cap_meta = _build_capped_train_windows(
        train_trials=train_trials,
        win_len=win_len,
        hop_len=hop_len,
        cap_k=int(args.nominal_cap_k),
        cap_seed=int(seed),
        cap_policy=str(args.cap_sampling_policy),
        seed_out_dir=str(args.out_root),
        memmap_threshold_bytes=int(float(args.memmap_threshold_gb) * (1024**3)),
    )
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] "
        f"build_train_windows_done total_train_windows={int(cap_meta['total_train_windows'])}",
        flush=True,
    )

    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] model_build_start",
        flush=True,
    )
    model = _build_model(
        n_kernels=int(args.n_kernels),
        random_state=int(seed),
        n_jobs=int(args.n_jobs),
    )
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] model_fit_start",
        flush=True,
    )
    model.fit(x_train, y_train)
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] model_fit_done",
        flush=True,
    )

    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] build_test_windows_start",
        flush=True,
    )
    x_test_list, y_test_list, tid_test_list, n_short_test = _iter_test_windows(
        test_trials=test_trials,
        win_len=win_len,
        hop_len=hop_len,
    )
    if not x_test_list:
        raise RuntimeError(f"No test windows generated for dataset={dataset}, seed={seed}.")
    x_test = np.stack(x_test_list, axis=0).astype(np.float32, copy=False)
    y_test = np.asarray(y_test_list, dtype=np.int64)
    tid_test = np.asarray(tid_test_list, dtype=object)
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] "
        f"build_test_windows_done total_test_windows={len(y_test)}",
        flush=True,
    )
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] predict_start",
        flush=True,
    )
    y_pred_win = np.asarray(model.predict(x_test), dtype=np.int64)
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] predict_done",
        flush=True,
    )

    window_acc = float(accuracy_score(y_test, y_pred_win))
    window_macro_f1 = float(f1_score(y_test, y_pred_win, average="macro"))
    classes = np.unique(np.concatenate([y_train, y_test]))
    scores_win = _to_scores(y_pred_win, classes=classes)
    y_true_trial, y_pred_trial = _aggregate_trials(
        y_true_win=y_test,
        y_pred_win=y_pred_win,
        scores_win=scores_win,
        tid_win=tid_test,
        mode=str(args.aggregation_mode),
    )
    trial_acc = float(accuracy_score(y_true_trial, y_pred_trial))
    trial_macro_f1 = float(f1_score(y_true_trial, y_pred_trial, average="macro"))

    metrics = {
        "trial_acc": trial_acc,
        "trial_macro_f1": trial_macro_f1,
        "window_acc": window_acc,
        "window_macro_f1": window_macro_f1,
        "train_trial_count": int(len(train_trials)),
        "test_trial_count": int(len(test_trials)),
        "n_short_trials_padded_test": int(n_short_test),
    }
    run_meta = {
        "dataset": dataset,
        "seed": int(seed),
        "pipeline_name": "bridge_curriculum_pilot_raw_minirocket",
        "window_policy_name": str(policy["window_policy_name"]),
        "window_len_samples": int(policy["window_len_samples"]),
        "hop_len_samples": int(policy["hop_len_samples"]),
        "window_norm_mode": "per_window_per_channel_zscore",
        "aggregation_mode": str(args.aggregation_mode),
        "model_type": "aeon_minirocket_wrapper",
        "n_kernels": int(args.n_kernels),
        "n_jobs": int(args.n_jobs),
        "train_only_fit": True,
        "test_augmented": False,
            "cap_sampling_policy": str(args.cap_sampling_policy),
            "nominal_cap_K": int(args.nominal_cap_k),
            "effective_cap_K": int(cap_meta["effective_cap_K"]),
            "total_train_windows_before_cap": int(cap_meta["total_train_windows_before_cap"]),
            "total_train_windows": int(cap_meta["total_train_windows"]),
        }
    print(
        f"[bridge-curriculum-pilot][{dataset}][seed={seed}][raw-minirocket] "
        f"trial_f1={trial_macro_f1:.4f} trial_acc={trial_acc:.4f}",
        flush=True,
    )
    return metrics, run_meta


def _bridge_aug_trials(
    *,
    train_records: Sequence[TrialRecord],
    mean_log_train: np.ndarray,
    z_aug: np.ndarray,
    y_aug: np.ndarray,
    tid_aug: np.ndarray,
    variant_tag: str,
    bridge_eps: float,
) -> Tuple[List[TrialRecord], Dict[str, object]]:
    tid_to_rec = {str(r.tid): r for r in train_records}
    aug_trials: List[TrialRecord] = []
    cov_match: List[float] = []
    cov_match_fro: List[float] = []
    cov_match_logeuc: List[float] = []
    cov_to_orig_fro: List[float] = []
    cov_to_orig_logeuc: List[float] = []
    energy_ratio: List[float] = []
    cond_A: List[float] = []
    raw_mean_shift: List[float] = []
    classwise_shift: Dict[str, List[float]] = {}

    import torch

    for i, (z_vec, y_val, tid) in enumerate(zip(z_aug, y_aug, tid_aug)):
        src = tid_to_rec[str(tid)]
        sigma_aug = logvec_to_spd(np.asarray(z_vec, dtype=np.float32), mean_log_train)
        x_aug, bmeta = bridge_single(
            torch.from_numpy(np.asarray(src.x_raw, dtype=np.float32)),
            torch.from_numpy(np.asarray(src.sigma_orig, dtype=np.float32)),
            torch.from_numpy(np.asarray(sigma_aug, dtype=np.float32)),
            eps=float(bridge_eps),
        )
        new_tid = f"{src.tid}__{variant_tag}_aug_{i:06d}"
        aug_trials.append(
            TrialRecord(
                tid=new_tid,
                y=int(y_val),
                x_raw=x_aug.cpu().numpy().astype(np.float32),
                sigma_orig=np.asarray(sigma_aug, dtype=np.float32),
                log_cov=np.asarray(src.log_cov, dtype=np.float32),
                z=np.asarray(z_vec, dtype=np.float32),
            )
        )
        cov_match.append(float(bmeta["bridge_cov_match_error"]))
        cov_match_fro.append(float(bmeta["bridge_cov_match_error_fro"]))
        cov_match_logeuc.append(float(bmeta["bridge_cov_match_error_logeuc"]))
        cov_to_orig_fro.append(float(bmeta["bridge_cov_to_orig_distance_fro"]))
        cov_to_orig_logeuc.append(float(bmeta["bridge_cov_to_orig_distance_logeuc"]))
        energy_ratio.append(float(bmeta["bridge_energy_ratio"]))
        cond_A.append(float(bmeta["bridge_cond_A"]))
        raw_mean_shift.append(float(bmeta["raw_mean_shift_abs"]))
        classwise_shift.setdefault(str(int(y_val)), []).append(float(bmeta["raw_mean_shift_abs"]))

    orig_counts: Dict[str, int] = {}
    for r in train_records:
        key = str(int(r.y))
        orig_counts[key] = int(orig_counts.get(key, 0) + 1)
    aug_counts: Dict[str, int] = {}
    for r in aug_trials:
        key = str(int(r.y))
        aug_counts[key] = int(aug_counts.get(key, 0) + 1)
    orig_total = float(sum(orig_counts.values()))
    aug_total = float(sum(aug_counts.values()))
    orig_share = {k: (v / orig_total if orig_total > 0 else 0.0) for k, v in sorted(orig_counts.items())}
    aug_share = {k: (aug_counts.get(k, 0) / aug_total if aug_total > 0 else 0.0) for k in sorted(orig_counts)}
    class_balance_shift = {k: float(aug_share.get(k, 0.0) - orig_share.get(k, 0.0)) for k in sorted(orig_counts)}
    classwise_shift_summary = {
        k: float(np.mean(v)) if v else 0.0 for k, v in sorted(classwise_shift.items(), key=lambda kv: int(kv[0]))
    }
    meta = {
        "bridge_aug_count": int(len(aug_trials)),
        "train_selected_aug_ratio": float(len(aug_trials) / max(1, len(train_records))),
        "bridge_cov_match_error_mean": float(np.mean(cov_match)) if cov_match else 0.0,
        "bridge_cov_match_error_std": float(np.std(cov_match)) if cov_match else 0.0,
        "bridge_cov_match_error_fro_mean": float(np.mean(cov_match_fro)) if cov_match_fro else 0.0,
        "bridge_cov_match_error_logeuc_mean": float(np.mean(cov_match_logeuc)) if cov_match_logeuc else 0.0,
        "bridge_cov_to_orig_distance_fro_mean": float(np.mean(cov_to_orig_fro)) if cov_to_orig_fro else 0.0,
        "bridge_cov_to_orig_distance_logeuc_mean": float(np.mean(cov_to_orig_logeuc)) if cov_to_orig_logeuc else 0.0,
        "energy_ratio_mean": float(np.mean(energy_ratio)) if energy_ratio else 0.0,
        "energy_ratio_std": float(np.std(energy_ratio)) if energy_ratio else 0.0,
        "cond_A_mean": float(np.mean(cond_A)) if cond_A else 0.0,
        "cond_A_std": float(np.std(cond_A)) if cond_A else 0.0,
        "raw_mean_shift_abs_mean": float(np.mean(raw_mean_shift)) if raw_mean_shift else 0.0,
        "raw_mean_shift_abs_max": float(np.max(raw_mean_shift)) if raw_mean_shift else 0.0,
        "class_balance_shift_summary": _compact_json(class_balance_shift),
        "class_balance_shift_max_abs": max((abs(v) for v in class_balance_shift.values()), default=0.0),
        "classwise_mean_shift_summary": _compact_json(classwise_shift_summary),
    }
    return aug_trials, meta


def _risk_comment(meta: Dict[str, object]) -> str:
    cond_mean = float(meta.get("cond_A_mean", 0.0))
    mean_shift = float(meta.get("raw_mean_shift_abs_mean", 0.0))
    energy = float(meta.get("energy_ratio_mean", 1.0))
    cov_log = float(meta.get("bridge_cov_match_error_logeuc_mean", 0.0))
    if cond_mean <= 2.0 and mean_shift <= 1e-6 and abs(energy - 1.0) <= 0.02 and cov_log <= 0.08:
        return "clean_low_risk"
    if cond_mean <= 5.0 and mean_shift <= 1e-4 and abs(energy - 1.0) <= 0.05:
        return "usable_small_nonzero_shift"
    return "watch_numerical_or_distribution_risk"


def _target_health_comment(variant: str, dir_summary: Dict[str, object], entropy: float, ref_margin: float | None = None) -> str:
    worst_margin = dir_summary.get("worst_dir_margin_drop")
    if variant == "bridge_single_round":
        return "single_round_equal_weight_reference"
    if variant == "bridge_fisher_curriculum":
        if worst_margin is None:
            return "fisher_curriculum_missing_dir_summary"
        if ref_margin is not None and float(worst_margin) >= float(ref_margin) - 1e-12:
            if entropy < 1.2:
                return "fisher_curriculum_cleaner_with_direction_focus"
            return "fisher_curriculum_cleaner_than_single_round"
        return "fisher_curriculum_not_cleaner_than_single_round"
    if worst_margin is None:
        return "curriculum_target_missing_dir_summary"
    if ref_margin is not None and float(worst_margin) >= float(ref_margin) - 1e-12:
        if entropy < 1.2:
            return "curriculum_cleaner_with_direction_focus"
        return "curriculum_cleaner_than_single_round"
    return "curriculum_not_cleaner_than_single_round"


def _result_label(delta_vs_single: float) -> str:
    if delta_vs_single > 1e-6:
        return "positive"
    if delta_vs_single < -1e-6:
        return "negative"
    return "flat"


def _select_best_curriculum_round(round_rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not round_rows:
        raise RuntimeError("No multiround rows produced.")
    return max(
        round_rows,
        key=lambda row: (
            float(row["z_trial_macro_f1"]),
            float(row["z_window_macro_f1"]),
            -int(row["round_id"]),
        ),
    )


def _dataset_title(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key == "har":
        return "HAR"
    if key == "fingermovements":
        return "FingerMovements"
    if key == "natops":
        return "NATOPS"
    if key == "selfregulationscp1":
        return "SelfRegulationSCP1"
    if key == "basicmotions":
        return "BasicMotions"
    if key == "handmovementdirection":
        return "HandMovementDirection"
    if key == "uwavegesturelibrary":
        return "UWaveGestureLibrary"
    if key == "epilepsy":
        return "Epilepsy"
    if key == "atrialfibrillation":
        return "AtrialFibrillation"
    if key == "pendigits":
        return "PenDigits"
    return str(dataset).upper()


def _output_file_map(dataset: str) -> Dict[str, str]:
    key = str(dataset).strip().lower()
    if key == "natops":
        return {
            "per_seed": "bridge_curriculum_natops_pilot_per_seed.csv",
            "summary": "bridge_curriculum_natops_pilot_summary.csv",
            "target_health": "bridge_curriculum_natops_target_health_summary.csv",
            "fidelity": "bridge_curriculum_natops_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_natops_pilot_conclusion.md",
        }
    if key == "fingermovements":
        return {
            "per_seed": "bridge_curriculum_fm_pilot_per_seed.csv",
            "summary": "bridge_curriculum_fm_pilot_summary.csv",
            "target_health": "bridge_curriculum_fm_target_health_summary.csv",
            "fidelity": "bridge_curriculum_fm_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_fm_pilot_conclusion.md",
        }
    if key == "selfregulationscp1":
        return {
            "per_seed": "bridge_curriculum_scp1_pilot_per_seed.csv",
            "summary": "bridge_curriculum_scp1_pilot_summary.csv",
            "target_health": "bridge_curriculum_scp1_target_health_summary.csv",
            "fidelity": "bridge_curriculum_scp1_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_scp1_pilot_conclusion.md",
        }
    if key == "basicmotions":
        return {
            "per_seed": "bridge_curriculum_basicmotions_pilot_per_seed.csv",
            "summary": "bridge_curriculum_basicmotions_pilot_summary.csv",
            "target_health": "bridge_curriculum_basicmotions_target_health_summary.csv",
            "fidelity": "bridge_curriculum_basicmotions_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_basicmotions_pilot_conclusion.md",
        }
    if key == "handmovementdirection":
        return {
            "per_seed": "bridge_curriculum_handmovementdirection_pilot_per_seed.csv",
            "summary": "bridge_curriculum_handmovementdirection_pilot_summary.csv",
            "target_health": "bridge_curriculum_handmovementdirection_target_health_summary.csv",
            "fidelity": "bridge_curriculum_handmovementdirection_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_handmovementdirection_pilot_conclusion.md",
        }
    if key == "uwavegesturelibrary":
        return {
            "per_seed": "bridge_curriculum_uwavegesturelibrary_pilot_per_seed.csv",
            "summary": "bridge_curriculum_uwavegesturelibrary_pilot_summary.csv",
            "target_health": "bridge_curriculum_uwavegesturelibrary_target_health_summary.csv",
            "fidelity": "bridge_curriculum_uwavegesturelibrary_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_uwavegesturelibrary_pilot_conclusion.md",
        }
    if key == "epilepsy":
        return {
            "per_seed": "bridge_curriculum_epilepsy_pilot_per_seed.csv",
            "summary": "bridge_curriculum_epilepsy_pilot_summary.csv",
            "target_health": "bridge_curriculum_epilepsy_target_health_summary.csv",
            "fidelity": "bridge_curriculum_epilepsy_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_epilepsy_pilot_conclusion.md",
        }
    if key == "atrialfibrillation":
        return {
            "per_seed": "bridge_curriculum_atrialfibrillation_pilot_per_seed.csv",
            "summary": "bridge_curriculum_atrialfibrillation_pilot_summary.csv",
            "target_health": "bridge_curriculum_atrialfibrillation_target_health_summary.csv",
            "fidelity": "bridge_curriculum_atrialfibrillation_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_atrialfibrillation_pilot_conclusion.md",
        }
    if key == "pendigits":
        return {
            "per_seed": "bridge_curriculum_pendigits_pilot_per_seed.csv",
            "summary": "bridge_curriculum_pendigits_pilot_summary.csv",
            "target_health": "bridge_curriculum_pendigits_target_health_summary.csv",
            "fidelity": "bridge_curriculum_pendigits_fidelity_summary.csv",
            "conclusion": "bridge_curriculum_pendigits_pilot_conclusion.md",
        }
    return {
        "per_seed": "bridge_curriculum_pilot_per_seed.csv",
        "summary": "bridge_curriculum_pilot_summary.csv",
        "target_health": "bridge_curriculum_target_health_summary.csv",
        "fidelity": "bridge_curriculum_fidelity_summary.csv",
        "conclusion": "bridge_curriculum_pilot_conclusion.md",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Curriculum target -> bridge -> MiniROCKET pilot.")
    p.add_argument(
        "--dataset",
        type=str,
        default="har",
        choices=[
            "har",
            "natops",
            "selfregulationscp1",
            "fingermovements",
            "basicmotions",
            "handmovementdirection",
            "uwavegesturelibrary",
            "epilepsy",
            "atrialfibrillation",
            "pendigits",
        ],
    )
    p.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    p.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    p.add_argument("--basicmotions-root", type=str, default=DEFAULT_BASICMOTIONS_ROOT)
    p.add_argument("--handmovementdirection-root", type=str, default=DEFAULT_HANDMOVEMENTDIRECTION_ROOT)
    p.add_argument("--uwavegesturelibrary-root", type=str, default=DEFAULT_UWAVEGESTURELIBRARY_ROOT)
    p.add_argument("--epilepsy-root", type=str, default=DEFAULT_EPILEPSY_ROOT)
    p.add_argument("--atrialfibrillation-root", type=str, default=DEFAULT_ATRIALFIBRILLATION_ROOT)
    p.add_argument("--pendigits-root", type=str, default=DEFAULT_PENDIGITS_ROOT)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/bridge_curriculum_pilot_20260320")
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
    p.add_argument("--include-fisher", action="store_true")
    p.add_argument("--fisher-beta", type=float, default=1.0)
    p.add_argument("--fisher-knn-k", type=int, default=20)
    p.add_argument("--fisher-boundary-quantile", type=float, default=0.30)
    p.add_argument("--fisher-interior-quantile", type=float, default=0.70)
    p.add_argument("--fisher-hetero-k", type=int, default=3)
    p.add_argument("--fisher-low-quality-quantile", type=float, default=0.35)
    p.add_argument("--fisher-downweight-quantile", type=float, default=0.55)
    p.add_argument("--fisher-downweight-gamma-mult", type=float, default=0.70)
    args = p.parse_args()

    seeds = [int(tok.strip()) for tok in str(args.seeds).split(",") if tok.strip()]
    if not seeds:
        raise ValueError("seed list cannot be empty")
    if int(args.subset_size) != 1:
        raise ValueError("This pilot locks subset_size=1 to stay aligned with Step1B.")

    dataset_name = str(args.dataset).strip().lower()
    dataset_title = _dataset_title(dataset_name)
    output_files = _output_file_map(dataset_name)
    _ensure_dir(args.out_root)
    all_trials = load_trials_for_dataset(
        dataset=dataset_name,
        fingermovements_root=args.fingermovements_root,
        har_root=args.har_root,
        natops_root=args.natops_root,
        selfregulationscp1_root=args.selfregulationscp1_root,
        basicmotions_root=args.basicmotions_root,
        handmovementdirection_root=args.handmovementdirection_root,
        uwavegesturelibrary_root=args.uwavegesturelibrary_root,
        epilepsy_root=args.epilepsy_root,
        atrialfibrillation_root=args.atrialfibrillation_root,
        pendigits_root=args.pendigits_root,
    )
    fisher_cfg = FisherPIAConfig(
        knn_k=int(args.fisher_knn_k),
        interior_quantile=float(args.fisher_interior_quantile),
        boundary_quantile=float(args.fisher_boundary_quantile),
        hetero_k=int(args.fisher_hetero_k),
    )

    per_seed_rows: List[Dict[str, object]] = []
    target_health_rows: List[Dict[str, object]] = []
    fidelity_rows: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] split_start", flush=True)
        seed_dir = os.path.join(args.out_root, dataset_name, f"seed{seed}")
        _ensure_dir(seed_dir)

        train_trials, test_trials, split_meta = _make_trial_split(list(all_trials), int(seed))
        train_tmp, mean_log_train = _build_trial_records(train_trials, spd_eps=float(args.spd_eps))
        test_tmp, _ = _build_trial_records(test_trials, spd_eps=float(args.spd_eps))
        train_records = _apply_mean_log(train_tmp, mean_log_train)
        test_records = _apply_mean_log(test_tmp, mean_log_train)

        X_train_base = np.stack([r.z for r in train_records], axis=0).astype(np.float32)
        y_train_base = np.asarray([int(r.y) for r in train_records], dtype=np.int64)
        tid_train = np.asarray([r.tid for r in train_records], dtype=object)
        X_test = np.stack([r.z for r in test_records], axis=0).astype(np.float32)
        y_test = np.asarray([int(r.y) for r in test_records], dtype=np.int64)
        tid_test = np.asarray([r.tid for r in test_records], dtype=object)

        direction_bank, bank_meta = _build_direction_bank_d1(
            X_train=X_train_base,
            k_dir=int(args.k_dir),
            seed=int(seed * 10000 + int(args.k_dir) * 113 + 17),
            n_iters=int(args.pia_n_iters),
            activation=str(args.pia_activation),
            bias_update_mode=str(args.pia_bias_update_mode),
            c_repr=float(args.pia_c_repr),
        )

        z_base_metrics, _ = _fit_eval_linearsvc(
            X_train_base,
            y_train_base,
            tid_train,
            X_test,
            y_test,
            tid_test,
            seed=int(seed),
            cap_k=int(args.nominal_cap_k),
            cap_seed=int(seed + 41),
            cap_sampling_policy="balanced_real_aug",
            linear_c=float(args.linear_c),
            class_weight=str(args.linear_class_weight),
            max_iter=int(args.linear_max_iter),
            agg_mode="majority",
            is_aug_train=np.zeros((len(y_train_base),), dtype=bool),
            progress_prefix=f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}][z-baseline]",
        )

        X_step1b, y_step1b, tid_step1b, src_step1b, dir_step1b, step1b_aug_meta = _build_multidir_aug_candidates(
            X_train=X_train_base,
            y_train=y_train_base,
            tid_train=tid_train,
            direction_bank=direction_bank,
            subset_size=int(args.subset_size),
            gamma=float(args.pia_gamma),
            multiplier=int(args.pia_multiplier),
            seed=int(seed + 100000 + int(args.k_dir) * 101 + int(args.subset_size) * 7),
        )
        mech_step1b = _compute_mech_metrics(
            X_train_real=X_train_base,
            y_train_real=y_train_base,
            X_aug_generated=X_step1b,
            y_aug_generated=y_step1b,
            X_aug_accepted=X_step1b,
            y_aug_accepted=y_step1b,
            X_src_accepted=src_step1b,
            dir_generated=dir_step1b,
            dir_accepted=dir_step1b,
            seed=int(seed),
            linear_c=float(args.linear_c),
            class_weight=str(args.linear_class_weight),
            linear_max_iter=int(args.linear_max_iter),
            knn_k=int(args.mech_knn_k),
                max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                max_real_knn_ref=int(args.mech_max_real_knn_ref),
                max_real_knn_query=int(args.mech_max_real_knn_query),
                progress_prefix=f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}][z-step1b-mech]",
        )
        X_train_step1b = np.vstack([X_train_base, X_step1b]) if len(y_step1b) else X_train_base.copy()
        y_train_step1b = np.concatenate([y_train_base, y_step1b]) if len(y_step1b) else y_train_base.copy()
        tid_train_step1b = np.concatenate([tid_train, tid_step1b]) if len(y_step1b) else tid_train.copy()
        is_aug_step1b = (
            np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_step1b),), dtype=bool)])
            if len(y_step1b)
            else np.zeros((len(y_train_base),), dtype=bool)
        )
        z_step1b_metrics, _ = _fit_eval_linearsvc(
            X_train_step1b,
            y_train_step1b,
            tid_train_step1b,
            X_test,
            y_test,
            tid_test,
            seed=int(seed),
            cap_k=int(args.nominal_cap_k),
            cap_seed=int(seed + 41),
            cap_sampling_policy="balanced_real_aug",
            linear_c=float(args.linear_c),
            class_weight=str(args.linear_class_weight),
            max_iter=int(args.linear_max_iter),
            agg_mode="majority",
            is_aug_train=is_aug_step1b,
            progress_prefix=f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}][z-step1b-fit]",
        )
        step1b_dir_summary = _summarize_dir_profile(mech_step1b.get("dir_profile", {}))

        round_rows: List[Dict[str, object]] = []
        gamma_by_dir = np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64)
        for round_id in range(1, int(args.curriculum_rounds) + 1):
            direction_probs = _active_direction_probs(
                gamma_by_dir,
                freeze_eps=float(args.curriculum_freeze_eps),
            )
            gamma_before = gamma_by_dir.copy()
            X_curr, y_curr, tid_curr, src_curr, dir_curr, curr_aug_meta = _build_curriculum_aug_candidates(
                X_train=X_train_base,
                y_train=y_train_base,
                tid_train=tid_train,
                direction_bank=direction_bank,
                direction_probs=direction_probs,
                gamma_by_dir=gamma_before,
                multiplier=int(args.pia_multiplier),
                seed=int(seed + 400000 + round_id * 1009),
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
                progress_prefix=f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}][z-curr-round={round_id}-mech]",
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
            X_train_curr = np.vstack([X_train_base, X_curr]) if len(y_curr) else X_train_base.copy()
            y_train_curr = np.concatenate([y_train_base, y_curr]) if len(y_curr) else y_train_base.copy()
            tid_train_curr = np.concatenate([tid_train, tid_curr]) if len(y_curr) else tid_train.copy()
            is_aug_curr = (
                np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_curr),), dtype=bool)])
                if len(y_curr)
                else np.zeros((len(y_train_base),), dtype=bool)
            )
            z_curr_metrics, _ = _fit_eval_linearsvc(
                X_train_curr,
                y_train_curr,
                tid_train_curr,
                X_test,
                y_test,
                tid_test,
                seed=int(seed),
                cap_k=int(args.nominal_cap_k),
                cap_seed=int(seed + 41),
                cap_sampling_policy="balanced_real_aug",
                linear_c=float(args.linear_c),
                class_weight=str(args.linear_class_weight),
                max_iter=int(args.linear_max_iter),
                agg_mode="majority",
                is_aug_train=is_aug_curr,
                progress_prefix=f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}][z-curr-round={round_id}-fit]",
            )
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
            round_rows.append(
                {
                    "round_id": int(round_id),
                    "z_aug": X_curr,
                    "y_aug": y_curr,
                    "tid_aug": tid_curr,
                    "dir_aug": dir_curr,
                    "z_trial_macro_f1": float(z_curr_metrics["trial_macro_f1"]),
                    "z_window_macro_f1": float(z_curr_metrics["window_macro_f1"]),
                    "delta_vs_step1b": float(z_curr_metrics["trial_macro_f1"] - z_step1b_metrics["trial_macro_f1"]),
                    "mech": mech_curr,
                    "dir_summary": _summarize_dir_profile(mech_curr.get("dir_profile", {})),
                    "direction_usage_entropy": float(curr_aug_meta.get("direction_usage_entropy", 0.0)),
                    "direction_probs": curr_aug_meta.get("direction_probs", {}),
                    "gamma_before": {str(i): float(gamma_before[i]) for i in range(len(gamma_before))},
                    "gamma_after": {str(i): float(gamma_after[i]) for i in range(len(gamma_after))},
                    "direction_state": {str(k): str(v) for k, v in state_by_dir.items()},
                    "direction_score": {str(k): float(v) for k, v in score_by_dir.items()},
                    "aug_meta": curr_aug_meta,
                }
            )
            gamma_by_dir = gamma_after.copy()

        best_round = _select_best_curriculum_round(round_rows)
        best_fisher_round = None
        fisher_rounds: List[Dict[str, object]] = []
        if bool(args.include_fisher):
            class_terms, _ = compute_fisher_pia_terms(X_train_base, y_train_base, cfg=fisher_cfg)
            _, fisher_global_df = compute_safe_axis_scores(
                direction_bank,
                class_terms,
                beta=float(args.fisher_beta),
                gamma=0.0,
                include_approach=False,
                direction_score_mode="axis_level",
            )
            fisher_scores = fisher_global_df["revised_score"].to_numpy(dtype=np.float64)
            fisher_norm = _minmax_norm(fisher_scores, constant_fill=0.5)
            q_low = float(np.quantile(fisher_norm, float(args.fisher_low_quality_quantile))) if fisher_norm.size else 0.0
            q_down = float(np.quantile(fisher_norm, float(args.fisher_downweight_quantile))) if fisher_norm.size else 0.5
            prior_frozen_mask = fisher_norm <= q_low
            if np.all(prior_frozen_mask) and prior_frozen_mask.size:
                prior_frozen_mask[int(np.argmax(fisher_norm))] = False
            downweighted_mask = (~prior_frozen_mask) & (fisher_norm <= q_down)
            fisher_probs = _score_to_probs(np.where(prior_frozen_mask, np.min(fisher_scores) - 1.0, fisher_scores))
            gamma_by_dir_fisher = np.full((int(args.k_dir),), float(args.curriculum_init_gamma), dtype=np.float64)
            gamma_by_dir_fisher[downweighted_mask] *= float(args.fisher_downweight_gamma_mult)
            gamma_by_dir_fisher[prior_frozen_mask] = 0.0

            for round_id in range(1, int(args.curriculum_rounds) + 1):
                fisher_direction_probs = _build_direction_probs_with_prior(
                    fisher_probs=fisher_probs,
                    gamma_by_dir=gamma_by_dir_fisher,
                    freeze_eps=float(args.curriculum_freeze_eps),
                )
                fisher_gamma_before = gamma_by_dir_fisher.copy()
                X_fisher, y_fisher, tid_fisher, src_fisher, dir_fisher, fisher_aug_meta = _build_curriculum_aug_candidates(
                    X_train=X_train_base,
                    y_train=y_train_base,
                    tid_train=tid_train,
                    direction_bank=direction_bank,
                    direction_probs=fisher_direction_probs,
                    gamma_by_dir=fisher_gamma_before,
                    multiplier=int(args.pia_multiplier),
                    seed=int(seed + 700000 + round_id * 1009),
                )
                mech_fisher = _compute_mech_metrics(
                    X_train_real=X_train_base,
                    y_train_real=y_train_base,
                    X_aug_generated=X_fisher,
                    y_aug_generated=y_fisher,
                    X_aug_accepted=X_fisher,
                    y_aug_accepted=y_fisher,
                    X_src_accepted=src_fisher,
                    dir_generated=dir_fisher,
                    dir_accepted=dir_fisher,
                    seed=int(seed),
                    linear_c=float(args.linear_c),
                    class_weight=str(args.linear_class_weight),
                    linear_max_iter=int(args.linear_max_iter),
                    knn_k=int(args.mech_knn_k),
                    max_aug_for_mech=int(args.mech_max_aug_for_metrics),
                    max_real_knn_ref=int(args.mech_max_real_knn_ref),
                    max_real_knn_query=int(args.mech_max_real_knn_query),
                    progress_prefix=f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}][z-fisher-round={round_id}-mech]",
                )
                fisher_intrusion_by_dir = _compute_direction_intrusion(
                    X_anchor=X_train_base,
                    y_anchor=y_train_base,
                    X_aug_accepted=X_fisher,
                    y_aug_accepted=y_fisher,
                    dir_accepted=dir_fisher,
                    seed=int(seed),
                    knn_k=int(args.mech_knn_k),
                    max_eval=int(args.mech_max_aug_for_metrics),
                )
                fisher_maps = _mech_dir_maps(mech_fisher, intrusion_by_dir=fisher_intrusion_by_dir)
                X_train_fisher = np.vstack([X_train_base, X_fisher]) if len(y_fisher) else X_train_base.copy()
                y_train_fisher = np.concatenate([y_train_base, y_fisher]) if len(y_fisher) else y_train_base.copy()
                tid_train_fisher = np.concatenate([tid_train, tid_fisher]) if len(y_fisher) else tid_train.copy()
                is_aug_fisher = (
                    np.concatenate([np.zeros((len(y_train_base),), dtype=bool), np.ones((len(y_fisher),), dtype=bool)])
                    if len(y_fisher)
                    else np.zeros((len(y_train_base),), dtype=bool)
                )
                z_fisher_metrics, _ = _fit_eval_linearsvc(
                    X_train_fisher,
                    y_train_fisher,
                    tid_train_fisher,
                    X_test,
                    y_test,
                    tid_test,
                    seed=int(seed),
                    cap_k=int(args.nominal_cap_k),
                    cap_seed=int(seed + 41),
                    cap_sampling_policy="balanced_real_aug",
                    linear_c=float(args.linear_c),
                    class_weight=str(args.linear_class_weight),
                    max_iter=int(args.linear_max_iter),
                    agg_mode="majority",
                    is_aug_train=is_aug_fisher,
                    progress_prefix=f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}][z-fisher-round={round_id}-fit]",
                )
                fisher_gamma_after, fisher_state_by_dir, fisher_score_by_dir = _update_direction_budget(
                    gamma_before=fisher_gamma_before,
                    margin_by_dir=fisher_maps["margin_drop_median"],
                    flip_by_dir=fisher_maps["flip_rate"],
                    intrusion_by_dir=fisher_maps["intrusion"],
                    expand_factor=float(args.curriculum_expand_factor),
                    shrink_factor=float(args.curriculum_shrink_factor),
                    gamma_max=float(args.curriculum_gamma_max),
                    freeze_eps=float(args.curriculum_freeze_eps),
                )
                fisher_rounds.append(
                    {
                        "round_id": int(round_id),
                        "z_aug": X_fisher,
                        "y_aug": y_fisher,
                        "tid_aug": tid_fisher,
                        "dir_aug": dir_fisher,
                        "z_trial_macro_f1": float(z_fisher_metrics["trial_macro_f1"]),
                        "z_window_macro_f1": float(z_fisher_metrics["window_macro_f1"]),
                        "delta_vs_step1b": float(z_fisher_metrics["trial_macro_f1"] - z_step1b_metrics["trial_macro_f1"]),
                        "delta_vs_curriculum": float(z_fisher_metrics["trial_macro_f1"] - float(best_round["z_trial_macro_f1"])),
                        "mech": mech_fisher,
                        "dir_summary": _summarize_dir_profile(mech_fisher.get("dir_profile", {})),
                        "direction_usage_entropy": float(fisher_aug_meta.get("direction_usage_entropy", 0.0)),
                        "direction_probs": fisher_aug_meta.get("direction_probs", {}),
                        "gamma_before": {str(i): float(fisher_gamma_before[i]) for i in range(len(fisher_gamma_before))},
                        "gamma_after": {str(i): float(fisher_gamma_after[i]) for i in range(len(fisher_gamma_after))},
                        "direction_state": {str(k): str(v) for k, v in fisher_state_by_dir.items()},
                        "direction_score": {str(k): float(v) for k, v in fisher_score_by_dir.items()},
                        "aug_meta": fisher_aug_meta,
                    }
                )
                gamma_by_dir_fisher = fisher_gamma_after.copy()

            best_fisher_round = _select_best_curriculum_round(fisher_rounds)

        print(f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] raw_only_start", flush=True)
        raw_base_metrics, raw_base_meta = _fit_raw_minirocket(
            dataset=dataset_name,
            train_trials=_records_to_trial_dicts(train_records),
            test_trials=_records_to_trial_dicts(test_records),
            seed=int(seed),
            args=args,
        )
        print(f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] raw_only_done", flush=True)

        print(f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] bridge_single_build_start", flush=True)
        single_aug_trials, single_bridge_meta = _bridge_aug_trials(
            train_records=train_records,
            mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
            z_aug=np.asarray(X_step1b, dtype=np.float32),
            y_aug=np.asarray(y_step1b, dtype=np.int64),
            tid_aug=np.asarray(tid_step1b),
            variant_tag="single_round",
            bridge_eps=float(args.bridge_eps),
        )
        print(
            f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] "
            f"bridge_single_build_done aug_trials={len(single_aug_trials)}",
            flush=True,
        )
        raw_single_metrics, raw_single_meta = _fit_raw_minirocket(
            dataset=dataset_name,
            train_trials=_records_to_trial_dicts(list(train_records) + single_aug_trials),
            test_trials=_records_to_trial_dicts(test_records),
            seed=int(seed),
            args=args,
        )
        print(f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] bridge_single_done", flush=True)

        print(
            f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] "
            f"bridge_multiround_build_start round={int(best_round['round_id'])}",
            flush=True,
        )
        multi_aug_trials, multi_bridge_meta = _bridge_aug_trials(
            train_records=train_records,
            mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
            z_aug=np.asarray(best_round["z_aug"], dtype=np.float32),
            y_aug=np.asarray(best_round["y_aug"], dtype=np.int64),
            tid_aug=np.asarray(best_round["tid_aug"]),
            variant_tag=f"multiround_r{int(best_round['round_id'])}",
            bridge_eps=float(args.bridge_eps),
        )
        print(
            f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] "
            f"bridge_multiround_build_done aug_trials={len(multi_aug_trials)}",
            flush=True,
        )
        raw_multi_metrics, raw_multi_meta = _fit_raw_minirocket(
            dataset=dataset_name,
            train_trials=_records_to_trial_dicts(list(train_records) + multi_aug_trials),
            test_trials=_records_to_trial_dicts(test_records),
            seed=int(seed),
            args=args,
        )
        print(f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] bridge_multiround_done", flush=True)
        fisher_bridge_meta = None
        raw_fisher_metrics = None
        raw_fisher_meta = None
        fisher_aug_trials = []
        if best_fisher_round is not None:
            print(
                f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] "
                f"bridge_fisher_build_start round={int(best_fisher_round['round_id'])}",
                flush=True,
            )
            fisher_aug_trials, fisher_bridge_meta = _bridge_aug_trials(
                train_records=train_records,
                mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
                z_aug=np.asarray(best_fisher_round["z_aug"], dtype=np.float32),
                y_aug=np.asarray(best_fisher_round["y_aug"], dtype=np.int64),
                tid_aug=np.asarray(best_fisher_round["tid_aug"]),
                variant_tag=f"fisher_curriculum_r{int(best_fisher_round['round_id'])}",
                bridge_eps=float(args.bridge_eps),
            )
            print(
                f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] "
                f"bridge_fisher_build_done aug_trials={len(fisher_aug_trials)}",
                flush=True,
            )
            raw_fisher_metrics, raw_fisher_meta = _fit_raw_minirocket(
                dataset=dataset_name,
                train_trials=_records_to_trial_dicts(list(train_records) + fisher_aug_trials),
                test_trials=_records_to_trial_dicts(test_records),
                seed=int(seed),
                args=args,
            )
            print(f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] bridge_fisher_done", flush=True)

        variants = [
            (
                "bridge_single_round",
                None,
                step1b_aug_meta,
                mech_step1b,
                step1b_dir_summary,
                z_step1b_metrics,
                single_bridge_meta,
                raw_single_metrics,
                raw_single_meta,
            ),
            (
                "bridge_multiround_curriculum",
                int(best_round["round_id"]),
                best_round["aug_meta"],
                best_round["mech"],
                best_round["dir_summary"],
                {
                    "trial_macro_f1": best_round["z_trial_macro_f1"],
                    "window_macro_f1": best_round["z_window_macro_f1"],
                },
                multi_bridge_meta,
                raw_multi_metrics,
                raw_multi_meta,
            ),
        ]
        if best_fisher_round is not None and fisher_bridge_meta is not None and raw_fisher_metrics is not None and raw_fisher_meta is not None:
            variants.append(
                (
                    "bridge_fisher_curriculum",
                    int(best_fisher_round["round_id"]),
                    best_fisher_round["aug_meta"],
                    best_fisher_round["mech"],
                    best_fisher_round["dir_summary"],
                    {
                        "trial_macro_f1": best_fisher_round["z_trial_macro_f1"],
                        "window_macro_f1": best_fisher_round["z_window_macro_f1"],
                    },
                    fisher_bridge_meta,
                    raw_fisher_metrics,
                    raw_fisher_meta,
                )
            )

        for target_variant, best_round_id, aug_meta, mech, dir_summary, z_metrics, bridge_meta, raw_metrics, raw_meta in variants:
            target_health_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "target_variant": target_variant,
                    "best_round": "" if best_round_id is None else int(best_round_id),
                    "direction_usage_entropy": _direction_usage_entropy_from_aug_meta(aug_meta),
                    "worst_dir_id": dir_summary.get("worst_dir_id"),
                    "worst_dir_summary": str(dir_summary.get("dir_profile_summary", "n/a")),
                    "direction_health_comment": _target_health_comment(
                        target_variant,
                        dir_summary,
                        _direction_usage_entropy_from_aug_meta(aug_meta),
                        ref_margin=step1b_dir_summary.get("worst_dir_margin_drop"),
                    ),
                    "z_target_trial_macro_f1": float(z_metrics["trial_macro_f1"]),
                    "z_target_window_macro_f1": float(z_metrics["window_macro_f1"]),
                }
            )
            fidelity_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "target_variant": target_variant,
                    "best_round": "" if best_round_id is None else int(best_round_id),
                    "bridge_cov_match_error": float(bridge_meta["bridge_cov_match_error_mean"]),
                    "bridge_cov_to_orig_distance": float(bridge_meta["bridge_cov_to_orig_distance_logeuc_mean"]),
                    "energy_ratio": float(bridge_meta["energy_ratio_mean"]),
                    "cond_A": float(bridge_meta["cond_A_mean"]),
                    "raw_mean_shift_abs": float(bridge_meta["raw_mean_shift_abs_mean"]),
                    "bridge_cov_match_error_fro": float(bridge_meta["bridge_cov_match_error_fro_mean"]),
                    "bridge_cov_match_error_logeuc": float(bridge_meta["bridge_cov_match_error_logeuc_mean"]),
                    "bridge_cov_to_orig_distance_fro": float(bridge_meta["bridge_cov_to_orig_distance_fro_mean"]),
                    "bridge_cov_to_orig_distance_logeuc": float(bridge_meta["bridge_cov_to_orig_distance_logeuc_mean"]),
                    "risk_comment": _risk_comment(bridge_meta),
                }
            )

        seed_summary = {
            "dataset": dataset_name,
            "seed": int(seed),
            "raw_only_acc": float(raw_base_metrics["trial_acc"]),
            "raw_only_f1": float(raw_base_metrics["trial_macro_f1"]),
            "bridge_single_round_acc": float(raw_single_metrics["trial_acc"]),
            "bridge_single_round_f1": float(raw_single_metrics["trial_macro_f1"]),
            "bridge_multiround_acc": float(raw_multi_metrics["trial_acc"]),
            "bridge_multiround_f1": float(raw_multi_metrics["trial_macro_f1"]),
            "delta_vs_raw_only": float(raw_multi_metrics["trial_macro_f1"] - raw_base_metrics["trial_macro_f1"]),
            "delta_vs_bridge_single_round": float(raw_multi_metrics["trial_macro_f1"] - raw_single_metrics["trial_macro_f1"]),
            "best_multiround_round": int(best_round["round_id"]),
            "single_round_target_trial_f1": float(z_step1b_metrics["trial_macro_f1"]),
            "multiround_target_trial_f1": float(best_round["z_trial_macro_f1"]),
        }
        seed_summary["bridge_fisher_curriculum_acc"] = float(raw_fisher_metrics["trial_acc"]) if raw_fisher_metrics is not None else np.nan
        seed_summary["bridge_fisher_curriculum_f1"] = float(raw_fisher_metrics["trial_macro_f1"]) if raw_fisher_metrics is not None else np.nan
        seed_summary["delta_fisher_vs_bridge_multiround"] = (
            float(raw_fisher_metrics["trial_macro_f1"] - raw_multi_metrics["trial_macro_f1"])
            if raw_fisher_metrics is not None
            else np.nan
        )
        seed_summary["best_fisher_round"] = int(best_fisher_round["round_id"]) if best_fisher_round is not None else np.nan
        seed_summary["fisher_target_trial_f1"] = float(best_fisher_round["z_trial_macro_f1"]) if best_fisher_round is not None else np.nan
        seed_summary["result_label"] = _result_label(
            float(seed_summary["delta_fisher_vs_bridge_multiround"])
            if raw_fisher_metrics is not None
            else float(seed_summary["delta_vs_bridge_single_round"])
        )
        per_seed_rows.append(seed_summary)

        seed_payload = {
            "split_meta": split_meta,
            "zspace_baseline_metrics": z_base_metrics,
            "zspace_single_round_step1b_metrics": z_step1b_metrics,
            "zspace_best_multiround_round": {
                "round_id": int(best_round["round_id"]),
                "trial_macro_f1": float(best_round["z_trial_macro_f1"]),
                "window_macro_f1": float(best_round["z_window_macro_f1"]),
                "delta_vs_step1b": float(best_round["delta_vs_step1b"]),
                "direction_probs": best_round["direction_probs"],
                "gamma_before": best_round["gamma_before"],
                "gamma_after": best_round["gamma_after"],
                "direction_state": best_round["direction_state"],
                "direction_score": best_round["direction_score"],
            },
            "raw_only_metrics": raw_base_metrics,
            "raw_only_run_meta": raw_base_meta,
            "bridge_single_round_metrics": raw_single_metrics,
            "bridge_single_round_run_meta": raw_single_meta,
            "bridge_single_round_fidelity": single_bridge_meta,
            "bridge_multiround_metrics": raw_multi_metrics,
            "bridge_multiround_run_meta": raw_multi_meta,
            "bridge_multiround_fidelity": multi_bridge_meta,
        }
        if best_fisher_round is not None and fisher_bridge_meta is not None and raw_fisher_metrics is not None and raw_fisher_meta is not None:
            seed_payload["zspace_best_fisher_round"] = {
                "round_id": int(best_fisher_round["round_id"]),
                "trial_macro_f1": float(best_fisher_round["z_trial_macro_f1"]),
                "window_macro_f1": float(best_fisher_round["z_window_macro_f1"]),
                "delta_vs_step1b": float(best_fisher_round["delta_vs_step1b"]),
                "delta_vs_curriculum": float(best_fisher_round["delta_vs_curriculum"]),
                "direction_probs": best_fisher_round["direction_probs"],
                "gamma_before": best_fisher_round["gamma_before"],
                "gamma_after": best_fisher_round["gamma_after"],
                "direction_state": best_fisher_round["direction_state"],
                "direction_score": best_fisher_round["direction_score"],
            }
            seed_payload["bridge_fisher_curriculum_metrics"] = raw_fisher_metrics
            seed_payload["bridge_fisher_curriculum_run_meta"] = raw_fisher_meta
            seed_payload["bridge_fisher_curriculum_fidelity"] = fisher_bridge_meta
        _write_json(os.path.join(seed_dir, "pilot_run_meta.json"), seed_payload)
        print(
            f"[bridge-curriculum-pilot][{dataset_name}][seed={seed}] "
            f"raw_only_f1={raw_base_metrics['trial_macro_f1']:.4f} "
            f"single_bridge_f1={raw_single_metrics['trial_macro_f1']:.4f} "
            f"multiround_bridge_f1={raw_multi_metrics['trial_macro_f1']:.4f} "
            + (
                f"fisher_bridge_f1={raw_fisher_metrics['trial_macro_f1']:.4f} "
                if raw_fisher_metrics is not None
                else ""
            )
            + f"best_round={int(best_round['round_id'])}",
            flush=True,
        )

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed").reset_index(drop=True)
    per_seed_df.to_csv(os.path.join(args.out_root, output_files["per_seed"]), index=False)

    summary_row = {
        "dataset": dataset_name,
        "raw_only_acc": float(per_seed_df["raw_only_acc"].mean()),
        "raw_only_f1": float(per_seed_df["raw_only_f1"].mean()),
        "raw_only_acc_mean_std": _format_mean_std(per_seed_df["raw_only_acc"]),
        "raw_only_f1_mean_std": _format_mean_std(per_seed_df["raw_only_f1"]),
        "bridge_single_round_acc": float(per_seed_df["bridge_single_round_acc"].mean()),
        "bridge_single_round_f1": float(per_seed_df["bridge_single_round_f1"].mean()),
        "bridge_single_round_acc_mean_std": _format_mean_std(per_seed_df["bridge_single_round_acc"]),
        "bridge_single_round_f1_mean_std": _format_mean_std(per_seed_df["bridge_single_round_f1"]),
        "bridge_multiround_acc": float(per_seed_df["bridge_multiround_acc"].mean()),
        "bridge_multiround_f1": float(per_seed_df["bridge_multiround_f1"].mean()),
        "bridge_multiround_acc_mean_std": _format_mean_std(per_seed_df["bridge_multiround_acc"]),
        "bridge_multiround_f1_mean_std": _format_mean_std(per_seed_df["bridge_multiround_f1"]),
        "delta_vs_raw_only": float(per_seed_df["delta_vs_raw_only"].mean()),
        "delta_vs_bridge_single_round": float(per_seed_df["delta_vs_bridge_single_round"].mean()),
        "best_rounds": "|".join(
            f"{int(row.seed)}:{int(row.best_multiround_round)}"
            for row in per_seed_df.itertuples(index=False)
        ),
    }
    if per_seed_df["bridge_fisher_curriculum_f1"].notna().any():
        fisher_acc_vals = per_seed_df["bridge_fisher_curriculum_acc"].dropna().astype(float)
        fisher_f1_vals = per_seed_df["bridge_fisher_curriculum_f1"].dropna().astype(float)
        summary_row["bridge_fisher_curriculum_acc"] = float(fisher_acc_vals.mean())
        summary_row["bridge_fisher_curriculum_f1"] = float(fisher_f1_vals.mean())
        summary_row["bridge_fisher_curriculum_acc_mean_std"] = _format_mean_std(fisher_acc_vals)
        summary_row["bridge_fisher_curriculum_f1_mean_std"] = _format_mean_std(fisher_f1_vals)
        summary_row["delta_fisher_vs_bridge_multiround"] = float(
            per_seed_df["delta_fisher_vs_bridge_multiround"].dropna().astype(float).mean()
        )
        summary_row["best_fisher_rounds"] = "|".join(
            f"{int(row.seed)}:{int(row.best_fisher_round)}"
            for row in per_seed_df.itertuples(index=False)
            if not pd.isna(row.best_fisher_round)
        )
    summary_row["result_label"] = _result_label(
        float(summary_row["delta_fisher_vs_bridge_multiround"])
        if "delta_fisher_vs_bridge_multiround" in summary_row
        else float(summary_row["delta_vs_bridge_single_round"])
    )
    pd.DataFrame([summary_row]).to_csv(
        os.path.join(args.out_root, output_files["summary"]),
        index=False,
    )

    target_df = pd.DataFrame(target_health_rows).sort_values(["target_variant", "seed"]).reset_index(drop=True)
    target_df.to_csv(
        os.path.join(args.out_root, output_files["target_health"]),
        index=False,
    )
    fidelity_df = pd.DataFrame(fidelity_rows).sort_values(["target_variant", "seed"]).reset_index(drop=True)
    fidelity_df.to_csv(
        os.path.join(args.out_root, output_files["fidelity"]),
        index=False,
    )

    bridge_single_mean = float(per_seed_df["bridge_single_round_f1"].mean())
    bridge_multi_mean = float(per_seed_df["bridge_multiround_f1"].mean())
    bridge_fisher_mean = (
        float(per_seed_df["bridge_fisher_curriculum_f1"].dropna().astype(float).mean())
        if per_seed_df["bridge_fisher_curriculum_f1"].notna().any()
        else np.nan
    )
    delta_vs_single = float(bridge_multi_mean - bridge_single_mean)
    delta_fisher_vs_multi = (
        float(bridge_fisher_mean - bridge_multi_mean)
        if np.isfinite(bridge_fisher_mean)
        else np.nan
    )
    single_fid = fidelity_df[fidelity_df["target_variant"] == "bridge_single_round"]
    multi_fid = fidelity_df[fidelity_df["target_variant"] == "bridge_multiround_curriculum"]
    fidelity_not_worse = bool(
        float(multi_fid["bridge_cov_match_error_logeuc"].mean()) <= float(single_fid["bridge_cov_match_error_logeuc"].mean()) + 1e-6
        and float(multi_fid["cond_A"].mean()) <= float(single_fid["cond_A"].mean()) + 1e-6
        and float(multi_fid["raw_mean_shift_abs"].mean()) <= float(single_fid["raw_mean_shift_abs"].mean()) + 1e-6
    )
    fisher_fid = fidelity_df[fidelity_df["target_variant"] == "bridge_fisher_curriculum"]
    fisher_fidelity_not_worse = bool(
        (not fisher_fid.empty)
        and float(fisher_fid["bridge_cov_match_error_logeuc"].mean()) <= float(multi_fid["bridge_cov_match_error_logeuc"].mean()) + 1e-6
        and float(fisher_fid["cond_A"].mean()) <= float(multi_fid["cond_A"].mean()) + 1e-6
        and float(fisher_fid["raw_mean_shift_abs"].mean()) <= float(multi_fid["raw_mean_shift_abs"].mean()) + 1e-6
    )

    if np.isfinite(delta_fisher_vs_multi):
        if delta_fisher_vs_multi > 1e-6 and fisher_fidelity_not_worse:
            route = "continue_fisher_curriculum_to_bridge"
            label = "fisher_curriculum_bridge_upgrade_positive"
            second_dataset = "yes"
        elif abs(delta_fisher_vs_multi) <= 1e-6 and fisher_fidelity_not_worse:
            route = "fisher_curriculum_bridge_plausible_but_flat"
            label = "fisher_curriculum_bridge_temporarily_flat"
            second_dataset = "no"
        else:
            route = "fisher_curriculum_not_ready_for_bridge_scaleout"
            label = "fisher_curriculum_bridge_negative_or_unclear"
            second_dataset = "no"
    elif delta_vs_single > 1e-6 and fidelity_not_worse:
        route = "continue_curriculum_to_bridge"
        label = "worth_continuing_curriculum_bridge"
        second_dataset = "yes"
    elif abs(delta_vs_single) <= 1e-6 and fidelity_not_worse:
        route = "curriculum_to_bridge_locally_plausible_but_flat"
        label = "temporarily_local_only"
        second_dataset = "no"
    else:
        route = "go_back_to_fisher_c0_plus_curriculum_first"
        label = "not_ready_for_second_bridge_dataset"
        second_dataset = "no"

    md = [
        "# Bridge Curriculum Pilot Conclusion",
        "",
        "This pilot is an independent upgrade-line result.",
        "It is not for the Phase15 mainline freeze table and not for the SEED battlefield.",
        "",
        f"## {dataset_title} Result",
        "",
        f"- `raw_only` F1: `{summary_row['raw_only_f1_mean_std']}`",
        f"- `bridge_single_round` F1: `{summary_row['bridge_single_round_f1_mean_std']}`",
        f"- `bridge_multiround` F1: `{summary_row['bridge_multiround_f1_mean_std']}`",
        f"- `delta_vs_raw_only`: `{summary_row['delta_vs_raw_only']:.6f}`",
        f"- `delta_vs_bridge_single_round`: `{summary_row['delta_vs_bridge_single_round']:.6f}`",
        f"- `best_rounds`: `{summary_row['best_rounds']}`",
        "",
        "## Decision",
        "",
        f"- Current route decision: `{route}`",
        f"- Pilot result label: `{label}`",
        f"- Worth moving to a second dataset now: `{second_dataset}`",
        "",
        "## Interpretation",
        "",
        "- This pilot tests whether a better PIA-side target transfers better through the bridge into raw MiniROCKET training.",
        "- If multiround remains flat while fidelity stays clean, the likely bottleneck is target-to-raw usefulness rather than bridge cleanliness.",
        "- If multiround is better than single-round with no new fidelity risk, curriculum->bridge is the cleaner next path before Fisher/C0.",
    ]
    if np.isfinite(bridge_fisher_mean):
        md[11:11] = [
            f"- `bridge_fisher_curriculum` F1: `{summary_row['bridge_fisher_curriculum_f1_mean_std']}`",
            f"- `delta_fisher_vs_bridge_multiround`: `{summary_row['delta_fisher_vs_bridge_multiround']:.6f}`",
            f"- `best_fisher_rounds`: `{summary_row['best_fisher_rounds']}`",
        ]
        md.extend(
            [
                "- Fisher/C0 + curriculum is only acting on the target side here; raw backbone and bridge protocol remain fixed.",
                "- If fisher+curriculum beats multiround with no new fidelity risk, the current bottleneck is more likely upstream target quality than bridge cleanliness.",
            ]
        )
    Path(os.path.join(args.out_root, output_files["conclusion"])).write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
