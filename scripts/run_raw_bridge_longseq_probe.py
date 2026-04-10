#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from PIA.augment import PIADirectionalAffineAugmenter  # noqa: E402
from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_SEEDIV_ROOT,
    load_trials_for_dataset,
    normalize_dataset_name,
)
from models.raw_cnn1d import RawCNN1D  # noqa: E402
from run_phase15_step0a_paired_lock import (  # noqa: E402
    _aggregate_trials,
    _apply_window_cap,
    _make_trial_split,
)
from run_phase15_step1a_maxplane import _apply_gates, _fit_gate1_from_train  # noqa: E402
from scripts.resource_probe_utils import ResourceProbeLogger  # noqa: E402
from transforms.whiten_color_bridge import bridge_single, logvec_to_spd  # noqa: E402
from run_raw_minirocket_baseline import (  # noqa: E402
    _extract_window_by_start,
    _resolve_window_policy,
    _window_starts,
)
from run_raw_bridge_probe import (  # noqa: E402
    TrialRecord,
    _apply_mean_log,
    _build_trial_records,
    _inner_train_val_split,
)


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


def _count_stats(counts: Dict[str, int]) -> Dict[str, float]:
    if not counts:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    vals = np.asarray(list(counts.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _trial_list_to_class_counts(trials: Sequence[TrialRecord]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tr in trials:
        key = str(int(tr.y))
        out[key] = int(out.get(key, 0) + 1)
    return dict(sorted(out.items()))


def _fit_channel_normalizer_stream(records: Sequence[TrialRecord]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        raise RuntimeError("No records provided to fit channel normalizer.")
    c = int(np.asarray(records[0].x_raw).shape[0])
    sum_x = np.zeros((c, 1), dtype=np.float64)
    sum_x2 = np.zeros((c, 1), dtype=np.float64)
    n_total = 0
    for r in records:
        x = np.asarray(r.x_raw, dtype=np.float64)
        sum_x += np.sum(x, axis=1, keepdims=True)
        sum_x2 += np.sum(x * x, axis=1, keepdims=True)
        n_total += int(x.shape[1])
    mean = sum_x / max(1, n_total)
    var = np.maximum(sum_x2 / max(1, n_total) - mean * mean, 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_window(x_win: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(x_win, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32, copy=False)


def _build_bridge_aug_trials(
    records: Sequence[TrialRecord],
    mean_log_train: np.ndarray,
    *,
    pia_gamma: float,
    pia_n_iters: int,
    pia_activation: str,
    pia_bias_update_mode: str,
    pia_c_repr: float,
    gate1_q: float,
    gate2_q_src: float,
    bridge_eps: float,
    seed: int,
) -> Tuple[List[TrialRecord], Dict[str, object]]:
    z_train = np.stack([r.z for r in records], axis=0).astype(np.float32)
    y_train = np.asarray([int(r.y) for r in records], dtype=np.int64)
    tid_train = np.asarray([r.tid for r in records], dtype=object)
    tid_to_rec = {r.tid: r for r in records}

    aug_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    tid_parts: List[np.ndarray] = []
    src_parts: List[np.ndarray] = []
    recon_last: List[float] = []

    for cls in sorted(np.unique(y_train).tolist()):
        idx = np.where(y_train == int(cls))[0]
        if idx.size < 2:
            continue
        aug = PIADirectionalAffineAugmenter(
            gamma=float(pia_gamma),
            n_iters=int(pia_n_iters),
            activation=str(pia_activation),
            bias_update_mode=str(pia_bias_update_mode),
            C_repr=float(pia_c_repr),
            seed=int(seed + int(cls) * 1009),
        )
        z_cls = z_train[idx]
        z_aug_cls = np.asarray(aug.fit_transform(z_cls), dtype=np.float32)
        st = aug.state()
        recon = st.get("recon_err")
        if isinstance(recon, list) and recon:
            recon_last.append(float(recon[-1]))
        aug_parts.append(z_aug_cls)
        y_parts.append(y_train[idx].copy())
        tid_parts.append(tid_train[idx].copy())
        src_parts.append(z_cls.copy())

    if not aug_parts:
        return [], {"bridge_aug_candidate_count": 0, "bridge_aug_accept_rate": 0.0}

    z_aug = np.vstack(aug_parts).astype(np.float32)
    y_aug = np.concatenate(y_parts).astype(np.int64)
    tid_aug = np.concatenate(tid_parts)
    src_aug = np.vstack(src_parts).astype(np.float32)

    mu_y, tau_y, gate1_meta = _fit_gate1_from_train(z_train, y_train, q=float(gate1_q))
    z_keep, y_keep, tid_keep, _src_keep, gate_meta = _apply_gates(
        z_aug,
        y_aug,
        tid_aug,
        src_aug,
        mu_y=mu_y,
        tau_y=tau_y,
        enable_gate2=True,
        gate2_q_src=float(gate2_q_src),
    )

    aug_trials: List[TrialRecord] = []
    bridge_cov_errors: List[float] = []
    bridge_gain_norms: List[float] = []
    bridge_energy_ratios: List[float] = []

    import torch

    for z_vec, y_val, tid in zip(z_keep, y_keep, tid_keep):
        rec = tid_to_rec[str(tid)]
        sigma_aug = logvec_to_spd(z_vec, mean_log_train)
        x_aug, bmeta = bridge_single(
            torch.from_numpy(np.asarray(rec.x_raw, dtype=np.float32)),
            torch.from_numpy(np.asarray(rec.sigma_orig, dtype=np.float32)),
            torch.from_numpy(np.asarray(sigma_aug, dtype=np.float32)),
            eps=float(bridge_eps),
        )
        aug_trials.append(
            TrialRecord(
                tid=str(rec.tid),
                y=int(y_val),
                x_raw=x_aug.cpu().numpy().astype(np.float32),
                sigma_orig=np.asarray(sigma_aug, dtype=np.float32),
                log_cov=np.asarray(rec.log_cov, dtype=np.float32),
                z=np.asarray(z_vec, dtype=np.float32),
            )
        )
        bridge_cov_errors.append(float(bmeta["bridge_cov_match_error"]))
        bridge_gain_norms.append(float(bmeta["bridge_gain_norm"]))
        bridge_energy_ratios.append(float(bmeta["bridge_energy_ratio"]))

    meta = {
        "bridge_aug_candidate_count": int(len(y_aug)),
        "bridge_aug_accepted_count": int(len(y_keep)),
        "bridge_aug_accept_rate": float(gate_meta.get("accept_rate_final", 0.0)),
        "bridge_cov_match_error_mean": float(np.mean(bridge_cov_errors)) if bridge_cov_errors else 0.0,
        "bridge_cov_match_error_std": float(np.std(bridge_cov_errors)) if bridge_cov_errors else 0.0,
        "bridge_cov_match_error_max": float(np.max(bridge_cov_errors)) if bridge_cov_errors else 0.0,
        "bridge_gain_norm_mean": float(np.mean(bridge_gain_norms)) if bridge_gain_norms else 0.0,
        "bridge_energy_ratio_mean": float(np.mean(bridge_energy_ratios)) if bridge_energy_ratios else 0.0,
        "pia_recon_last_mean": float(np.mean(recon_last)) if recon_last else 0.0,
        "gate1": gate1_meta,
        "gate_meta": gate_meta,
    }
    return aug_trials, meta


@dataclass
class WindowIndex:
    source_kind: np.ndarray
    trial_pos: np.ndarray
    start_pos: np.ndarray
    short_flags: np.ndarray
    y: np.ndarray
    tid: np.ndarray
    per_trial_counts_before: Dict[str, int]
    n_short_trials: int


def _build_window_index(
    records: Sequence[TrialRecord],
    win_len: int,
    hop_len: int,
    *,
    source_kind: int,
) -> WindowIndex:
    source_parts: List[int] = []
    trial_pos_parts: List[int] = []
    start_parts: List[int] = []
    short_parts: List[bool] = []
    y_parts: List[int] = []
    tid_parts: List[str] = []
    per_trial_counts_before: Dict[str, int] = {}
    n_short_trials = 0

    for i, rec in enumerate(records):
        x = np.asarray(rec.x_raw, dtype=np.float32)
        starts, is_short = _window_starts(int(x.shape[1]), int(win_len), int(hop_len))
        tid = str(rec.tid)
        per_trial_counts_before[tid] = int(starts.size)
        if is_short:
            n_short_trials += 1
        for s in starts.tolist():
            source_parts.append(int(source_kind))
            trial_pos_parts.append(int(i))
            start_parts.append(int(s))
            short_parts.append(bool(is_short))
            y_parts.append(int(rec.y))
            tid_parts.append(tid)

    return WindowIndex(
        source_kind=np.asarray(source_parts, dtype=np.int8),
        trial_pos=np.asarray(trial_pos_parts, dtype=np.int32),
        start_pos=np.asarray(start_parts, dtype=np.int32),
        short_flags=np.asarray(short_parts, dtype=bool),
        y=np.asarray(y_parts, dtype=np.int64),
        tid=np.asarray(tid_parts, dtype=object),
        per_trial_counts_before=per_trial_counts_before,
        n_short_trials=int(n_short_trials),
    )


def _cap_window_index(
    index: WindowIndex,
    *,
    cap_k: int,
    seed: int,
) -> Tuple[WindowIndex, Dict[str, object]]:
    n_total = int(index.y.shape[0])
    if n_total <= 0:
        raise RuntimeError("No windows found while applying cap.")

    if int(cap_k) <= 0:
        counts_after = dict(index.per_trial_counts_before)
        meta = {
            "nominal_cap_K": int(cap_k),
            "effective_cap_K": 0,
            "total_windows_before_cap": int(n_total),
            "total_windows": int(n_total),
            "per_trial_counts_before": index.per_trial_counts_before,
            "per_trial_counts_after": counts_after,
            "window_count_stats_after_cap": _count_stats(counts_after),
            "n_short_trials": int(index.n_short_trials),
        }
        return index, meta

    max_before = max(index.per_trial_counts_before.values()) if index.per_trial_counts_before else 0
    effective_cap_k = int(min(int(cap_k), int(max_before))) if int(cap_k) > 0 else 0
    dummy_x = np.arange(n_total, dtype=np.int64).reshape(-1, 1)
    dummy_selected, y_cap, tid_cap, _is_aug_cap, per_trial_after, _aug_ratio = _apply_window_cap(
        X=dummy_x,
        y=index.y,
        tid=index.tid,
        cap_k=effective_cap_k,
        seed=int(seed),
        is_aug=np.zeros((n_total,), dtype=bool),
        policy="random",
    )
    keep = np.asarray(dummy_selected[:, 0], dtype=np.int64)
    capped = WindowIndex(
        source_kind=index.source_kind[keep],
        trial_pos=index.trial_pos[keep],
        start_pos=index.start_pos[keep],
        short_flags=index.short_flags[keep],
        y=np.asarray(y_cap, dtype=np.int64),
        tid=np.asarray(tid_cap, dtype=object),
        per_trial_counts_before=index.per_trial_counts_before,
        n_short_trials=int(index.n_short_trials),
    )
    meta = {
        "nominal_cap_K": int(cap_k),
        "effective_cap_K": int(effective_cap_k),
        "total_windows_before_cap": int(n_total),
        "total_windows": int(capped.y.shape[0]),
        "per_trial_counts_before": index.per_trial_counts_before,
        "per_trial_counts_after": per_trial_after,
        "window_count_stats_after_cap": _count_stats(per_trial_after),
        "n_short_trials": int(index.n_short_trials),
    }
    return capped, meta


def _concat_window_indices(parts: Sequence[WindowIndex]) -> WindowIndex:
    non_empty = [p for p in parts if int(p.y.shape[0]) > 0]
    if not non_empty:
        raise RuntimeError("No window indices to concatenate.")
    per_trial_counts_before: Dict[str, int] = {}
    n_short_trials = 0
    for p in non_empty:
        per_trial_counts_before.update(p.per_trial_counts_before)
        n_short_trials += int(p.n_short_trials)
    return WindowIndex(
        source_kind=np.concatenate([p.source_kind for p in non_empty], axis=0),
        trial_pos=np.concatenate([p.trial_pos for p in non_empty], axis=0),
        start_pos=np.concatenate([p.start_pos for p in non_empty], axis=0),
        short_flags=np.concatenate([p.short_flags for p in non_empty], axis=0),
        y=np.concatenate([p.y for p in non_empty], axis=0),
        tid=np.concatenate([p.tid for p in non_empty], axis=0),
        per_trial_counts_before=per_trial_counts_before,
        n_short_trials=int(n_short_trials),
    )


class TrialWindowDataset:
    def __init__(
        self,
        *,
        orig_records: Sequence[TrialRecord],
        aug_records: Sequence[TrialRecord],
        index: WindowIndex,
        win_len: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.orig_records = list(orig_records)
        self.aug_records = list(aug_records)
        self.index = index
        self.win_len = int(win_len)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.index.y.shape[0])

    def __getitem__(self, idx: int):
        src = int(self.index.source_kind[idx])
        pos = int(self.index.trial_pos[idx])
        start = int(self.index.start_pos[idx])
        is_short = bool(self.index.short_flags[idx])
        rec = self.orig_records[pos] if src == 0 else self.aug_records[pos]
        x_win = _extract_window_by_start(
            x_trial=np.asarray(rec.x_raw, dtype=np.float32),
            start=start,
            win_len=int(self.win_len),
            is_short=is_short,
        )
        x_norm = _normalize_window(x_win, self.mean, self.std)
        return x_norm, int(self.index.y[idx]), str(self.index.tid[idx])


def _build_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    import torch
    from torch.utils.data import DataLoader

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _predict_loader(model, loader, dev, *, amp_enabled: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_tid: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for xb, yb, tidb in loader:
            xb = xb.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(amp_enabled)):
                logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            all_true.append(np.asarray(yb.cpu().numpy(), dtype=np.int64))
            all_pred.append(np.asarray(pred.cpu().numpy(), dtype=np.int64))
            all_scores.append(np.asarray(logits.float().cpu().numpy(), dtype=np.float32))
            all_tid.append(np.asarray(list(tidb), dtype=object))

    return (
        np.concatenate(all_true, axis=0).astype(np.int64, copy=False),
        np.concatenate(all_pred, axis=0).astype(np.int64, copy=False),
        np.concatenate(all_scores, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_tid, axis=0).astype(object, copy=False),
    )


def _evaluate_loader(model, loader, dev, *, amp_enabled: bool, agg_mode: str) -> Dict[str, object]:
    y_true, y_pred, scores, tid = _predict_loader(model, loader, dev, amp_enabled=bool(amp_enabled))
    y_true_trial, y_pred_trial = _aggregate_trials(
        y_true_win=y_true,
        y_pred_win=y_pred,
        scores_win=scores,
        tid_win=tid,
        mode=str(agg_mode),
    )
    return {
        "window_acc": float(accuracy_score(y_true, y_pred)),
        "window_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "trial_acc": float(accuracy_score(y_true_trial, y_pred_trial)),
        "trial_macro_f1": float(f1_score(y_true_trial, y_pred_trial, average="macro")),
        "window_count": int(y_true.shape[0]),
        "trial_count": int(y_true_trial.shape[0]),
    }


def _train_eval_raw_cnn_windowed(
    *,
    train_ds,
    val_ds,
    test_ds,
    seed: int,
    in_channels: int,
    num_classes: int,
    hidden_channels: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    val_every: int,
    num_workers: int,
    device: str,
    amp: bool,
    agg_mode: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = True

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    pin_memory = bool(dev.type == "cuda")
    train_loader = _build_loader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
    )
    test_loader = _build_loader(
        test_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
    )

    model = RawCNN1D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        hidden_channels=int(hidden_channels),
        dropout=float(dropout),
    ).to(dev)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and dev.type == "cuda"))

    best_state = None
    best_epoch = 0
    best_val = {"trial_macro_f1": -1.0, "trial_acc": -1.0}
    stale = 0
    history: List[Dict[str, float]] = []
    fit_t0 = time.perf_counter()

    for epoch in range(int(epochs)):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for xb, yb, _tid in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(amp and dev.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss became non-finite at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.detach().cpu().item()) * int(yb.shape[0])
            n_seen += int(yb.shape[0])

        should_eval = ((epoch + 1) % max(1, int(val_every)) == 0) or (epoch == int(epochs) - 1)
        if not should_eval:
            continue

        val_metrics = _evaluate_loader(model, val_loader, dev, amp_enabled=bool(amp and dev.type == "cuda"), agg_mode=agg_mode)
        mean_loss = epoch_loss / max(1, n_seen)
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(mean_loss),
                "val_trial_macro_f1": float(val_metrics["trial_macro_f1"]),
                "val_trial_acc": float(val_metrics["trial_acc"]),
                "val_window_macro_f1": float(val_metrics["window_macro_f1"]),
                "val_window_acc": float(val_metrics["window_acc"]),
            }
        )
        print(
            f"[train][epoch={epoch + 1}] "
            f"loss={mean_loss:.4f} val_trial_f1={val_metrics['trial_macro_f1']:.4f} "
            f"val_trial_acc={val_metrics['trial_acc']:.4f}",
            flush=True,
        )

        better = (
            float(val_metrics["trial_macro_f1"]) > float(best_val["trial_macro_f1"]) + 1e-12
            or (
                abs(float(val_metrics["trial_macro_f1"]) - float(best_val["trial_macro_f1"])) <= 1e-12
                and float(val_metrics["trial_acc"]) > float(best_val["trial_acc"])
            )
        )
        if better:
            best_val = {
                "trial_macro_f1": float(val_metrics["trial_macro_f1"]),
                "trial_acc": float(val_metrics["trial_acc"]),
            }
            best_epoch = int(epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    fit_elapsed = float(time.perf_counter() - fit_t0)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate_loader(model, test_loader, dev, amp_enabled=bool(amp and dev.type == "cuda"), agg_mode=agg_mode)
    cuda_max_alloc_gb = 0.0
    cuda_max_reserved_gb = 0.0
    if dev.type == "cuda":
        cuda_max_alloc_gb = float(torch.cuda.max_memory_allocated(dev)) / (1024**3)
        cuda_max_reserved_gb = float(torch.cuda.max_memory_reserved(dev)) / (1024**3)

    train_meta = {
        "best_epoch": int(best_epoch),
        "best_val_trial_macro_f1": float(best_val["trial_macro_f1"]),
        "best_val_trial_acc": float(best_val["trial_acc"]),
        "fit_sec": float(fit_elapsed),
        "device": str(dev),
        "cuda_max_alloc_gb": float(cuda_max_alloc_gb),
        "cuda_max_reserved_gb": float(cuda_max_reserved_gb),
        "history": history,
    }
    return test_metrics, train_meta


def _refresh_dataset_summary(dataset_out_dir: str) -> None:
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(Path(dataset_out_dir).glob("seed*/E*/metrics.json")):
        run_meta_path = metrics_path.parent / "run_meta.json"
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            run_meta = {}
            if run_meta_path.is_file():
                with open(run_meta_path, "r", encoding="utf-8") as f:
                    run_meta = json.load(f)
            row = dict(metrics)
            row["source_dir"] = str(metrics_path.parent)
            row["split_hash"] = run_meta.get("split_hash", "")
            row["peak_rss_gb"] = run_meta.get("resource_summary", {}).get("peak_rss_gb")
            row["cuda_max_alloc_gb"] = run_meta.get("train_meta", {}).get("cuda_max_alloc_gb")
            rows.append(row)
        except Exception:
            continue

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(["dataset", "seed", "experiment"]).reset_index(drop=True)
    df.to_csv(os.path.join(dataset_out_dir, "summary_per_run.csv"), index=False)

    agg = (
        df.groupby("experiment", as_index=False)
        .agg(
            trial_macro_f1_mean=("trial_macro_f1", "mean"),
            trial_macro_f1_std=("trial_macro_f1", "std"),
            trial_acc_mean=("trial_acc", "mean"),
            trial_acc_std=("trial_acc", "std"),
            peak_rss_gb_mean=("peak_rss_gb", "mean"),
            cuda_max_alloc_gb_mean=("cuda_max_alloc_gb", "mean"),
        )
    )
    agg.to_csv(os.path.join(dataset_out_dir, "summary_agg.csv"), index=False)


def _load_resource_summary(exp_dir: str) -> Dict[str, object]:
    path = os.path.join(exp_dir, "resource_summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_experiment(
    *,
    dataset: str,
    seed: int,
    exp_name: str,
    all_trials: Sequence[Dict],
    args,
) -> Dict[str, object]:
    exp_dir = os.path.join(args.out_root, dataset, f"seed{seed}", exp_name)
    _ensure_dir(exp_dir)
    probe = ResourceProbeLogger(exp_dir) if bool(args.resource_probe) else None
    current_stage = "init"

    try:
        current_stage = "split_trials"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        train_trials, test_trials, split_meta = _make_trial_split(list(all_trials), int(seed))
        train_core_trials, val_core_trials = _inner_train_val_split(
            train_trials,
            seed=int(seed),
            val_fraction=float(args.val_fraction),
        )
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=(
                    f"train_core_trials={len(train_core_trials)} "
                    f"val_trials={len(val_core_trials)} test_trials={len(test_trials)}"
                ),
            )

        current_stage = "build_geometry"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        train_core_tmp, mean_log_train = _build_trial_records(train_core_trials, spd_eps=float(args.spd_eps))
        val_tmp, _ = _build_trial_records(val_core_trials, spd_eps=float(args.spd_eps))
        test_tmp, _ = _build_trial_records(test_trials, spd_eps=float(args.spd_eps))
        train_core = _apply_mean_log(train_core_tmp, mean_log_train)
        val_core = _apply_mean_log(val_tmp, mean_log_train)
        test_records = _apply_mean_log(test_tmp, mean_log_train)
        norm_mean, norm_std = _fit_channel_normalizer_stream(train_core)
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"train_core={len(train_core)}")

        aug_records: List[TrialRecord] = []
        aug_meta: Dict[str, object] = {}
        if exp_name == "E3_raw_bridge_geom_aug":
            current_stage = "build_bridge"
            if probe is not None:
                probe.mark_stage_start(current_stage)
            aug_records, aug_meta = _build_bridge_aug_trials(
                train_core,
                mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
                pia_gamma=float(args.pia_gamma),
                pia_n_iters=int(args.pia_n_iters),
                pia_activation=str(args.pia_activation),
                pia_bias_update_mode=str(args.pia_bias_update_mode),
                pia_c_repr=float(args.pia_c_repr),
                gate1_q=float(args.gate1_q),
                gate2_q_src=float(args.gate2_q_src),
                bridge_eps=float(args.bridge_eps),
                seed=int(seed),
            )
            if probe is not None:
                probe.mark_stage_end(
                    current_stage,
                    note=(
                        f"aug_trials={len(aug_records)} "
                        f"bridge_cov_err_mean={float(aug_meta.get('bridge_cov_match_error_mean', 0.0)):.6f}"
                    ),
                )

        current_stage = "build_window_indices"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        policy = _resolve_window_policy(
            dataset=dataset,
            train_trials=train_core_trials,
            fixed_window_sec=float(args.window_sec),
            fixed_hop_sec=float(args.hop_sec),
            prop_win_ratio=0.5,
            prop_hop_ratio=0.25,
            min_win_len=32,
            min_hop_len=8,
        )
        win_len = int(policy["window_len_samples"])
        hop_len = int(policy["hop_len_samples"])

        train_orig_idx = _build_window_index(train_core, win_len=win_len, hop_len=hop_len, source_kind=0)
        train_orig_idx_cap, train_orig_meta = _cap_window_index(
            train_orig_idx,
            cap_k=int(args.train_cap_k),
            seed=int(seed) ^ 0x101,
        )
        train_indices = [train_orig_idx_cap]
        train_aug_meta = {
            "nominal_cap_K": 0,
            "effective_cap_K": 0,
            "total_windows_before_cap": 0,
            "total_windows": 0,
            "per_trial_counts_before": {},
            "per_trial_counts_after": {},
            "window_count_stats_after_cap": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "n_short_trials": 0,
        }
        if aug_records:
            train_aug_idx = _build_window_index(aug_records, win_len=win_len, hop_len=hop_len, source_kind=1)
            train_aug_idx_cap, train_aug_meta = _cap_window_index(
                train_aug_idx,
                cap_k=int(args.aug_cap_k),
                seed=int(seed) ^ 0x202,
            )
            train_indices.append(train_aug_idx_cap)
        train_idx = _concat_window_indices(train_indices)

        val_idx = _build_window_index(val_core, win_len=win_len, hop_len=hop_len, source_kind=0)
        val_idx_cap, val_meta = _cap_window_index(
            val_idx,
            cap_k=int(args.val_cap_k),
            seed=int(seed) ^ 0x303,
        )
        test_idx = _build_window_index(test_records, win_len=win_len, hop_len=hop_len, source_kind=0)
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=(
                    f"train_windows={len(train_idx.y)} val_windows={len(val_idx_cap.y)} "
                    f"test_windows={len(test_idx.y)}"
                ),
            )

        current_stage = "build_datasets"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        train_ds = TrialWindowDataset(
            orig_records=train_core,
            aug_records=aug_records,
            index=train_idx,
            win_len=win_len,
            mean=norm_mean,
            std=norm_std,
        )
        val_ds = TrialWindowDataset(
            orig_records=val_core,
            aug_records=[],
            index=val_idx_cap,
            win_len=win_len,
            mean=norm_mean,
            std=norm_std,
        )
        test_ds = TrialWindowDataset(
            orig_records=test_records,
            aug_records=[],
            index=test_idx,
            win_len=win_len,
            mean=norm_mean,
            std=norm_std,
        )
        if probe is not None:
            probe.mark_stage_end(current_stage, note=f"in_channels={int(train_core[0].x_raw.shape[0])}")

        current_stage = "model_fit"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        metrics, train_meta = _train_eval_raw_cnn_windowed(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            seed=int(seed),
            in_channels=int(train_core[0].x_raw.shape[0]),
            num_classes=int(len(sorted(set(int(r.y) for r in train_core)))),
            hidden_channels=int(args.hidden_channels),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            eval_batch_size=int(args.eval_batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            val_every=int(args.val_every),
            num_workers=int(args.num_workers),
            device=str(args.device),
            amp=bool(args.amp),
            agg_mode=str(args.aggregation_mode),
        )
        if probe is not None:
            probe.mark_stage_end(
                current_stage,
                note=(
                    f"fit_sec={float(train_meta['fit_sec']):.2f} "
                    f"best_epoch={int(train_meta['best_epoch'])}"
                ),
            )

        current_stage = "write_outputs"
        if probe is not None:
            probe.mark_stage_start(current_stage)
        metric_payload = {
            "dataset": str(dataset),
            "seed": int(seed),
            "experiment": str(exp_name),
            "trial_acc": float(metrics["trial_acc"]),
            "trial_macro_f1": float(metrics["trial_macro_f1"]),
            "window_acc": float(metrics["window_acc"]),
            "window_macro_f1": float(metrics["window_macro_f1"]),
            "train_orig_count": int(len(train_core)),
            "train_aug_count": int(len(aug_records)),
            "val_count": int(len(val_core)),
            "test_count": int(len(test_records)),
            "train_orig_windows": int(train_orig_meta["total_windows"]),
            "train_aug_windows": int(train_aug_meta["total_windows"]),
            "val_windows": int(val_meta["total_windows"]),
            "test_windows": int(test_idx.y.shape[0]),
            "fit_sec": float(train_meta["fit_sec"]),
            "best_epoch": int(train_meta["best_epoch"]),
            "best_val_trial_macro_f1": float(train_meta["best_val_trial_macro_f1"]),
            "best_val_trial_acc": float(train_meta["best_val_trial_acc"]),
            "cuda_max_alloc_gb": float(train_meta["cuda_max_alloc_gb"]),
            "bridge_cov_match_error_mean": float(aug_meta.get("bridge_cov_match_error_mean", 0.0)),
            "bridge_cov_match_error_std": float(aug_meta.get("bridge_cov_match_error_std", 0.0)),
            "bridge_cov_match_error_max": float(aug_meta.get("bridge_cov_match_error_max", 0.0)),
            "bridge_gain_norm_mean": float(aug_meta.get("bridge_gain_norm_mean", 0.0)),
            "bridge_energy_ratio_mean": float(aug_meta.get("bridge_energy_ratio_mean", 0.0)),
            "bridge_aug_accept_rate": float(aug_meta.get("bridge_aug_accept_rate", 0.0)),
        }
        _write_json(os.path.join(exp_dir, "metrics.json"), metric_payload)
        run_meta = {
            "dataset": str(dataset),
            "seed": int(seed),
            "experiment": str(exp_name),
            "split_hash": split_meta["split_hash"],
            "split_meta": split_meta,
            "window_policy": policy,
            "train_class_counts": _trial_list_to_class_counts(train_core),
            "val_class_counts": _trial_list_to_class_counts(val_core),
            "test_class_counts": _trial_list_to_class_counts(test_records),
            "train_orig_window_meta": train_orig_meta,
            "train_aug_window_meta": train_aug_meta,
            "val_window_meta": val_meta,
            "test_window_total": int(test_idx.y.shape[0]),
            "train_meta": train_meta,
            "augmentation_meta": aug_meta,
            "bridge_enabled": bool(exp_name == "E3_raw_bridge_geom_aug"),
            "resource_summary": _load_resource_summary(exp_dir),
            "config": {
                "window_sec": float(args.window_sec),
                "hop_sec": float(args.hop_sec),
                "train_cap_k": int(args.train_cap_k),
                "aug_cap_k": int(args.aug_cap_k),
                "val_cap_k": int(args.val_cap_k),
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "num_workers": int(args.num_workers),
                "device": str(args.device),
                "amp": bool(args.amp),
                "aggregation_mode": str(args.aggregation_mode),
            },
        }
        _write_json(os.path.join(exp_dir, "run_meta.json"), run_meta)
        if probe is not None:
            probe.mark_stage_end(current_stage, note="metrics.json + run_meta.json")
            probe.mark_success()

        print(
            f"[{dataset}][seed={seed}][{exp_name}] "
            f"trial_f1={metric_payload['trial_macro_f1']:.4f} "
            f"trial_acc={metric_payload['trial_acc']:.4f} "
            f"test_windows={metric_payload['test_windows']}",
            flush=True,
        )
        return metric_payload
    except Exception as exc:
        if probe is not None:
            probe.mark_failure(current_stage, exc)
        raise


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Long-sequence raw-bridge probe for seed1 / seediv.")
    p.add_argument("--datasets", nargs="+", default=["seed1"])
    p.add_argument("--seeds", nargs="+", type=int, default=[3])
    p.add_argument("--experiments", nargs="+", default=["E1_raw_only", "E3_raw_bridge_geom_aug"])
    p.add_argument("--out-root", type=str, default="out/raw_bridge_longseq_probe")

    p.add_argument("--processed-root", type=str, default="data/SEED/SEED_EEG/Preprocessed_EEG")
    p.add_argument("--stim-xlsx", type=str, default="data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    p.add_argument("--seediv-root", type=str, default=DEFAULT_SEEDIV_ROOT)

    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--train-cap-k", type=int, default=120)
    p.add_argument("--aug-cap-k", type=int, default=120)
    p.add_argument("--val-cap-k", type=int, default=16)

    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--bridge-eps", type=float, default=1e-4)
    p.add_argument("--pia-gamma", type=float, default=0.10)
    p.add_argument("--pia-n-iters", type=int, default=2)
    p.add_argument("--pia-activation", type=str, default="sine")
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--gate1-q", type=float, default=95.0)
    p.add_argument("--gate2-q-src", type=float, default=95.0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--aggregation-mode", type=str, default="majority", choices=["majority", "meanlogit"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resource-probe", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    experiments = [str(x) for x in args.experiments]
    valid_experiments = {"E1_raw_only", "E3_raw_bridge_geom_aug"}
    for exp in experiments:
        if exp not in valid_experiments:
            raise ValueError(f"Unsupported experiment: {exp}")

    for dataset_raw in args.datasets:
        dataset = normalize_dataset_name(dataset_raw)
        if dataset not in {"seed1", "seediv"}:
            raise ValueError(f"run_raw_bridge_longseq_probe supports seed1/seediv only, got {dataset_raw}")
        load_kwargs = {"dataset": dataset}
        if dataset == "seed1":
            load_kwargs["processed_root"] = str(args.processed_root)
            load_kwargs["stim_xlsx"] = str(args.stim_xlsx)
        else:
            load_kwargs["seediv_root"] = str(args.seediv_root)

        print(f"[load] dataset={dataset}", flush=True)
        all_trials = load_trials_for_dataset(**load_kwargs)
        dataset_out_dir = os.path.join(args.out_root, dataset)
        _ensure_dir(dataset_out_dir)

        for seed in args.seeds:
            for exp_name in experiments:
                _run_experiment(
                    dataset=dataset,
                    seed=int(seed),
                    exp_name=str(exp_name),
                    all_trials=all_trials,
                    args=args,
                )
                _refresh_dataset_summary(dataset_out_dir)


if __name__ == "__main__":
    main()
