#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from PIA.augment import PIADirectionalAffineAugmenter  # noqa: E402
from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_FINGERMOVEMENTS_ROOT,
    DEFAULT_HAR_ROOT,
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    load_trials_for_dataset,
)
from models.raw_dcnet_adapter import RawDCNetAdapter, RawDCNetTemporalAdapter  # noqa: E402
from models.raw_cnn1d import RawCNN1D  # noqa: E402
from scripts.legacy_phase.run_phase14r_step6b1_rev2 import logm_spd, vec_utri  # noqa: E402
from scripts.legacy_phase.run_phase15_step0a_paired_lock import _make_trial_split  # noqa: E402
from scripts.legacy_phase.run_phase15_step1a_maxplane import _apply_gates, _fit_gate1_from_train  # noqa: E402
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


def _covariance_from_trial(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_c = x - x.mean(axis=1, keepdims=True)
    denom = max(1, int(x.shape[1]) - 1)
    cov = (x_c @ x_c.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(eps) * np.eye(cov.shape[0], dtype=np.float64)
    return cov.astype(np.float32)


@dataclass
class TrialRecord:
    tid: str
    y: int
    x_raw: np.ndarray
    sigma_orig: np.ndarray
    log_cov: np.ndarray
    z: np.ndarray


def _build_trial_records(trials: Sequence[Dict], spd_eps: float) -> Tuple[List[TrialRecord], np.ndarray]:
    covs = [_covariance_from_trial(np.asarray(t["x_trial"], dtype=np.float32), spd_eps) for t in trials]
    log_covs = [logm_spd(np.asarray(c, dtype=np.float64), spd_eps).astype(np.float32) for c in covs]
    mean_log = np.mean(np.stack(log_covs, axis=0), axis=0).astype(np.float32)
    out: List[TrialRecord] = []
    for t, cov, log_cov in zip(trials, covs, log_covs):
        z = vec_utri(log_cov - mean_log).astype(np.float32)
        out.append(
            TrialRecord(
                tid=str(t["trial_id_str"]),
                y=int(t["label"]),
                x_raw=np.asarray(t["x_trial"], dtype=np.float32),
                sigma_orig=np.asarray(cov, dtype=np.float32),
                log_cov=np.asarray(log_cov, dtype=np.float32),
                z=np.asarray(z, dtype=np.float32),
            )
        )
    return out, mean_log


def _apply_mean_log(records: Sequence[TrialRecord], mean_log: np.ndarray) -> List[TrialRecord]:
    out: List[TrialRecord] = []
    for r in records:
        z = vec_utri(np.asarray(r.log_cov, dtype=np.float64) - np.asarray(mean_log, dtype=np.float64)).astype(np.float32)
        out.append(
            TrialRecord(
                tid=r.tid,
                y=r.y,
                x_raw=r.x_raw,
                sigma_orig=r.sigma_orig,
                log_cov=r.log_cov,
                z=z,
            )
        )
    return out


def _inner_train_val_split(trials: Sequence[Dict], seed: int, val_fraction: float) -> Tuple[List[Dict], List[Dict]]:
    if not (0.0 < float(val_fraction) < 1.0):
        return list(trials), []
    y = np.asarray([int(t["label"]) for t in trials], dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=int(seed))
    idx_train, idx_val = next(splitter.split(np.zeros((len(trials), 1)), y))
    train_core = [trials[int(i)] for i in idx_train.tolist()]
    val_core = [trials[int(i)] for i in idx_val.tolist()]
    return train_core, val_core


def _stack_raw(xs: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float32) for x in xs], axis=0)


def _fit_channel_normalizer(train_x: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    cat = np.concatenate([np.asarray(x, dtype=np.float32) for x in train_x], axis=1)
    mean = np.mean(cat, axis=1, keepdims=True).astype(np.float32)
    std = np.std(cat, axis=1, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _normalize_raw_batch(x: Sequence[np.ndarray], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = _stack_raw(x)
    return (arr - mean[None, :, :]) / std[None, :, :]


def _encode_labels(labels: Sequence[int]) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    uniq = sorted(set(int(v) for v in labels))
    lab2idx = {int(v): i for i, v in enumerate(uniq)}
    idx2lab = {i: int(v) for i, v in enumerate(uniq)}
    y = np.asarray([lab2idx[int(v)] for v in labels], dtype=np.int64)
    return y, lab2idx, idx2lab


def _decode_labels(y_idx: np.ndarray, idx2lab: Dict[int, int]) -> np.ndarray:
    return np.asarray([idx2lab[int(v)] for v in np.asarray(y_idx).ravel().tolist()], dtype=np.int64)


def _train_eval_raw_cnn(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    hidden_channels: int,
    device: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = RawCNN1D(
        in_channels=int(train_x.shape[1]),
        num_classes=int(len(np.unique(train_y))),
        hidden_channels=int(hidden_channels),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long()),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(torch.from_numpy(test_x).float(), batch_size=int(batch_size), shuffle=False)

    best_state = None
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = -1
    stale = 0

    def _predict(loader) -> np.ndarray:
        preds: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                logits = model(xb.to(dev))
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.int64)

    for epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        val_pred = _predict(val_loader)
        val_acc = float(accuracy_score(val_y, val_pred))
        val_f1 = float(f1_score(val_y, val_pred, average="macro"))
        better = (val_f1 > best_val_f1 + 1e-12) or (
            abs(val_f1 - best_val_f1) <= 1e-12 and val_acc > best_val_acc
        )
        if better:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = _predict(test_loader)
    return test_pred, {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "device": str(dev),
    }


def _train_eval_raw_dcnet(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = RawDCNetAdapter(
        in_channels=int(train_x.shape[1]),
        seq_len=int(train_x.shape[2]),
        num_classes=int(len(np.unique(train_y))),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long()),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(torch.from_numpy(test_x).float(), batch_size=int(batch_size), shuffle=False)

    best_state = None
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = -1
    stale = 0

    def _predict(loader) -> np.ndarray:
        preds: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                logits = model(xb.to(dev))
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.int64)

    for epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"DCNet loss became non-finite at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_pred = _predict(val_loader)
        val_acc = float(accuracy_score(val_y, val_pred))
        val_f1 = float(f1_score(val_y, val_pred, average="macro"))
        better = (val_f1 > best_val_f1 + 1e-12) or (
            abs(val_f1 - best_val_f1) <= 1e-12 and val_acc > best_val_acc
        )
        if better:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = _predict(test_loader)
    return test_pred, {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "device": str(dev),
        "input_dim": int(train_x.shape[1] * train_x.shape[2]),
    }


def _train_eval_raw_dcnet_tproj(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    proj_channels: int,
    proj_bins: int,
    device: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = RawDCNetTemporalAdapter(
        in_channels=int(train_x.shape[1]),
        seq_len=int(train_x.shape[2]),
        num_classes=int(len(np.unique(train_y))),
        proj_channels=int(proj_channels),
        proj_bins=int(proj_bins),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long()),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(torch.from_numpy(test_x).float(), batch_size=int(batch_size), shuffle=False)

    best_state = None
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = -1
    stale = 0

    def _predict(loader) -> np.ndarray:
        preds: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                logits = model(xb.to(dev))
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.int64)

    for epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"DCNet temporal adapter loss became non-finite at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_pred = _predict(val_loader)
        val_acc = float(accuracy_score(val_y, val_pred))
        val_f1 = float(f1_score(val_y, val_pred, average="macro"))
        better = (val_f1 > best_val_f1 + 1e-12) or (
            abs(val_f1 - best_val_f1) <= 1e-12 and val_acc > best_val_acc
        )
        if better:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = _predict(test_loader)
    return test_pred, {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "device": str(dev),
        "input_dim": int(proj_channels * proj_bins),
        "proj_channels": int(proj_channels),
        "proj_bins": int(proj_bins),
    }


def _build_naive_aug(records: Sequence[TrialRecord], noise_scale: float, seed: int) -> Tuple[List[np.ndarray], List[int], Dict[str, float]]:
    rs = np.random.RandomState(int(seed))
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for r in records:
        x = np.asarray(r.x_raw, dtype=np.float32)
        sigma = np.std(x, axis=1, keepdims=True).astype(np.float32)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        noise = rs.normal(size=x.shape).astype(np.float32) * sigma * float(noise_scale)
        xs.append((x + noise).astype(np.float32))
        ys.append(int(r.y))
    return xs, ys, {"naive_aug_count": int(len(xs)), "naive_noise_scale": float(noise_scale)}


def _build_bridge_aug(
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
) -> Tuple[List[np.ndarray], List[int], Dict[str, object]]:
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
        return [], [], {"bridge_aug_candidate_count": 0, "bridge_aug_accept_rate": 0.0}

    z_aug = np.vstack(aug_parts).astype(np.float32)
    y_aug = np.concatenate(y_parts).astype(np.int64)
    tid_aug = np.concatenate(tid_parts)
    src_aug = np.vstack(src_parts).astype(np.float32)

    mu_y, tau_y, gate1_meta = _fit_gate1_from_train(z_train, y_train, q=float(gate1_q))
    z_keep, y_keep, tid_keep, src_keep, gate_meta = _apply_gates(
        z_aug,
        y_aug,
        tid_aug,
        src_aug,
        mu_y=mu_y,
        tau_y=tau_y,
        enable_gate2=True,
        gate2_q_src=float(gate2_q_src),
    )

    xs_aug: List[np.ndarray] = []
    ys_aug: List[int] = []
    bridge_cov_errors_rel: List[float] = []
    bridge_cov_errors_fro: List[float] = []
    bridge_cov_errors_logeuc: List[float] = []
    bridge_cov_to_orig_fro: List[float] = []
    bridge_cov_to_orig_logeuc: List[float] = []
    bridge_gain_norms: List[float] = []
    bridge_energy_ratios: List[float] = []
    bridge_cond_As: List[float] = []
    sigma_orig_min_eigs: List[float] = []
    sigma_orig_max_eigs: List[float] = []
    raw_mean_shifts: List[float] = []
    classwise_mean_shift: Dict[str, List[float]] = {}

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
        xs_aug.append(x_aug.cpu().numpy().astype(np.float32))
        ys_aug.append(int(y_val))
        bridge_cov_errors_rel.append(float(bmeta["bridge_cov_match_error"]))
        bridge_cov_errors_fro.append(float(bmeta["bridge_cov_match_error_fro"]))
        bridge_cov_errors_logeuc.append(float(bmeta["bridge_cov_match_error_logeuc"]))
        bridge_cov_to_orig_fro.append(float(bmeta["bridge_cov_to_orig_distance_fro"]))
        bridge_cov_to_orig_logeuc.append(float(bmeta["bridge_cov_to_orig_distance_logeuc"]))
        bridge_gain_norms.append(float(bmeta["bridge_gain_norm"]))
        bridge_energy_ratios.append(float(bmeta["bridge_energy_ratio"]))
        bridge_cond_As.append(float(bmeta["bridge_cond_A"]))
        sigma_orig_min_eigs.append(float(bmeta["sigma_orig_min_eig"]))
        sigma_orig_max_eigs.append(float(bmeta["sigma_orig_max_eig"]))
        raw_mean_shifts.append(float(bmeta["raw_mean_shift_abs"]))
        classwise_mean_shift.setdefault(str(int(y_val)), []).append(float(bmeta["raw_mean_shift_abs"]))

    orig_counts = {str(int(cls)): int(np.sum(y_train == int(cls))) for cls in sorted(np.unique(y_train).tolist())}
    accepted_counts = {str(int(cls)): int(np.sum(np.asarray(ys_aug, dtype=np.int64) == int(cls))) for cls in sorted(np.unique(y_train).tolist())}
    orig_total = float(sum(orig_counts.values()))
    accepted_total = float(sum(accepted_counts.values()))
    orig_share = {k: (v / orig_total if orig_total > 0 else 0.0) for k, v in orig_counts.items()}
    accepted_share = {k: (accepted_counts.get(k, 0) / accepted_total if accepted_total > 0 else 0.0) for k in orig_counts}
    class_balance_shift = {k: float(accepted_share.get(k, 0.0) - orig_share.get(k, 0.0)) for k in orig_counts}
    class_balance_shift_max_abs = max((abs(v) for v in class_balance_shift.values()), default=0.0)
    classwise_mean_shift_summary = {
        k: float(np.mean(v)) if v else 0.0 for k, v in sorted(classwise_mean_shift.items(), key=lambda kv: int(kv[0]))
    }

    meta = {
        "bridge_aug_candidate_count": int(len(y_aug)),
        "bridge_aug_accepted_count": int(len(y_keep)),
        "bridge_aug_accept_rate": float(gate_meta.get("accept_rate_final", 0.0)),
        "train_selected_aug_ratio": float(len(xs_aug) / max(1, len(records))),
        "bridge_cov_match_error_mean": float(np.mean(bridge_cov_errors_rel)) if bridge_cov_errors_rel else 0.0,
        "bridge_cov_match_error_std": float(np.std(bridge_cov_errors_rel)) if bridge_cov_errors_rel else 0.0,
        "bridge_cov_match_error_fro_mean": float(np.mean(bridge_cov_errors_fro)) if bridge_cov_errors_fro else 0.0,
        "bridge_cov_match_error_fro_std": float(np.std(bridge_cov_errors_fro)) if bridge_cov_errors_fro else 0.0,
        "bridge_cov_match_error_logeuc_mean": float(np.mean(bridge_cov_errors_logeuc)) if bridge_cov_errors_logeuc else 0.0,
        "bridge_cov_match_error_logeuc_std": float(np.std(bridge_cov_errors_logeuc)) if bridge_cov_errors_logeuc else 0.0,
        "bridge_cov_to_orig_distance_fro_mean": float(np.mean(bridge_cov_to_orig_fro)) if bridge_cov_to_orig_fro else 0.0,
        "bridge_cov_to_orig_distance_fro_std": float(np.std(bridge_cov_to_orig_fro)) if bridge_cov_to_orig_fro else 0.0,
        "bridge_cov_to_orig_distance_logeuc_mean": float(np.mean(bridge_cov_to_orig_logeuc)) if bridge_cov_to_orig_logeuc else 0.0,
        "bridge_cov_to_orig_distance_logeuc_std": float(np.std(bridge_cov_to_orig_logeuc)) if bridge_cov_to_orig_logeuc else 0.0,
        "bridge_gain_norm_mean": float(np.mean(bridge_gain_norms)) if bridge_gain_norms else 0.0,
        "bridge_energy_ratio_mean": float(np.mean(bridge_energy_ratios)) if bridge_energy_ratios else 0.0,
        "bridge_cond_A_mean": float(np.mean(bridge_cond_As)) if bridge_cond_As else 0.0,
        "bridge_cond_A_std": float(np.std(bridge_cond_As)) if bridge_cond_As else 0.0,
        "sigma_orig_min_eig_mean": float(np.mean(sigma_orig_min_eigs)) if sigma_orig_min_eigs else 0.0,
        "sigma_orig_max_eig_mean": float(np.mean(sigma_orig_max_eigs)) if sigma_orig_max_eigs else 0.0,
        "raw_mean_shift_abs_mean": float(np.mean(raw_mean_shifts)) if raw_mean_shifts else 0.0,
        "raw_mean_shift_abs_max": float(np.max(raw_mean_shifts)) if raw_mean_shifts else 0.0,
        "class_balance_shift_max_abs": float(class_balance_shift_max_abs),
        "class_balance_shift_summary": _compact_json(class_balance_shift),
        "classwise_mean_shift_summary": _compact_json(classwise_mean_shift_summary),
        "trial_window_gap_status": "not_applicable_raw_trial_model",
        "pia_recon_last_mean": float(np.mean(recon_last)) if recon_last else 0.0,
        "class_balance_orig_counts": orig_counts,
        "class_balance_aug_counts": accepted_counts,
        "class_balance_orig_share": orig_share,
        "class_balance_aug_share": accepted_share,
        "class_balance_shift": class_balance_shift,
        "classwise_mean_shift": classwise_mean_shift_summary,
        "gate1": gate1_meta,
        "gate_meta": gate_meta,
    }
    return xs_aug, ys_aug, meta


def _run_one_experiment(
    *,
    exp_name: str,
    train_core: Sequence[TrialRecord],
    val_core: Sequence[TrialRecord],
    test_records: Sequence[TrialRecord],
    seed: int,
    mean_log_train: np.ndarray,
    args,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    train_x = [r.x_raw for r in train_core]
    train_y_raw = [int(r.y) for r in train_core]
    aug_x: List[np.ndarray] = []
    aug_y_raw: List[int] = []
    aug_meta: Dict[str, object] = {}

    if exp_name == "E2_raw_naive_aug":
        aug_x, aug_y_raw, aug_meta = _build_naive_aug(train_core, noise_scale=float(args.naive_noise_scale), seed=int(seed))
    elif exp_name == "E3_raw_bridge_geom_aug":
        aug_x, aug_y_raw, aug_meta = _build_bridge_aug(
            train_core,
            mean_log_train,
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

    norm_mean, norm_std = _fit_channel_normalizer(train_x)
    train_all_x = train_x + aug_x
    train_all_y = train_y_raw + aug_y_raw
    y_train_idx, lab2idx, idx2lab = _encode_labels(train_all_y)
    y_val_idx = np.asarray([lab2idx[int(r.y)] for r in val_core], dtype=np.int64)
    y_test_idx = np.asarray([lab2idx[int(r.y)] for r in test_records], dtype=np.int64)

    x_train = _normalize_raw_batch(train_all_x, norm_mean, norm_std).astype(np.float32)
    x_val = _normalize_raw_batch([r.x_raw for r in val_core], norm_mean, norm_std).astype(np.float32)
    x_test = _normalize_raw_batch([r.x_raw for r in test_records], norm_mean, norm_std).astype(np.float32)

    if str(args.backbone) == "rawcnn1d":
        y_pred_idx, train_meta = _train_eval_raw_cnn(
            train_x=x_train,
            train_y=y_train_idx,
            val_x=x_val,
            val_y=y_val_idx,
            test_x=x_test,
            seed=int(seed),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            patience=int(args.patience),
            hidden_channels=int(args.hidden_channels),
            device=str(args.device),
        )
    elif str(args.backbone) == "dcnet":
        y_pred_idx, train_meta = _train_eval_raw_dcnet(
            train_x=x_train,
            train_y=y_train_idx,
            val_x=x_val,
            val_y=y_val_idx,
            test_x=x_test,
            seed=int(seed),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            patience=int(args.patience),
            device=str(args.device),
        )
    elif str(args.backbone) == "dcnet_tproj":
        y_pred_idx, train_meta = _train_eval_raw_dcnet_tproj(
            train_x=x_train,
            train_y=y_train_idx,
            val_x=x_val,
            val_y=y_val_idx,
            test_x=x_test,
            seed=int(seed),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            patience=int(args.patience),
            proj_channels=int(args.dcnet_proj_channels),
            proj_bins=int(args.dcnet_proj_bins),
            device=str(args.device),
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")
    y_pred = _decode_labels(y_pred_idx, idx2lab)
    y_true = np.asarray([int(r.y) for r in test_records], dtype=np.int64)

    metrics = {
        "dataset": str(args.dataset),
        "seed": int(seed),
        "experiment": exp_name,
        "trial_acc": float(accuracy_score(y_true, y_pred)),
        "trial_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "train_orig_count": int(len(train_core)),
        "train_aug_count": int(len(aug_x)),
        "test_count": int(len(test_records)),
    }
    metrics.update({k: v for k, v in aug_meta.items() if isinstance(v, (int, float, str, bool))})

    run_meta = {
        "dataset": str(args.dataset),
        "seed": int(seed),
        "experiment": exp_name,
        "geometry_mode": "trial_cov_no_bands",
        "split_policy": "phase15_trial_split_lock",
        "inner_val_fraction": float(args.val_fraction),
        "pia_mode": "class_local_single_direction" if exp_name == "E3_raw_bridge_geom_aug" else "none",
        "gate_mode": "gate1_gate2" if exp_name == "E3_raw_bridge_geom_aug" else "none",
        "bridge_enabled": bool(exp_name == "E3_raw_bridge_geom_aug"),
        "bridge_eps": float(args.bridge_eps),
        "raw_backbone": str(args.backbone),
        "raw_backbone_params": {
            "hidden_channels": int(args.hidden_channels),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "patience": int(args.patience),
            "dcnet_proj_channels": int(args.dcnet_proj_channels),
            "dcnet_proj_bins": int(args.dcnet_proj_bins),
        },
        "label_map": {str(k): int(v) for k, v in lab2idx.items()},
        "train_meta": train_meta,
        "augmentation_meta": aug_meta,
    }
    return metrics, run_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal raw+bridge geometry probe.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="natops",
        choices=["natops", "har", "fingermovements", "selfregulationscp1"],
    )
    parser.add_argument("--seeds", type=str, default="3")
    parser.add_argument("--out-root", type=str, default="out/raw_bridge_probe")
    parser.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    parser.add_argument("--har-root", type=str, default=DEFAULT_HAR_ROOT)
    parser.add_argument("--fingermovements-root", type=str, default=DEFAULT_FINGERMOVEMENTS_ROOT)
    parser.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--spd-eps", type=float, default=1e-4)
    parser.add_argument("--bridge-eps", type=float, default=1e-4)
    parser.add_argument("--pia-gamma", type=float, default=0.10)
    parser.add_argument("--pia-n-iters", type=int, default=2)
    parser.add_argument("--pia-activation", type=str, default="sine")
    parser.add_argument("--pia-bias-update-mode", type=str, default="residual")
    parser.add_argument("--pia-c-repr", type=float, default=1.0)
    parser.add_argument("--gate1-q", type=float, default=95.0)
    parser.add_argument("--gate2-q-src", type=float, default=95.0)
    parser.add_argument("--naive-noise-scale", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="rawcnn1d", choices=["rawcnn1d", "dcnet", "dcnet_tproj"])
    parser.add_argument("--dcnet-proj-channels", type=int, default=16)
    parser.add_argument("--dcnet-proj-bins", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-naive", action="store_true")
    parser.add_argument("--only-experiments", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")

    load_kwargs = {"dataset": str(args.dataset)}
    if str(args.dataset) == "natops":
        load_kwargs["natops_root"] = str(args.natops_root)
    elif str(args.dataset) == "har":
        load_kwargs["har_root"] = str(args.har_root)
    elif str(args.dataset) == "fingermovements":
        load_kwargs["fingermovements_root"] = str(args.fingermovements_root)
    elif str(args.dataset) == "selfregulationscp1":
        load_kwargs["selfregulationscp1_root"] = str(args.selfregulationscp1_root)
    else:
        raise ValueError(f"Unsupported dataset for raw bridge probe: {args.dataset}")
    all_trials = load_trials_for_dataset(**load_kwargs)

    all_rows: List[Dict[str, object]] = []
    experiments = ["E1_raw_only", "E3_raw_bridge_geom_aug"]
    if not bool(args.skip_naive):
        experiments.insert(1, "E2_raw_naive_aug")
    if str(args.only_experiments).strip():
        req = [s.strip() for s in str(args.only_experiments).split(",") if s.strip()]
        experiments = [e for e in experiments if e in req]
        if not experiments:
            raise ValueError("--only-experiments did not select any valid experiments")

    for seed in seeds:
        train_trials, test_trials, split_meta = _make_trial_split(all_trials, int(seed))
        train_core_trials, val_core_trials = _inner_train_val_split(train_trials, seed=int(seed), val_fraction=float(args.val_fraction))
        train_core_tmp, mean_log_train = _build_trial_records(train_core_trials, spd_eps=float(args.spd_eps))
        val_tmp, _ = _build_trial_records(val_core_trials, spd_eps=float(args.spd_eps))
        test_tmp, _ = _build_trial_records(test_trials, spd_eps=float(args.spd_eps))
        train_core = _apply_mean_log(train_core_tmp, mean_log_train)
        val_core = _apply_mean_log(val_tmp, mean_log_train)
        test_records = _apply_mean_log(test_tmp, mean_log_train)

        for exp_name in experiments:
            exp_dir = os.path.join(args.out_root, str(args.dataset), f"seed{seed}", exp_name)
            _ensure_dir(exp_dir)
            metrics, run_meta = _run_one_experiment(
                exp_name=exp_name,
                train_core=train_core,
                val_core=val_core,
                test_records=test_records,
                seed=int(seed),
                mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
                args=args,
            )
            run_meta["split_meta"] = split_meta
            _write_json(os.path.join(exp_dir, "metrics.json"), metrics)
            _write_json(os.path.join(exp_dir, "run_meta.json"), run_meta)
            row = dict(metrics)
            row["source_dir"] = exp_dir
            all_rows.append(row)
            print(
                f"[{args.dataset}][seed={seed}][{exp_name}] "
                f"f1={metrics['trial_macro_f1']:.4f} acc={metrics['trial_acc']:.4f}",
                flush=True,
            )

    summary_df = pd.DataFrame(all_rows)
    out_dataset_dir = os.path.join(args.out_root, str(args.dataset))
    _ensure_dir(out_dataset_dir)
    summary_df.to_csv(os.path.join(out_dataset_dir, "summary_per_run.csv"), index=False)
    if not summary_df.empty:
        agg = (
            summary_df.groupby("experiment", as_index=False)
            .agg(
                trial_macro_f1_mean=("trial_macro_f1", "mean"),
                trial_macro_f1_std=("trial_macro_f1", "std"),
                trial_acc_mean=("trial_acc", "mean"),
                trial_acc_std=("trial_acc", "std"),
            )
        )
        agg.to_csv(os.path.join(out_dataset_dir, "summary_agg.csv"), index=False)


if __name__ == "__main__":
    main()
