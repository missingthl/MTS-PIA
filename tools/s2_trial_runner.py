from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

from datasets.official_mat_dataset import OfficialMatSequenceDataset
from datasets.manifold_streaming_riemann import RiemannianUtils
from manifold_raw.features import cov_shrink, vec_utri
from manifold_raw.spd_eps import compute_spd_eps


def _subject_split(
    trials: List[dict], split_seed: int, val_ratio: float
) -> Tuple[List[int], List[int], List[str], List[str]]:
    subjects = sorted({str(t["subject"]) for t in trials})
    rng = np.random.default_rng(int(split_seed))
    rng.shuffle(subjects)
    val_size = int(len(subjects) * val_ratio)
    if val_ratio > 0.0 and val_size == 0 and len(subjects) > 1:
        val_size = 1
    if val_size >= len(subjects):
        val_size = len(subjects) - 1 if len(subjects) > 1 else 0
    val_subjects = subjects[:val_size]
    train_subjects = subjects[val_size:]
    train_idx = [i for i, t in enumerate(trials) if str(t["subject"]) in train_subjects]
    val_idx = [i for i, t in enumerate(trials) if str(t["subject"]) in val_subjects]
    overlap = set(train_subjects) & set(val_subjects)
    if overlap:
        raise ValueError(f"Subject leakage detected: {sorted(overlap)}")
    return train_idx, val_idx, train_subjects, val_subjects


def _agg_stats(values: List[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p100": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p100": float(np.max(arr)),
    }


def _spd_stats(cov: np.ndarray) -> Dict[str, float]:
    sym_err = float(np.linalg.norm(cov - cov.T, ord="fro"))
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    cond = float(max_eig / max(min_eig, 1e-30))
    return {
        "symmetry_error": sym_err,
        "min_eig": min_eig,
        "max_eig": max_eig,
        "cond": cond,
    }


def _tsm_vec(tsm: torch.Tensor) -> np.ndarray:
    mat = tsm.cpu().numpy()
    return vec_utri(mat)


def _build_cov(X: np.ndarray, cov_method: str) -> np.ndarray:
    if cov_method:
        return cov_shrink(X, method=cov_method)
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = max(1, Xc.shape[0] - 1)
    cov = (Xc.T @ Xc) / float(denom)
    return 0.5 * (cov + cov.T)


def _label_hist(labels: np.ndarray) -> List[int]:
    hist = np.bincount(labels, minlength=3)
    return [int(v) for v in hist.tolist()]


def _scores(model, X):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = model.predict_proba(X)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return scores


def _apply_spd_kind(
    cov: np.ndarray,
    kind: str,
    eps_params: dict,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float], float, float, float]:
    cov = 0.5 * (cov + cov.T)
    dim = cov.shape[0]
    if kind == "corr":
        diag = np.diag(cov)
        diag = np.maximum(diag, 1e-12)
        inv_std = 1.0 / np.sqrt(diag)
        base = cov * inv_std[None, :] * inv_std[:, None]
        base = 0.5 * (base + base.T)
    elif kind == "trace_norm":
        tr = float(np.trace(cov))
        if tr <= 0.0 or not np.isfinite(tr):
            tr = 1.0
        base = cov / tr
        base = 0.5 * (base + base.T)
    else:
        base = cov

    stats_before = _spd_stats(base)
    eps, base_val = compute_spd_eps(base, **eps_params)
    cov_after = base + float(eps) * np.eye(dim)
    stats_after = _spd_stats(cov_after)
    trace_base = float(np.trace(base)) if np.isfinite(np.trace(base)) else 0.0
    eps_ratio = float((eps * dim) / max(trace_base, 1e-30))
    return cov_after, stats_before, stats_after, float(eps), float(base_val), eps_ratio


def _logits_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden1: int,
        hidden2: int,
        drop1: float,
        drop2: float,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 runner (official mat -> SPD -> TSM).")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--feature-key", default="de_LDS1")
    parser.add_argument("--split-by", default="subject")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--classifier", choices=["logreg", "linearsvc", "mlp"], default="logreg")
    parser.add_argument("--s2-mode", choices=["full_spd", "diag_only"], default="full_spd")
    parser.add_argument("--data-mode", choices=["trial_spd", "window_spd"], default="trial_spd")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--max-ref-windows", type=int, default=2000)
    parser.add_argument("--max-windows-per-trial", type=int, default=0)
    parser.add_argument("--band-mode", choices=["all", "gamma"], default="all")
    parser.add_argument("--gamma-index", type=int, default=4)
    parser.add_argument("--spd-kind", choices=["cov", "corr", "trace_norm"], default="cov")
    parser.add_argument("--cov-method", default="shrinkage_oas")
    parser.add_argument("--spd-eps-mode", default="relative_trace")
    parser.add_argument("--spd-eps-alpha", type=float, default=1e-2)
    parser.add_argument("--spd-eps-absolute", type=float, default=1e-5)
    parser.add_argument("--spd-eps-floor-mult", type=float, default=1e-6)
    parser.add_argument("--spd-eps-ceil-mult", type=float, default=1e-1)
    parser.add_argument("--eps-diag", type=float, default=1e-12)
    parser.add_argument("--cache-features", type=int, default=0)
    parser.add_argument("--mlp-hidden1", type=int, default=256)
    parser.add_argument("--mlp-hidden2", type=int, default=64)
    parser.add_argument("--mlp-dropout1", type=float, default=0.5)
    parser.add_argument("--mlp-dropout2", type=float, default=0.3)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-3)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument("--mlp-seed", type=int, default=42)
    parser.add_argument("--mlp-device", type=str, default="cpu")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    if args.split_by != "subject":
        raise ValueError("Only split_by=subject is supported for S2 runner")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_root = None
    try:
        manifest_data = json.loads(Path(args.manifest_path).read_text())
        if isinstance(manifest_data, dict):
            manifest_root = manifest_data.get("root_dir")
    except Exception:
        manifest_root = None

    if args.band_mode == "gamma" and not (0 <= args.gamma_index <= 4):
        raise ValueError("gamma-index must be in [0,4]")
    band_indices = list(range(5)) if args.band_mode == "all" else [args.gamma_index]
    spd_kind = args.spd_kind

    eps_params = {
        "mode": args.spd_eps_mode,
        "absolute": args.spd_eps_absolute,
        "alpha": args.spd_eps_alpha,
        "floor_mult": args.spd_eps_floor_mult,
        "ceil_mult": args.spd_eps_ceil_mult,
    }

    dataset = OfficialMatSequenceDataset(
        manifest_path=args.manifest_path,
        feature_key=args.feature_key,
        mode=args.data_mode,
        window_size=args.window_size,
        window_step=args.window_step,
        verbose=False,
    )
    trials = dataset.trials
    if not trials:
        raise ValueError("No trials loaded from manifest")

    train_idx, val_idx, train_subjects, val_subjects = _subject_split(
        trials, args.split_seed, args.val_ratio
    )

    shapes_sample = []
    labels_arr = np.asarray([int(t["label"]) for t in trials], dtype=np.int64)
    spd_stats_train: Dict[str, List[float]] = {
        "min_eig": [],
        "min_eig_before": [],
        "cond": [],
        "symmetry_error": [],
        "eps_trace_ratio": [],
    }
    spd_stats_val: Dict[str, List[float]] = {
        "min_eig": [],
        "min_eig_before": [],
        "cond": [],
        "symmetry_error": [],
        "eps_trace_ratio": [],
    }
    eps_used: List[float] = []
    eps_base: List[float] = []

    t_stats = {}
    tsm_audit = {}
    window_audit = {}

    if args.data_mode == "trial_spd":
        t_list: List[int] = []
        spd_all: List[np.ndarray] = []
        for idx in range(len(dataset)):
            X, label, meta = dataset[idx]
            if idx < 5:
                shapes_sample.append(
                    {
                        "trial_id": meta["trial"],
                        "file_path": meta["mat_path"],
                        "raw_shape": list(meta["raw_shape"]),
                        "final_shape": list(meta["final_shape"]),
                        "T": int(X.shape[0]),
                    }
                )
            if not np.all(np.isfinite(X)):
                raise ValueError(f"NaN/Inf in trial {meta}")
            t_list.append(int(X.shape[0]))

            cov_bands = []
            for b in band_indices:
                xb = X[:, :, b].T
                cov_raw = _build_cov(xb, args.cov_method)
                cov, stats_before, stats_after, eps, base, eps_ratio = _apply_spd_kind(
                    cov_raw, spd_kind, eps_params
                )
                if stats_after["min_eig"] <= 0.0 or not np.isfinite(stats_after["min_eig"]):
                    raise ValueError(
                        f"Non-SPD after eps: trial={meta} band={b} min_eig={stats_after['min_eig']:.3e}"
                    )
                if idx in train_idx:
                    spd_stats_train["min_eig"].append(stats_after["min_eig"])
                    spd_stats_train["min_eig_before"].append(stats_before["min_eig"])
                    spd_stats_train["cond"].append(stats_after["cond"])
                    spd_stats_train["symmetry_error"].append(stats_after["symmetry_error"])
                    spd_stats_train["eps_trace_ratio"].append(eps_ratio)
                else:
                    spd_stats_val["min_eig"].append(stats_after["min_eig"])
                    spd_stats_val["min_eig_before"].append(stats_before["min_eig"])
                    spd_stats_val["cond"].append(stats_after["cond"])
                    spd_stats_val["symmetry_error"].append(stats_after["symmetry_error"])
                    spd_stats_val["eps_trace_ratio"].append(eps_ratio)
                eps_used.append(float(eps))
                eps_base.append(float(base))
                cov_bands.append(cov.astype(np.float32, copy=False))
            cov_arr = np.stack(cov_bands, axis=0)
            if cov_arr.shape != (len(band_indices), 62, 62):
                raise ValueError(f"Unexpected SPD shape {cov_arr.shape} for trial {meta}")
            spd_all.append(cov_arr)

        t_stats = {
            "min": int(np.min(t_list)),
            "mean": float(np.mean(t_list)),
            "max": int(np.max(t_list)),
        }
        spd_arr = np.stack(spd_all, axis=0)  # [N, B, 62, 62]
        train_covs = spd_arr[train_idx]
        ref_means = []
        for b in range(len(band_indices)):
            cov_t = torch.as_tensor(train_covs[:, b], dtype=torch.float64)
            ref = RiemannianUtils.cal_riemannian_mean(cov_t)
            ref_means.append(ref)
        ref_mean = torch.stack(ref_means, dim=0)
        for b in range(len(band_indices)):
            np.save(outdir / f"ref_spd_band{b}.npy", ref_mean[b].cpu().numpy())

        tsm_norms = []
        tsm_nan = 0
        tsm_inf = 0
        tsm_dim = None
        feats_full = []
        feats_diag = []
        for cov_bands in spd_arr:
            diag_feat = []
            tsm_feat = []
            for b in range(len(band_indices)):
                cov = cov_bands[b]
                diag = np.diag(cov)
                diag = np.maximum(diag, args.eps_diag)
                diag_feat.append(np.log(diag))

                cov_t = torch.as_tensor(cov, dtype=torch.float32)
                with torch.no_grad():
                    tsm = RiemannianUtils.tangent_space_mapping(cov_t, ref_mean[b])
                if not torch.isfinite(tsm).all():
                    tsm_nan += int(torch.isnan(tsm).any().item())
                    tsm_inf += int(torch.isinf(tsm).any().item())
                tsm_norms.append(float(torch.linalg.norm(tsm).item()))
                vec = _tsm_vec(tsm)
                tsm_feat.append(vec)
                if tsm_dim is None:
                    tsm_dim = int(vec.shape[0])
            feats_full.append(np.concatenate(tsm_feat, axis=0))
            feats_diag.append(np.concatenate(diag_feat, axis=0))

        if tsm_nan or tsm_inf:
            raise ValueError(f"TSM produced NaN/Inf (nan={tsm_nan} inf={tsm_inf})")

        tsm_audit = {
            "tsm_dim": int(tsm_dim or 0),
            "tsm_norm": _agg_stats(tsm_norms),
            "nan_count": int(tsm_nan),
            "inf_count": int(tsm_inf),
        }
        (outdir / "tsm_audit.json").write_text(json.dumps(tsm_audit, indent=2))

        if args.s2_mode == "full_spd":
            X_feat = np.stack(feats_full, axis=0)
        else:
            X_feat = np.stack(feats_diag, axis=0)
        y = labels_arr
        X_train = X_feat[train_idx]
        y_train = y[train_idx]
        X_val = X_feat[val_idx]
        y_val = y[val_idx]
        meta_train = [
            {"trial_id": trials[idx]["trial"], "subject": trials[idx]["subject"]} for idx in train_idx
        ]
        meta_val = [
            {"trial_id": trials[idx]["trial"], "subject": trials[idx]["subject"]} for idx in val_idx
        ]
    else:
        rng = np.random.default_rng(int(args.split_seed))
        sample_covs: List[List[np.ndarray]] = [[] for _ in range(len(band_indices))]
        seen = 0
        window_counts: Dict[Tuple[str, int], int] = {}

        for idx in train_idx:
            windows, _, meta = dataset[idx]
            if idx < 5:
                shapes_sample.append(
                    {
                        "trial_id": meta["trial"],
                        "file_path": meta["mat_path"],
                        "raw_shape": list(meta["raw_shape"]),
                        "final_shape": list(meta["final_shape"]),
                        "T": int(windows.shape[1]),
                    }
                )
            if args.s2_mode == "diag_only":
                continue
            if args.max_windows_per_trial and windows.shape[0] > args.max_windows_per_trial:
                select = rng.choice(windows.shape[0], size=args.max_windows_per_trial, replace=False)
                windows = windows[select]
            for w in windows:
                for b in band_indices:
                    xb = w[:, :, b].T
                    cov_raw = _build_cov(xb, args.cov_method)
                    cov, _, _, eps, base, _ = _apply_spd_kind(cov_raw, spd_kind, eps_params)
                    seen += 1
                    idx_b = band_indices.index(b)
                    if len(sample_covs[idx_b]) < args.max_ref_windows:
                        sample_covs[idx_b].append(cov.astype(np.float32, copy=False))
                    else:
                        j = rng.integers(0, seen)
                        if j < args.max_ref_windows:
                            sample_covs[idx_b][j] = cov.astype(np.float32, copy=False)

        ref_mean = None
        if args.s2_mode == "full_spd":
            ref_means = []
            for b in range(len(band_indices)):
                if not sample_covs[b]:
                    raise ValueError("No covariances sampled for ref_mean in window mode")
                cov_t = torch.as_tensor(np.stack(sample_covs[b], axis=0), dtype=torch.float64)
                ref = RiemannianUtils.cal_riemannian_mean(cov_t)
                ref_means.append(ref)
            ref_mean = torch.stack(ref_means, dim=0)

        tsm_norms = []
        tsm_nan = 0
        tsm_inf = 0
        tsm_dim = None

        def _build_window_features(indices: List[int]):
            nonlocal tsm_dim, tsm_nan, tsm_inf, tsm_norms
            feats = []
            labels_w = []
            meta_w = []
            for idx in indices:
                windows, label, meta = dataset[idx]
                trial_id = meta["trial"]
                subject = meta["subject"]
                key = (str(subject), int(trial_id))
                if args.max_windows_per_trial and windows.shape[0] > args.max_windows_per_trial:
                    select = rng.choice(windows.shape[0], size=args.max_windows_per_trial, replace=False)
                    windows = windows[select]
                if key not in window_counts:
                    window_counts[key] = int(windows.shape[0])
                for w_idx, w in enumerate(windows):
                    diag_feat = []
                    tsm_feat = []
                    for b_idx, b in enumerate(band_indices):
                        xb = w[:, :, b].T
                        cov_raw = _build_cov(xb, args.cov_method)
                        cov, stats_before, stats_after, eps, base, eps_ratio = _apply_spd_kind(
                            cov_raw, spd_kind, eps_params
                        )
                        if idx in train_idx:
                            spd_stats_train["min_eig"].append(stats_after["min_eig"])
                            spd_stats_train["min_eig_before"].append(stats_before["min_eig"])
                            spd_stats_train["cond"].append(stats_after["cond"])
                            spd_stats_train["symmetry_error"].append(stats_after["symmetry_error"])
                            spd_stats_train["eps_trace_ratio"].append(eps_ratio)
                        else:
                            spd_stats_val["min_eig"].append(stats_after["min_eig"])
                            spd_stats_val["min_eig_before"].append(stats_before["min_eig"])
                            spd_stats_val["cond"].append(stats_after["cond"])
                            spd_stats_val["symmetry_error"].append(stats_after["symmetry_error"])
                            spd_stats_val["eps_trace_ratio"].append(eps_ratio)
                        eps_used.append(float(eps))
                        eps_base.append(float(base))
                        diag = np.diag(cov)
                        diag = np.maximum(diag, args.eps_diag)
                        diag_feat.append(np.log(diag))
                        if args.s2_mode == "full_spd":
                            cov_t = torch.as_tensor(cov, dtype=torch.float32)
                            with torch.no_grad():
                                tsm = RiemannianUtils.tangent_space_mapping(cov_t, ref_mean[b_idx])
                            if not torch.isfinite(tsm).all():
                                tsm_nan += int(torch.isnan(tsm).any().item())
                                tsm_inf += int(torch.isinf(tsm).any().item())
                            tsm_norms.append(float(torch.linalg.norm(tsm).item()))
                            vec = _tsm_vec(tsm)
                            tsm_feat.append(vec)
                            if tsm_dim is None:
                                tsm_dim = int(vec.shape[0])
                    if args.s2_mode == "diag_only":
                        feat = np.concatenate(diag_feat, axis=0)
                    else:
                        feat = np.concatenate(tsm_feat, axis=0)
                    if not np.all(np.isfinite(feat)):
                        raise ValueError(f"Non-finite window feature trial={meta}")
                    feats.append(feat.astype(np.float32, copy=False))
                    labels_w.append(int(label))
                    meta_w.append(
                        {
                            "trial_id": trial_id,
                            "subject": subject,
                            "win_index": w_idx,
                            "win_start": w_idx * args.window_step,
                            "win_end": w_idx * args.window_step + args.window_size,
                        }
                    )
            return np.stack(feats, axis=0), np.asarray(labels_w, dtype=np.int64), meta_w

        X_train, y_train, _ = _build_window_features(train_idx)
        X_val, y_val, meta_val = _build_window_features(val_idx)

        if tsm_nan or tsm_inf:
            raise ValueError(f"TSM produced NaN/Inf (nan={tsm_nan} inf={tsm_inf})")

        tsm_audit = {
            "tsm_dim": int(tsm_dim or 0),
            "tsm_norm": _agg_stats(tsm_norms),
            "nan_count": int(tsm_nan),
            "inf_count": int(tsm_inf),
        }
        (outdir / "tsm_audit.json").write_text(json.dumps(tsm_audit, indent=2))

        counts = list(window_counts.values())
        window_audit = {
            "window_size": args.window_size,
            "window_step": args.window_step,
            "count_min": int(np.min(counts)) if counts else 0,
            "count_mean": float(np.mean(counts)) if counts else 0.0,
            "count_max": int(np.max(counts)) if counts else 0,
        }
        (outdir / "window_audit.json").write_text(json.dumps(window_audit, indent=2))

    spd_audit = {
        "train": {
            "min_eig": _agg_stats(spd_stats_train["min_eig"]),
            "min_eig_before": _agg_stats(spd_stats_train["min_eig_before"]),
            "cond": _agg_stats(spd_stats_train["cond"]),
            "symmetry_error": _agg_stats(spd_stats_train["symmetry_error"]),
            "eps_injected_trace_ratio": _agg_stats(spd_stats_train["eps_trace_ratio"]),
        },
        "val": {
            "min_eig": _agg_stats(spd_stats_val["min_eig"]),
            "min_eig_before": _agg_stats(spd_stats_val["min_eig_before"]),
            "cond": _agg_stats(spd_stats_val["cond"]),
            "symmetry_error": _agg_stats(spd_stats_val["symmetry_error"]),
            "eps_injected_trace_ratio": _agg_stats(spd_stats_val["eps_trace_ratio"]),
        },
        "eps_used": _agg_stats(eps_used),
        "eps_base": _agg_stats(eps_base),
    }
    (outdir / "spd_audit.json").write_text(json.dumps(spd_audit, indent=2))

    feature_dim = int(X_train.shape[1])
    cache_features = bool(args.cache_features)
    cache_info = {}
    if cache_features and args.data_mode == "trial_spd":
        np.save(outdir / "features_train.npy", X_train)
        np.save(outdir / "features_val.npy", X_val)
        np.save(outdir / "y_train.npy", y_train)
        np.save(outdir / "y_val.npy", y_val)
        (outdir / "meta_train.json").write_text(json.dumps(meta_train, indent=2))
        (outdir / "meta_val.json").write_text(json.dumps(meta_val, indent=2))
        cache_info = {
            "feature_cache_path": str(outdir),
            "tsm_dim_per_band": int(tsm_audit.get("tsm_dim", 0)) if tsm_audit else 0,
            "bands_used": len(band_indices),
            "input_dim": feature_dim,
        }

    if args.classifier == "mlp":
        if args.data_mode != "trial_spd":
            raise ValueError("MLP classifier is only supported for trial_spd mode")
        if args.s2_mode != "full_spd":
            raise ValueError("MLP classifier is only supported for full_spd mode")

        torch.manual_seed(args.mlp_seed)
        np.random.seed(args.mlp_seed)
        device = torch.device(args.mlp_device)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        scaler_stats = {
            "mean_mean": float(np.mean(scaler.mean_)),
            "mean_std": float(np.std(scaler.mean_)),
            "scale_mean": float(np.mean(scaler.scale_)),
            "scale_std": float(np.std(scaler.scale_)),
        }

        train_ds = TensorDataset(
            torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train).long()
        )
        val_ds = TensorDataset(torch.from_numpy(X_val_s).float(), torch.from_numpy(y_val).long())
        train_loader = DataLoader(train_ds, batch_size=args.mlp_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.mlp_batch_size, shuffle=False)

        model = MLPHead(
            in_dim=feature_dim,
            hidden1=args.mlp_hidden1,
            hidden2=args.mlp_hidden2,
            drop1=args.mlp_dropout1,
            drop2=args.mlp_dropout2,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        best_state = None
        best_epoch = -1
        best_f1 = -1.0
        bad_epochs = 0
        curve = []

        for epoch in range(1, args.mlp_epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_entropy = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * int(yb.size(0))
                preds = torch.argmax(logits, dim=1)
                train_correct += int((preds == yb).sum().item())
                train_total += int(yb.size(0))
                train_entropy.append(_logits_entropy(logits).mean().item())

            model.eval()
            val_correct = 0
            val_total = 0
            val_entropy = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += int((preds == yb).sum().item())
                    val_total += int(yb.size(0))
                    val_entropy.append(_logits_entropy(logits).mean().item())
                    val_preds.extend(preds.cpu().numpy().tolist())
                    val_targets.extend(yb.cpu().numpy().tolist())

            train_acc = train_correct / max(1, train_total)
            val_acc = val_correct / max(1, val_total)
            val_macro = float(f1_score(val_targets, val_preds, average="macro"))
            curve.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / max(1, train_total),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_macro_f1": val_macro,
                    "train_entropy": float(np.mean(train_entropy)) if train_entropy else 0.0,
                    "val_entropy": float(np.mean(val_entropy)) if val_entropy else 0.0,
                }
            )

            if val_macro > best_f1 + 1e-6:
                best_f1 = val_macro
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.mlp_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val_s).float().to(device))
        val_pred = torch.argmax(logits, dim=1).cpu().numpy()
        val_entropy = float(_logits_entropy(logits).mean().item())
        val_acc = float(accuracy_score(y_val, val_pred))
        val_macro = float(f1_score(y_val, val_pred, average="macro"))

        with torch.no_grad():
            train_logits = model(torch.from_numpy(X_train_s).float().to(device))
        train_pred = torch.argmax(train_logits, dim=1).cpu().numpy()
        train_acc = float(accuracy_score(y_train, train_pred))
        train_entropy = float(_logits_entropy(train_logits).mean().item())

        metrics = {
            "val_acc": val_acc,
            "val_macro_f1": val_macro,
            "train_acc": train_acc,
            "train_val_gap": train_acc - val_acc,
            "best_epoch": int(best_epoch),
            "train_entropy": train_entropy,
            "val_entropy": val_entropy,
            "nan_count": 0,
            "inf_count": 0,
        }
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (outdir / "train_curve.json").write_text(json.dumps(curve, indent=2))

        with (outdir / "predictions_val_trial.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "subject", "y_true", "y_pred"])
            for meta, pred, y_true in zip(meta_val, val_pred, y_val):
                writer.writerow([meta["trial_id"], meta["subject"], int(y_true), int(pred)])
    else:
        if args.classifier == "logreg":
            clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs")
        else:
            clf = LinearSVC()

        pipeline = make_pipeline(StandardScaler(), clf)
        pipeline.fit(X_train, y_train)

        if args.data_mode == "trial_spd":
            val_pred = pipeline.predict(X_val)
            metrics = {
                "val_acc": float(accuracy_score(y_val, val_pred)),
                "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
                "nan_count": 0,
                "inf_count": 0,
            }
            (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            with (outdir / "predictions_val_trial.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["trial_id", "subject", "y_true", "y_pred"])
                for meta, pred, y_true in zip(meta_val, val_pred, y_val):
                    writer.writerow([meta["trial_id"], meta["subject"], int(y_true), int(pred)])
        else:
            scores = _scores(pipeline, X_val)
            trial_scores: Dict[Tuple[str, int], List[np.ndarray]] = {}
            trial_labels: Dict[Tuple[str, int], int] = {}
            for row, s, y_true in zip(meta_val, scores, y_val):
                key = (str(row["subject"]), int(row["trial_id"]))
                trial_scores.setdefault(key, []).append(s)
                trial_labels[key] = int(y_true)
            trial_preds = {}
            for key, arrs in trial_scores.items():
                mean_score = np.mean(np.stack(arrs, axis=0), axis=0)
                trial_preds[key] = int(np.argmax(mean_score))
            y_true_trial = []
            y_pred_trial = []
            for key, y_true in trial_labels.items():
                y_true_trial.append(y_true)
                y_pred_trial.append(trial_preds[key])
            metrics = {
                "val_acc": float(accuracy_score(y_true_trial, y_pred_trial)),
                "val_macro_f1": float(f1_score(y_true_trial, y_pred_trial, average="macro")),
                "nan_count": 0,
                "inf_count": 0,
            }
            (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            with (outdir / "predictions_val_window.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["trial_id", "subject", "win_start", "win_end", "y_true", "y_pred", "s0", "s1", "s2"]
                )
                for row, s, y_true, y_pred in zip(meta_val, scores, y_val, pipeline.predict(X_val)):
                    writer.writerow(
                        [
                            row["trial_id"],
                            row["subject"],
                            row["win_start"],
                            row["win_end"],
                            int(y_true),
                            int(y_pred),
                            float(s[0]),
                            float(s[1]),
                            float(s[2]),
                        ]
                    )

            with (outdir / "predictions_val_trial.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["trial_id", "subject", "y_true", "y_pred"])
                for (subject, trial_id), y_true in trial_labels.items():
                    writer.writerow([trial_id, subject, int(y_true), int(trial_preds[(subject, trial_id)])])

    run_config = {
        "root_dir": manifest_root or str(Path(args.manifest_path).parent),
        "output_dir": str(outdir),
        "feature_base": "de_LDS",
        "feature_key": args.feature_key,
        "split_by": args.split_by,
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "classifier": args.classifier,
        "input_dim": feature_dim,
        "tsm_dim_per_band": int(tsm_audit.get("tsm_dim", 0)) if tsm_audit else 0,
        "bands_used": len(band_indices),
        "s2_mode": args.s2_mode,
        "data_mode": args.data_mode,
        "window_size": args.window_size,
        "window_step": args.window_step,
        "sample_unit": "trial" if args.data_mode == "trial_spd" else "window",
        "band_mode": args.band_mode,
        "band_indices": band_indices,
        "max_windows_per_trial": args.max_windows_per_trial,
        "cov_method": args.cov_method,
        "spd_kind": spd_kind,
        "spd_eps": {
            "mode": args.spd_eps_mode,
            "alpha": args.spd_eps_alpha,
            "absolute": args.spd_eps_absolute,
            "floor_mult": args.spd_eps_floor_mult,
            "ceil_mult": args.spd_eps_ceil_mult,
        },
        "tsm_float64": True,
        "eps_diag": args.eps_diag,
        "T_stats": t_stats,
        "feature_cache_path": cache_info.get("feature_cache_path", ""),
        "cache_features": cache_features,
        "subject_counts": {
            "train": len(train_subjects),
            "val": len(val_subjects),
        },
        "subject_lists": {
            "train": train_subjects,
            "val": val_subjects,
        },
        "label_hist": {
            "train": _label_hist(y_train),
            "val": _label_hist(y_val),
        },
        "key_names_used": sorted({t.get("key_name") for t in trials if t.get("key_name")}),
        "reference_source": "sampled_windows" if args.data_mode == "window_spd" else "train_only",
        "reference_train_trials": len(train_idx),
        "ref_spd_paths": (
            [str(outdir / f"ref_spd_band{b}.npy") for b in range(len(band_indices))]
            if args.data_mode == "trial_spd"
            else []
        ),
        "max_ref_windows": args.max_ref_windows,
    }
    if args.classifier == "mlp":
        run_config["mlp_arch"] = {
            "hidden1": args.mlp_hidden1,
            "hidden2": args.mlp_hidden2,
            "dropout1": args.mlp_dropout1,
            "dropout2": args.mlp_dropout2,
        }
        run_config["optimizer"] = {
            "lr": args.mlp_lr,
            "weight_decay": args.mlp_weight_decay,
            "batch_size": args.mlp_batch_size,
            "epochs": args.mlp_epochs,
            "device": args.mlp_device,
        }
        run_config["early_stopping"] = {"patience": args.mlp_patience}
        run_config["scaler_stats"] = scaler_stats
        run_config["train_curve_path"] = str(outdir / "train_curve.json")
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (outdir / "shapes_sample.json").write_text(json.dumps(shapes_sample, indent=2))

    print(
        f"[s2] mode={args.s2_mode} data_mode={args.data_mode} clf={args.classifier} "
        f"val_acc={metrics['val_acc']:.4f} val_macro_f1={metrics['val_macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
