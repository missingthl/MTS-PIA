from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA

from core.bridge import bridge_single, logvec_to_spd


@dataclass
class ExternalAugResult:
    X_aug: np.ndarray
    y_aug: Optional[np.ndarray] = None
    y_aug_soft: Optional[np.ndarray] = None
    source_space: str = "raw_time"
    label_mode: str = "hard"
    uses_external_library: bool = False
    library_name: str = ""
    budget_matched: bool = True
    selection_rule: str = ""
    warning_count: int = 0
    fallback_count: int = 0
    meta: Dict[str, float] = field(default_factory=dict)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _repeat_anchor_indices(n_train: int, multiplier: int) -> np.ndarray:
    return np.repeat(np.arange(int(n_train), dtype=np.int64), int(multiplier))


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((int(y.shape[0]), int(n_classes)), dtype=np.float32)
    out[np.arange(int(y.shape[0])), y.astype(np.int64)] = 1.0
    return out


def _resample_ct(x_ct: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample one multivariate series from [C, T] to [C, target_len]."""
    x_ct = np.asarray(x_ct, dtype=np.float32)
    c, t = int(x_ct.shape[0]), int(x_ct.shape[1])
    target_len = int(target_len)
    if t == target_len:
        return x_ct.astype(np.float32, copy=True)
    if t <= 1:
        return np.repeat(x_ct, target_len, axis=1).astype(np.float32)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, target_len)
    out = np.empty((c, target_len), dtype=np.float32)
    for ch in range(c):
        out[ch] = np.interp(dst, src, x_ct[ch]).astype(np.float32)
    return out


def _class_to_indices(y_train: np.ndarray) -> Dict[int, np.ndarray]:
    y_train = np.asarray(y_train, dtype=np.int64)
    return {int(c): np.flatnonzero(y_train == c) for c in np.unique(y_train)}


def _finite_stack(xs: List[np.ndarray]) -> np.ndarray:
    return np.nan_to_num(np.stack(xs, axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def raw_aug_jitter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    scale: float = 0.05,
) -> ExternalAugResult:
    try:
        from tsaug import AddNoise
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_aug_jitter requires optional dependency `tsaug`.") from exc

    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    np.random.seed(int(seed) % (2**32 - 1))
    X_tc = np.transpose(X_src, (0, 2, 1))
    X_aug = AddNoise(scale=float(scale)).augment(X_tc)
    X_aug = np.transpose(np.asarray(X_aug, dtype=np.float32), (0, 2, 1))
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=True,
        library_name="tsaug",
        budget_matched=True,
        selection_rule="repeat_train_anchors_addnoise",
    )


def raw_aug_scaling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    low: float = 0.8,
    high: float = 1.2,
) -> ExternalAugResult:
    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    factors = rng.uniform(float(low), float(high), size=(len(idx), 1, 1)).astype(np.float32)
    return ExternalAugResult(
        X_aug=X_src * factors,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_amplitude_uniform",
        meta={"scaling_low": float(low), "scaling_high": float(high)},
    )


def raw_aug_timewarp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    n_speed_change: int = 3,
    max_speed_ratio: float = 2.0,
) -> ExternalAugResult:
    try:
        from tsaug import TimeWarp
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_aug_timewarp requires optional dependency `tsaug`.") from exc

    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    np.random.seed(int(seed) % (2**32 - 1))
    X_tc = np.transpose(X_src, (0, 2, 1))
    X_aug = TimeWarp(
        n_speed_change=int(n_speed_change),
        max_speed_ratio=float(max_speed_ratio),
    ).augment(X_tc)
    X_aug = np.transpose(np.asarray(X_aug, dtype=np.float32), (0, 2, 1))
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=True,
        library_name="tsaug",
        budget_matched=True,
        selection_rule="repeat_train_anchors_timewarp",
    )


def raw_aug_magnitude_warping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    sigma: float = 0.2,
    knots: int = 4,
    per_channel_curve: bool = True,
) -> ExternalAugResult:
    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    n_aug, c, t = X_src.shape
    n_knots = max(1, int(knots))
    x_knots = np.linspace(0.0, float(max(t - 1, 1)), n_knots + 2)
    x_full = np.arange(t, dtype=np.float64)
    X_out = np.empty_like(X_src, dtype=np.float32)

    for i in range(n_aug):
        n_curves = c if per_channel_curve else 1
        knot_vals = rng.normal(1.0, float(sigma), size=(n_curves, n_knots + 2))
        knot_vals = np.clip(knot_vals, 0.05, None)
        curves = []
        for curve_idx in range(n_curves):
            curve = CubicSpline(x_knots, knot_vals[curve_idx], bc_type="natural")(x_full)
            curves.append(np.clip(curve, 0.05, None).astype(np.float32))
        curve_arr = np.stack(curves, axis=0)
        if not per_channel_curve:
            curve_arr = np.repeat(curve_arr, c, axis=0)
        X_out[i] = X_src[i] * curve_arr

    return ExternalAugResult(
        X_aug=np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_magnitude_warping",
        meta={
            "warp_sigma": float(sigma),
            "warp_knots": float(knots),
            "per_channel_curve": float(bool(per_channel_curve)),
        },
    )


def raw_aug_window_warping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    window_ratio: float = 0.10,
    speed_factors: Tuple[float, ...] = (0.5, 2.0),
    min_window_len: int = 4,
) -> ExternalAugResult:
    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    t = int(X_src.shape[-1])
    X_out: List[np.ndarray] = []
    fallback_count = 0
    for x in X_src:
        if t < 3:
            fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        win_len = int(round(float(window_ratio) * t))
        win_len = max(int(min_window_len), win_len)
        win_len = min(max(1, win_len), max(1, t - 1))
        if win_len >= t:
            fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        start = int(rng.integers(0, t - win_len + 1))
        speed = float(rng.choice(np.asarray(speed_factors, dtype=np.float64)))
        warped_len = max(1, int(round(win_len * speed)))
        before = x[:, :start]
        segment = x[:, start:start + win_len]
        after = x[:, start + win_len:]
        warped = _resample_ct(segment, warped_len)
        stitched = np.concatenate([before, warped, after], axis=1)
        X_out.append(_resample_ct(stitched, t))

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_window_warping",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "window_warp_ratio": float(window_ratio),
            "window_warp_min_window_len": float(min_window_len),
            "window_warp_speed_min": float(np.min(speed_factors)),
            "window_warp_speed_max": float(np.max(speed_factors)),
        },
    )


def raw_aug_window_slicing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    slice_ratio: float = 0.90,
    min_window_len: int = 4,
) -> ExternalAugResult:
    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    t = int(X_src.shape[-1])
    X_out: List[np.ndarray] = []
    fallback_count = 0
    for x in X_src:
        slice_len = int(round(float(slice_ratio) * t))
        slice_len = max(int(min_window_len), slice_len)
        slice_len = min(max(1, slice_len), t)
        if t <= 1 or slice_len >= t:
            if t <= 1:
                fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        start = int(rng.integers(0, t - slice_len + 1))
        X_out.append(_resample_ct(x[:, start:start + slice_len], t))

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_window_slicing",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "window_slice_ratio": float(slice_ratio),
            "window_slice_min_window_len": float(min_window_len),
        },
    )


def raw_mixup(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    alpha: float = 0.4,
    n_classes: Optional[int] = None,
) -> ExternalAugResult:
    rng = _rng(seed)
    n_train = int(len(X_train))
    n_aug = int(multiplier) * n_train
    n_classes_i = int(n_classes if n_classes is not None else np.max(y_train) + 1)
    i = rng.integers(0, n_train, size=n_aug)
    j = rng.integers(0, n_train, size=n_aug)
    lam = rng.beta(float(alpha), float(alpha), size=(n_aug, 1, 1)).astype(np.float32)
    X_aug = lam * np.asarray(X_train[i], dtype=np.float32) + (1.0 - lam) * np.asarray(X_train[j], dtype=np.float32)

    lam_y = lam.reshape(n_aug, 1)
    y_i = _one_hot(np.asarray(y_train[i], dtype=np.int64), n_classes_i)
    y_j = _one_hot(np.asarray(y_train[j], dtype=np.int64), n_classes_i)
    y_soft = lam_y * y_i + (1.0 - lam_y) * y_j
    return ExternalAugResult(
        X_aug=X_aug.astype(np.float32),
        y_aug_soft=y_soft.astype(np.float32),
        source_space="raw_mixup",
        label_mode="soft",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="train_split_random_pair_beta",
        meta={"mixup_alpha": float(alpha)},
    )


def dba_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    k: int = 5,
    max_iter: int = 5,
) -> ExternalAugResult:
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("dba_sameclass requires optional dependency `tslearn`.") from exc

    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = {int(c): np.flatnonzero(y_train == c) for c in np.unique(y_train)}
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    warning_count = 0

    for anchor in anchor_idx:
        cls = int(y_train[int(anchor)])
        pool = class_to_idx[cls]
        replace = len(pool) < int(k)
        if replace:
            warning_count += 1
        chosen = rng.choice(pool, size=int(k), replace=replace)
        group_tc = np.transpose(np.asarray(X_train[chosen], dtype=np.float64), (0, 2, 1))
        bary_tc = dtw_barycenter_averaging(group_tc, max_iter=int(max_iter))
        X_out.append(np.transpose(np.asarray(bary_tc, dtype=np.float32), (1, 0)))
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=np.stack(X_out, axis=0).astype(np.float32),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_barycenter",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="same_class_dba",
        warning_count=int(warning_count),
        meta={"dba_k": float(k), "dba_max_iter": float(max_iter)},
    )


def wdba_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    k: int = 5,
    max_iter: int = 5,
    tau: Optional[float] = None,
) -> ExternalAugResult:
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
        from tslearn.metrics import dtw
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("wdba_sameclass requires optional dependency `tslearn`.") from exc

    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = _class_to_indices(y_train)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0
    tau_values: List[float] = []
    k_eff = int(k)

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        pool = class_to_idx[cls]
        replace = len(pool) < k_eff
        if replace:
            fallback_count += 1
        chosen = rng.choice(pool, size=k_eff, replace=replace)
        if anchor_i not in chosen:
            chosen[0] = anchor_i
        group_tc = np.transpose(np.asarray(X_train[chosen], dtype=np.float64), (0, 2, 1))
        anchor_tc = np.asarray(X_train[anchor_i], dtype=np.float64).T
        dists = np.asarray([float(dtw(anchor_tc, group_tc[j])) for j in range(k_eff)], dtype=np.float64)
        if tau is None:
            positive = dists[dists > 1e-12]
            tau_i = float(np.median(positive)) if positive.size else 1.0
            if not np.isfinite(tau_i) or tau_i <= 1e-12:
                tau_i = 1.0
                fallback_count += 1
        else:
            tau_i = float(tau)
        tau_values.append(tau_i)
        logits = -dists / max(tau_i, 1e-12)
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        weights /= float(np.sum(weights) + 1e-12)
        try:
            bary_tc = dtw_barycenter_averaging(group_tc, weights=weights, max_iter=int(max_iter))
        except Exception:
            fallback_count += 1
            bary_tc = np.average(group_tc, axis=0, weights=weights)
        X_out.append(np.transpose(np.asarray(bary_tc, dtype=np.float32), (1, 0)))
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_barycenter",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="same_class_weighted_dba_anchor_dtw_softmax",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "wdba_k": float(k),
            "wdba_max_iter": float(max_iter),
            "wdba_tau": float(np.mean(tau_values)) if tau_values else float("nan"),
        },
    )


def spawner_sameclass_style(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    noise_scale: float = 0.05,
) -> ExternalAugResult:
    try:
        from tslearn.metrics import dtw_path
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("spawner_sameclass_style requires optional dependency `tslearn`.") from exc

    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = _class_to_indices(y_train)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        x = np.asarray(X_train[anchor_i], dtype=np.float32)
        pool = class_to_idx[cls]
        candidates = pool[pool != anchor_i]
        if len(candidates) == 0:
            mate_i = anchor_i
            fallback_count += 1
        else:
            mate_i = int(rng.choice(candidates))
        mate = np.asarray(X_train[mate_i], dtype=np.float32)
        try:
            path, _ = dtw_path(x.T.astype(np.float64), mate.T.astype(np.float64))
            aligned = np.empty_like(x, dtype=np.float32)
            buckets: List[List[int]] = [[] for _ in range(x.shape[1])]
            for i_t, j_t in path:
                if 0 <= int(i_t) < x.shape[1] and 0 <= int(j_t) < mate.shape[1]:
                    buckets[int(i_t)].append(int(j_t))
            for i_t, js in enumerate(buckets):
                if js:
                    aligned[:, i_t] = np.mean(mate[:, js], axis=1)
                else:
                    aligned[:, i_t] = mate[:, min(i_t, mate.shape[1] - 1)]
        except Exception:
            aligned = mate
            fallback_count += 1
        mixed = 0.5 * x + 0.5 * aligned
        ch_std = np.std(x, axis=1, keepdims=True).astype(np.float32)
        noise = rng.normal(0.0, float(noise_scale), size=x.shape).astype(np.float32) * (ch_std + 1e-6)
        X_out.append(mixed + noise)
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_pattern_mix",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="spawner_style_same_class_dtw_aligned_average",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={"spawner_noise_scale": float(noise_scale)},
    )


def raw_smote_flatten_balanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> ExternalAugResult:
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_smote_flatten_balanced requires optional dependency `imbalanced-learn`.") from exc

    y_train = np.asarray(y_train, dtype=np.int64)
    _, counts = np.unique(y_train, return_counts=True)
    if counts.size == 0 or int(counts.min()) < 2:
        empty = np.empty((0, X_train.shape[1], X_train.shape[2]), dtype=np.float32)
        return ExternalAugResult(
            X_aug=empty,
            y_aug=np.empty((0,), dtype=np.int64),
            source_space="flattened_raw",
            label_mode="hard",
            uses_external_library=True,
            library_name="imbalanced-learn",
            budget_matched=False,
            selection_rule="class_balancing_smote_auto",
            warning_count=1,
        )

    k_neighbors = max(1, min(5, int(counts.min()) - 1))
    flat = np.asarray(X_train, dtype=np.float32).reshape(len(X_train), -1)
    smote = SMOTE(sampling_strategy="auto", random_state=int(seed), k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(flat, y_train)
    X_new = np.asarray(X_res[len(X_train):], dtype=np.float32).reshape(-1, X_train.shape[1], X_train.shape[2])
    y_new = np.asarray(y_res[len(y_train):], dtype=np.int64)
    return ExternalAugResult(
        X_aug=X_new,
        y_aug=y_new,
        source_space="flattened_raw",
        label_mode="hard",
        uses_external_library=True,
        library_name="imbalanced-learn",
        budget_matched=False,
        selection_rule="class_balancing_smote_auto",
        meta={"smote_k_neighbors": float(k_neighbors)},
    )


def _build_covariance_records(X_train: np.ndarray, spd_eps: float = 1e-4) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    records: List[Dict[str, np.ndarray]] = []
    log_covs = []
    for x_np in np.asarray(X_train, dtype=np.float32):
        x = torch.from_numpy(x_np).double()
        x = x - x.mean(dim=-1, keepdim=True)
        cov = (x @ x.transpose(-1, -2)) / float(max(1, x.shape[-1] - 1))
        cov = cov + float(spd_eps) * torch.eye(cov.shape[0], dtype=cov.dtype)
        vals, vecs = torch.linalg.eigh(cov)
        log_cov = vecs @ torch.diag_embed(torch.log(torch.clamp(vals, min=spd_eps))) @ vecs.transpose(-1, -2)
        log_covs.append(log_cov.numpy())
        records.append({"x_raw": x_np, "sigma_orig": cov.numpy(), "log_cov": log_cov.numpy()})

    mean_log = np.mean(log_covs, axis=0)
    idx = np.triu_indices(mean_log.shape[0])
    for record in records:
        record["z"] = (record["log_cov"] - mean_log)[idx].astype(np.float32)
    return records, mean_log.astype(np.float64)


def _materialize_cov_state_aug(
    records: List[Dict[str, np.ndarray]],
    mean_log: np.ndarray,
    z_cands: np.ndarray,
    anchor_idx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    X_aug = []
    transport_errors = []
    for z, idx in zip(z_cands, anchor_idx):
        rec = records[int(idx)]
        sigma_aug = logvec_to_spd(np.asarray(z, dtype=np.float32), mean_log)
        x_aug, meta = bridge_single(
            torch.from_numpy(rec["x_raw"]),
            torch.from_numpy(rec["sigma_orig"]),
            torch.from_numpy(sigma_aug),
        )
        X_aug.append(x_aug.cpu().numpy().astype(np.float32))
        transport_errors.append(float(meta.get("transport_error_logeuc", np.nan)))
    mean_err = float(np.nanmean(transport_errors)) if transport_errors else float("nan")
    return np.stack(X_aug, axis=0).astype(np.float32), mean_err


def random_cov_state(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    gamma: float,
) -> ExternalAugResult:
    rng = _rng(seed)
    records, mean_log = _build_covariance_records(X_train)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    z_dim = int(records[0]["z"].shape[0])
    dirs = rng.normal(size=(len(anchor_idx), z_dim)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    signs = np.where(np.arange(len(anchor_idx)) % 2 == 0, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    z0 = np.stack([records[int(i)]["z"] for i in anchor_idx], axis=0)
    z_cands = z0 + signs * float(gamma) * dirs
    X_aug, transport_err = _materialize_cov_state_aug(records, mean_log, z_cands, anchor_idx)
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[anchor_idx], dtype=np.int64),
        source_space="covariance_state",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="random_unit_z_direction",
        meta={"transport_error_logeuc_mean": transport_err},
    )


def pca_cov_state(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    gamma: float,
    k_dir: int,
) -> ExternalAugResult:
    records, mean_log = _build_covariance_records(X_train)
    Z = np.stack([rec["z"] for rec in records], axis=0)
    n_components = max(1, min(int(k_dir), int(Z.shape[0]), int(Z.shape[1])))
    pca = PCA(n_components=n_components, random_state=int(seed))
    pca.fit(Z)
    components = np.asarray(pca.components_, dtype=np.float32)
    components /= np.linalg.norm(components, axis=1, keepdims=True) + 1e-12
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    slots = np.arange(len(anchor_idx), dtype=np.int64)
    dirs = components[slots % n_components]
    signs = np.where((slots // n_components) % 2 == 0, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    z0 = np.stack([records[int(i)]["z"] for i in anchor_idx], axis=0)
    z_cands = z0 + signs * float(gamma) * dirs
    X_aug, transport_err = _materialize_cov_state_aug(records, mean_log, z_cands, anchor_idx)
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[anchor_idx], dtype=np.int64),
        source_space="covariance_state",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="pca_top_z_direction",
        meta={
            "pca_n_components": float(n_components),
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "transport_error_logeuc_mean": transport_err,
        },
    )
