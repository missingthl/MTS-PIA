from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from PIA.telm2 import TELM2Config, TELM2Transformer


def _ordered_unique(values: Iterable[object]) -> List[object]:
    seen = set()
    out: List[object] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _stable_tid_hash(tid: object) -> int:
    h = hashlib.sha256(str(tid).encode("utf-8")).hexdigest()[:16]
    return int(h, 16) & 0x7FFFFFFF


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
    }


def _ensure_2d_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 1:
        return np.vstack([-s, s]).T
    return s


def _true_class_margin(scores: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    s = _ensure_2d_scores(scores)
    y = np.asarray(y_true).astype(int).ravel()
    classes_arr = np.asarray(classes).astype(int).ravel()
    class_to_idx = {int(c): i for i, c in enumerate(classes_arr.tolist())}
    n, c = s.shape
    out = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        yi = int(y[i])
        ci = class_to_idx.get(yi)
        if ci is None:
            continue
        s_true = float(s[i, ci])
        if c <= 1:
            out[i] = s_true
            continue
        s_other = float(np.max(np.delete(s[i], ci)))
        out[i] = s_true - s_other
    return out


def _sample_subset_indices(
    rs: np.random.RandomState,
    n_rows: int,
    *,
    k_dir: int,
    subset_size: int,
) -> np.ndarray:
    s = int(min(max(1, subset_size), k_dir))
    if s == 1:
        return rs.randint(0, k_dir, size=(n_rows, 1), dtype=np.int64)
    if s == 2:
        first = rs.randint(0, k_dir, size=(n_rows,), dtype=np.int64)
        second = rs.randint(0, k_dir - 1, size=(n_rows,), dtype=np.int64)
        second = np.where(second >= first, second + 1, second)
        return np.stack([first, second], axis=1).astype(np.int64)
    sel = np.empty((n_rows, s), dtype=np.int64)
    for i in range(n_rows):
        sel[i] = rs.choice(k_dir, size=s, replace=False)
    return sel


def _safe_quantile(arr: np.ndarray, q: float, default: float) -> float:
    x = np.asarray(arr, dtype=np.float64).ravel()
    if x.size == 0:
        return float(default)
    return float(np.quantile(x, float(q)))


def _minmax_norm(x: np.ndarray, *, constant_fill: float = 0.5) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0:
        return np.asarray([], dtype=np.float64)
    xmin = float(np.min(xx))
    xmax = float(np.max(xx))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) <= 1e-12:
        return np.full(xx.shape, float(constant_fill), dtype=np.float64)
    return (xx - xmin) / (xmax - xmin)


def _build_direction_bank_d1(
    X_train: np.ndarray,
    *,
    k_dir: int,
    seed: int,
    n_iters: int,
    activation: str,
    bias_update_mode: str,
    c_repr: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    cfg = TELM2Config(
        r_dimension=int(k_dir),
        n_iters=int(n_iters),
        activation=activation,
        bias_update_mode=bias_update_mode,
        C_repr=float(c_repr),
        enable_repr_learning=True,
        seed=int(seed),
    )
    telm = TELM2Transformer(cfg).fit(np.asarray(X_train, dtype=np.float64))
    arts = telm.get_artifacts()
    W = np.asarray(arts.W, dtype=np.float64)
    w_mean = np.mean(W, axis=0, keepdims=True)
    Wc = W - w_mean

    Wn = np.zeros_like(Wc, dtype=np.float64)
    for i in range(Wc.shape[0]):
        vec = Wc[i]
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or nrm <= 1e-12:
            vec = W[i]
            nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or nrm <= 1e-12:
            vec = np.zeros_like(vec)
            vec[0] = 1.0
            nrm = 1.0
        Wn[i] = vec / nrm

    row_norms = np.linalg.norm(Wn, axis=1)
    recon = np.asarray(getattr(arts, "recon_err", []), dtype=np.float64)
    meta = {
        "bank_source": "D1_telm2_templates_centered",
        "k_dir": int(k_dir),
        "seed": int(seed),
        "n_iters": int(n_iters),
        "activation": str(activation),
        "bias_update_mode": str(bias_update_mode),
        "c_repr": float(c_repr),
        "direction_norm_stats": _summary_stats(row_norms),
        "recon_last": float(recon[-1]) if recon.size else 0.0,
        "recon_mean": float(np.mean(recon)) if recon.size else 0.0,
        "recon_std": float(np.std(recon)) if recon.size else 0.0,
    }
    return np.asarray(Wn, dtype=np.float32), meta


def _build_multidir_aug_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    subset_size: int,
    gamma: float,
    multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    tid_arr = np.asarray(tid_train)
    y_arr = np.asarray(y_train).astype(int).ravel()
    k_dir = int(direction_bank.shape[0])
    trial_ids = sorted(_ordered_unique(tid_arr.tolist()))

    aug_X_parts: List[np.ndarray] = []
    aug_y_parts: List[np.ndarray] = []
    aug_tid_parts: List[np.ndarray] = []
    aug_src_parts: List[np.ndarray] = []
    aug_dir_parts: List[np.ndarray] = []
    aug_count_per_trial: Dict[str, int] = {}
    dir_pick_count = np.zeros((k_dir,), dtype=np.int64)

    abs_ai_sum = 0.0
    ai_count = 0
    subset_size_sum = 0.0
    subset_obs = 0

    for tid in trial_ids:
        idx = np.where(tid_arr == tid)[0]
        if idx.size == 0:
            aug_count_per_trial[str(tid)] = 0
            continue

        X_tid = np.asarray(X_train[idx], dtype=np.float32)
        y_tid = np.asarray(y_arr[idx], dtype=np.int64)
        added = 0
        for m in range(max(0, int(multiplier))):
            rs = np.random.RandomState(int(seed + m * 1009 + _stable_tid_hash(tid)))
            sel = _sample_subset_indices(rs, int(X_tid.shape[0]), k_dir=k_dir, subset_size=int(subset_size))
            s_eff = int(sel.shape[1])
            a = rs.normal(loc=0.0, scale=1.0, size=(X_tid.shape[0], s_eff)).astype(np.float32)
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

            W_sel = direction_bank[sel]
            w_mix = np.sum(a[:, :, None] * W_sel, axis=1).astype(np.float32)
            X_aug = (X_tid + float(gamma) * w_mix).astype(np.float32)

            aug_X_parts.append(X_aug)
            aug_y_parts.append(y_tid.copy())
            aug_tid_parts.append(np.asarray([tid] * len(idx), dtype=object))
            aug_src_parts.append(X_tid.copy())
            aug_dir_parts.append(sel[:, 0].astype(np.int64).copy())
            added += len(idx)

            binc = np.bincount(sel.reshape(-1), minlength=k_dir)
            dir_pick_count += binc.astype(np.int64)
            abs_ai_sum += float(np.sum(np.abs(a)))
            ai_count += int(a.size)
            subset_size_sum += float(s_eff * X_tid.shape[0])
            subset_obs += int(X_tid.shape[0])

        aug_count_per_trial[str(tid)] = int(added)

    if aug_X_parts:
        X_aug_all = np.vstack(aug_X_parts).astype(np.float32)
        y_aug_all = np.concatenate(aug_y_parts).astype(np.int64)
        tid_aug_all = np.concatenate(aug_tid_parts)
        src_aug_all = np.vstack(aug_src_parts).astype(np.float32)
        dir_aug_all = np.concatenate(aug_dir_parts).astype(np.int64)
    else:
        X_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_aug_all = np.empty((0,), dtype=np.int64)
        tid_aug_all = np.empty((0,), dtype=object)
        src_aug_all = np.empty((0, X_train.shape[1]), dtype=np.float32)
        dir_aug_all = np.empty((0,), dtype=np.int64)

    aug_vals = np.asarray(list(aug_count_per_trial.values()), dtype=np.float64) if aug_count_per_trial else np.asarray([], dtype=np.float64)
    total_picks = int(np.sum(dir_pick_count))
    pick_frac = (
        {str(i): float(c / total_picks) for i, c in enumerate(dir_pick_count.tolist())}
        if total_picks > 0
        else {str(i): 0.0 for i in range(k_dir)}
    )
    meta = {
        "aug_total_count": int(len(y_aug_all)),
        "aug_count_per_trial": aug_count_per_trial,
        "aug_per_trial_mean": float(np.mean(aug_vals)) if aug_vals.size else 0.0,
        "aug_per_trial_std": float(np.std(aug_vals)) if aug_vals.size else 0.0,
        "gamma": float(gamma),
        "k_dir": int(k_dir),
        "subset_size": int(subset_size),
        "mixing_stats": {
            "mean_abs_ai": float(abs_ai_sum / max(1, ai_count)),
            "avg_subset_size": float(subset_size_sum / max(1, subset_obs)),
            "direction_pick_fraction": pick_frac,
        },
    }
    return X_aug_all, y_aug_all, tid_aug_all, src_aug_all, dir_aug_all, meta


def _compute_mech_metrics(
    *,
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_aug_generated: np.ndarray,
    y_aug_generated: np.ndarray,
    X_aug_accepted: np.ndarray,
    y_aug_accepted: np.ndarray,
    X_src_accepted: np.ndarray,
    dir_generated: np.ndarray,
    dir_accepted: np.ndarray,
    seed: int,
    linear_c: float,
    class_weight: Optional[str],
    linear_max_iter: int,
    knn_k: int,
    max_aug_for_mech: int,
    max_real_knn_ref: int,
    max_real_knn_query: int,
    progress_prefix: Optional[str] = None,
) -> Dict[str, object]:
    Xr = np.asarray(X_train_real, dtype=np.float32)
    yr = np.asarray(y_train_real).astype(int).ravel()
    Xg = np.asarray(X_aug_generated, dtype=np.float32)
    yg = np.asarray(y_aug_generated).astype(int).ravel()
    Xa = np.asarray(X_aug_accepted, dtype=np.float32)
    ya = np.asarray(y_aug_accepted).astype(int).ravel()
    Xs = np.asarray(X_src_accepted, dtype=np.float32)
    dir_g = np.asarray(dir_generated).astype(int).ravel()
    dir_a = np.asarray(dir_accepted).astype(int).ravel()

    if not (len(Xg) == len(yg) == len(dir_g)):
        raise ValueError("Mechanism metric input mismatch on generated arrays.")
    if not (len(Xa) == len(ya) == len(Xs) == len(dir_a)):
        raise ValueError("Mechanism metric input mismatch on accepted arrays.")

    cw = None if class_weight in {None, "", "none"} else class_weight
    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    clf_ref = LinearSVC(
        C=float(linear_c),
        class_weight=cw,
        max_iter=int(linear_max_iter),
        random_state=int(seed),
        dual="auto",
    )
    clf_ref.fit(Xr_s, yr)

    n_acc = int(len(ya))
    rng_seed = int(seed + 8093)
    rs = np.random.RandomState(rng_seed)
    if n_acc <= 0:
        eval_idx = np.asarray([], dtype=np.int64)
    elif n_acc > int(max_aug_for_mech):
        eval_idx = np.sort(rs.choice(n_acc, size=int(max_aug_for_mech), replace=False))
    else:
        eval_idx = np.arange(n_acc, dtype=np.int64)

    Xa_eval = Xa[eval_idx] if eval_idx.size else np.empty((0, Xr.shape[1]), dtype=np.float32)
    Xs_eval = Xs[eval_idx] if eval_idx.size else np.empty((0, Xr.shape[1]), dtype=np.float32)
    ya_eval = ya[eval_idx] if eval_idx.size else np.empty((0,), dtype=np.int64)
    dir_eval = dir_a[eval_idx] if eval_idx.size else np.empty((0,), dtype=np.int64)

    if eval_idx.size:
        src_scores = clf_ref.decision_function(scaler.transform(Xs_eval))
        aug_scores = clf_ref.decision_function(scaler.transform(Xa_eval))
        src_margin = _true_class_margin(src_scores, ya_eval, clf_ref.classes_)
        aug_margin = _true_class_margin(aug_scores, ya_eval, clf_ref.classes_)
        flip = (src_margin >= 0.0) != (aug_margin >= 0.0)
        margin_delta = aug_margin - src_margin
        flip_rate = float(np.mean(flip))
        margin_drop_median = float(np.median(margin_delta))
    else:
        flip = np.asarray([], dtype=bool)
        margin_delta = np.asarray([], dtype=np.float64)
        flip_rate = 0.0
        margin_drop_median = 0.0

    n_real = int(len(yr))
    if int(max_real_knn_ref) > 0 and n_real > int(max_real_knn_ref):
        ref_idx = np.sort(rs.choice(n_real, size=int(max_real_knn_ref), replace=False))
    else:
        ref_idx = np.arange(n_real, dtype=np.int64)
    Xr_knn = Xr[ref_idx]
    yr_knn = yr[ref_idx]

    k_eff = int(min(max(1, int(knn_k)), max(1, len(yr_knn))))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xr_knn)

    if eval_idx.size and len(yr_knn) > 0:
        nn_idx = nn.kneighbors(Xa_eval, return_distance=False)
        y_nb = yr_knn[nn_idx]
        purity_each = np.mean(y_nb == ya_eval[:, None], axis=1).astype(np.float64)
        intrusion_rate = float(np.mean(1.0 - purity_each))
    else:
        intrusion_rate = 0.0
        purity_each = np.asarray([], dtype=np.float64)
    purity = float(1.0 - intrusion_rate)

    if len(yr_knn) <= 1:
        real_intrusion_rate = 0.0
        real_query_idx_local = np.asarray([], dtype=np.int64)
    else:
        n_ref = int(len(yr_knn))
        if int(max_real_knn_query) > 0 and n_ref > int(max_real_knn_query):
            real_query_idx_local = np.sort(rs.choice(n_ref, size=int(max_real_knn_query), replace=False))
        else:
            real_query_idx_local = np.arange(n_ref, dtype=np.int64)
        X_real_q = Xr_knn[real_query_idx_local]
        y_real_q = yr_knn[real_query_idx_local]
        k_real = int(min(max(1, int(knn_k) + 1), len(yr_knn)))
        nn_real = NearestNeighbors(n_neighbors=k_real, metric="euclidean")
        nn_real.fit(Xr_knn)
        nn_idx_real = nn_real.kneighbors(X_real_q, return_distance=False)
        if k_real > 1:
            nn_idx_real = nn_idx_real[:, 1:]
        y_nb_real = yr_knn[nn_idx_real]
        purity_real_each = np.mean(y_nb_real == y_real_q[:, None], axis=1).astype(np.float64)
        real_intrusion_rate = float(np.mean(1.0 - purity_real_each))

    dir_profile: Dict[int, Dict[str, float]] = {}
    for did in sorted(set(dir_g.tolist()) | set(dir_a.tolist())):
        gen_mask = dir_g == int(did)
        acc_mask = dir_a == int(did)
        eval_mask = dir_eval == int(did)
        if np.any(eval_mask):
            dir_profile[int(did)] = {
                "usage": float(np.mean(acc_mask)) if len(dir_a) else 0.0,
                "flip_rate": float(np.mean(flip[eval_mask])) if flip.size else 0.0,
                "margin_drop_median": float(np.median(margin_delta[eval_mask])) if margin_delta.size else 0.0,
                "intrusion": float(np.mean(1.0 - purity_each[eval_mask])) if purity_each.size else 0.0,
                "n_generated": int(np.sum(gen_mask)),
                "n_accepted": int(np.sum(acc_mask)),
            }
        else:
            dir_profile[int(did)] = {
                "usage": float(np.mean(acc_mask)) if len(dir_a) else 0.0,
                "flip_rate": 0.0,
                "margin_drop_median": 0.0,
                "intrusion": 0.0,
                "n_generated": int(np.sum(gen_mask)),
                "n_accepted": int(np.sum(acc_mask)),
            }

    best_dir_id = max(dir_profile, key=lambda k: dir_profile[k]["margin_drop_median"], default=-1)
    worst_dir_id = min(dir_profile, key=lambda k: dir_profile[k]["margin_drop_median"], default=-1)
    return {
        "flip_rate": float(flip_rate),
        "margin_drop_median": float(margin_drop_median),
        "intrusion_rate": float(intrusion_rate),
        "purity": float(purity),
        "real_intrusion_rate": float(real_intrusion_rate),
        "dir_profile": {str(int(k)): v for k, v in dir_profile.items()},
        "dir_profile_summary": f"best={best_dir_id} worst={worst_dir_id}",
        "best_dir_id": int(best_dir_id),
        "worst_dir_id": int(worst_dir_id),
        "n_aug_generated": int(len(yg)),
        "n_aug_accepted": int(len(ya)),
        "n_aug_used_for_mech": int(len(eval_idx)),
        "mech_rng_seed": int(rng_seed),
        "flip_margin_definition": "true_class_margin=(score_y-max_other); flip=sign_change(margin)",
        "knn_space": "z_raw",
    }


def _active_direction_probs(gamma_by_dir: np.ndarray, *, freeze_eps: float) -> np.ndarray:
    g = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    active = g > float(freeze_eps)
    if not np.any(active):
        return np.full(g.shape, 1.0 / float(max(1, len(g))), dtype=np.float64)
    probs = np.zeros_like(g, dtype=np.float64)
    probs[active] = 1.0 / float(np.sum(active))
    return probs


def _compute_direction_intrusion(
    *,
    X_anchor: np.ndarray,
    y_anchor: np.ndarray,
    X_aug_accepted: np.ndarray,
    y_aug_accepted: np.ndarray,
    dir_accepted: np.ndarray,
    seed: int,
    knn_k: int,
    max_eval: int,
) -> Dict[int, float]:
    Xa = np.asarray(X_aug_accepted, dtype=np.float32)
    ya = np.asarray(y_aug_accepted).astype(int).ravel()
    da = np.asarray(dir_accepted).astype(int).ravel()
    Xr = np.asarray(X_anchor, dtype=np.float32)
    yr = np.asarray(y_anchor).astype(int).ravel()
    if Xa.shape[0] == 0 or Xr.shape[0] == 0:
        return {}
    rs = np.random.RandomState(int(seed + 9917))
    if Xa.shape[0] > int(max_eval):
        idx = np.sort(rs.choice(Xa.shape[0], size=int(max_eval), replace=False))
        Xa = Xa[idx]
        ya = ya[idx]
        da = da[idx]
    k_eff = int(min(max(1, int(knn_k)), Xr.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xr)
    nn_idx = nn.kneighbors(Xa, return_distance=False)
    y_nb = yr[nn_idx]
    intrusion_by_dir: Dict[int, List[float]] = {}
    for i, did in enumerate(da.tolist()):
        purity = float(np.mean(y_nb[i] == ya[i]))
        intrusion_by_dir.setdefault(int(did), []).append(1.0 - purity)
    return {int(k): float(np.mean(v)) if v else 0.0 for k, v in intrusion_by_dir.items()}


def _mech_dir_maps(mech: Dict[str, object], *, intrusion_by_dir: Dict[int, float] | None = None) -> Dict[str, Dict[int, float]]:
    profile = mech.get("dir_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    usage: Dict[int, float] = {}
    flip: Dict[int, float] = {}
    margin: Dict[int, float] = {}
    intrusion: Dict[int, float] = {}
    for k, row in profile.items():
        if not isinstance(row, dict):
            continue
        did = int(k)
        usage[did] = float(row.get("usage", 0.0))
        flip[did] = float(row.get("flip_rate", 0.0))
        margin[did] = float(row.get("margin_drop_median", 0.0))
        if intrusion_by_dir and did in intrusion_by_dir:
            intrusion[did] = float(intrusion_by_dir[did])
    return {
        "usage": usage,
        "flip_rate": flip,
        "margin_drop_median": margin,
        "intrusion": intrusion,
    }


def _update_direction_budget(
    *,
    gamma_before: np.ndarray,
    margin_by_dir: Dict[int, float],
    flip_by_dir: Dict[int, float],
    intrusion_by_dir: Dict[int, float],
    expand_factor: float,
    shrink_factor: float,
    gamma_max: float,
    freeze_eps: float,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, float]]:
    k_dir = int(len(gamma_before))
    dir_ids = list(range(k_dir))
    margin = np.asarray([float(margin_by_dir.get(i, 0.0)) for i in dir_ids], dtype=np.float64)
    flip = np.asarray([float(flip_by_dir.get(i, 0.0)) for i in dir_ids], dtype=np.float64)
    intrusion = np.asarray([float(intrusion_by_dir.get(i, 0.0)) for i in dir_ids], dtype=np.float64)

    if np.allclose(margin, margin[0]) and np.allclose(flip, flip[0]) and np.allclose(intrusion, intrusion[0]):
        return gamma_before.copy(), {i: "hold" for i in dir_ids}, {i: 0.5 for i in dir_ids}

    margin_good = _minmax_norm(margin, constant_fill=0.5)
    flip_good = 1.0 - _minmax_norm(flip, constant_fill=0.5)
    intr_good = 1.0 - _minmax_norm(intrusion, constant_fill=0.5)
    safety = (margin_good + flip_good + intr_good) / 3.0

    q_expand = _safe_quantile(safety, 0.75, 0.5)
    q_shrink = _safe_quantile(safety, 0.35, 0.5)
    q_freeze = _safe_quantile(safety, 0.15, 0.25)
    med_margin = _safe_quantile(margin, 0.50, 0.0)
    med_flip = _safe_quantile(flip, 0.50, 0.0)
    med_intr = _safe_quantile(intrusion, 0.50, 0.0)

    gamma_after = np.asarray(gamma_before, dtype=np.float64).copy()
    state_by_dir: Dict[int, str] = {}
    score_by_dir: Dict[int, float] = {}
    for i in dir_ids:
        score = float(safety[i])
        score_by_dir[i] = score
        g0 = float(gamma_before[i])
        if g0 <= float(freeze_eps):
            gamma_after[i] = 0.0
            state_by_dir[i] = "freeze"
            continue
        risky = bool(margin[i] < min(0.0, med_margin) and (flip[i] >= med_flip or intrusion[i] >= med_intr))
        if score <= q_freeze and risky:
            gamma_after[i] = 0.0
            state_by_dir[i] = "freeze"
        elif score >= q_expand and margin[i] >= med_margin and flip[i] <= med_flip and intrusion[i] <= med_intr:
            gamma_after[i] = min(float(gamma_max), g0 * float(expand_factor))
            state_by_dir[i] = "expand"
        elif score <= q_shrink or risky:
            g1 = g0 * float(shrink_factor)
            if g1 <= float(freeze_eps):
                gamma_after[i] = 0.0
                state_by_dir[i] = "freeze"
            else:
                gamma_after[i] = g1
                state_by_dir[i] = "shrink"
        else:
            gamma_after[i] = g0
            state_by_dir[i] = "hold"
    return gamma_after.astype(np.float64), state_by_dir, score_by_dir
