from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _summary_stats(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64).ravel()
    if x.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }


def _scatter_from_center(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    cc = np.asarray(center, dtype=np.float64).ravel()
    d = int(cc.size)
    if xx.size == 0:
        return np.zeros((d, d), dtype=np.float64)
    diff = xx - cc[None, :]
    return (diff.T @ diff) / max(1, int(xx.shape[0]))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float | None, float | None]:
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return None, None
    if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
        return None, None
    pearson = float(np.corrcoef(xx, yy)[0, 1])
    x_rank = pd.Series(xx).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(yy).rank(method="average").to_numpy(dtype=np.float64)
    if np.allclose(x_rank, x_rank[0]) or np.allclose(y_rank, y_rank[0]):
        spearman = None
    else:
        spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson, spearman


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0:
        return np.asarray([], dtype=np.float64)
    xmin = float(np.min(xx))
    xmax = float(np.max(xx))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) <= 1e-12:
        return np.zeros_like(xx, dtype=np.float64)
    return (xx - xmin) / (xmax - xmin)


@dataclass
class FisherPIAConfig:
    knn_k: int = 20
    interior_quantile: float = 0.7
    boundary_quantile: float = 0.3
    hetero_k: int = 3
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    rho: float = 1e-6


def compute_fisher_pia_terms(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    cfg: FisherPIAConfig,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, object]]:
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train).astype(int).ravel()
    classes = sorted(np.unique(y).tolist())
    if X.ndim != 2:
        raise ValueError("X_train must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_train / y_train size mismatch.")

    d = int(X.shape[1])
    k_eff = int(min(max(1, int(cfg.knn_k)), max(1, X.shape[0] - 1)))
    n_query = int(min(X.shape[0], k_eff + 1))
    nn = NearestNeighbors(n_neighbors=n_query, metric="euclidean")
    nn.fit(X)
    nn_idx = nn.kneighbors(X, return_distance=False)
    if n_query >= 2:
        nn_idx_use = nn_idx[:, 1:]
    else:
        nn_idx_use = np.empty((X.shape[0], 0), dtype=np.int64)
    y_nb = y[nn_idx_use] if nn_idx_use.size else np.empty((X.shape[0], 0), dtype=np.int64)
    purity = (
        np.mean(y_nb == y[:, None], axis=1).astype(np.float64)
        if y_nb.size
        else np.ones((X.shape[0],), dtype=np.float64)
    )

    mu_by_class: Dict[int, np.ndarray] = {}
    class_counts: Dict[int, int] = {}
    for cls in classes:
        mask = y == cls
        mu_by_class[int(cls)] = np.mean(X[mask], axis=0).astype(np.float64)
        class_counts[int(cls)] = int(np.sum(mask))

    terms: Dict[int, Dict[str, object]] = {}
    total = float(max(1, X.shape[0]))
    for cls in classes:
        cls_i = int(cls)
        idx = np.where(y == cls_i)[0]
        Xy = X[idx]
        mu_y = mu_by_class[cls_i]
        purity_y = purity[idx]

        q_boundary = float(np.quantile(purity_y, float(cfg.boundary_quantile))) if purity_y.size else 0.0
        q_interior = float(np.quantile(purity_y, float(cfg.interior_quantile))) if purity_y.size else 1.0
        boundary_local = purity_y <= q_boundary
        interior_local = purity_y >= q_interior
        if purity_y.size and not np.any(boundary_local):
            boundary_local[np.argmin(purity_y)] = True
        if purity_y.size and not np.any(interior_local):
            interior_local[np.argmax(purity_y)] = True

        X_boundary = Xy[boundary_local]
        X_interior = Xy[interior_local]

        S_W = _scatter_from_center(Xy, mu_y)
        S_expand = _scatter_from_center(X_interior, mu_y) if X_interior.size else S_W.copy()

        S_B = np.zeros((d, d), dtype=np.float64)
        for other in classes:
            other_i = int(other)
            if other_i == cls_i:
                continue
            dm = mu_y - mu_by_class[other_i]
            S_B += np.outer(dm, dm)

        S_risk = np.zeros((d, d), dtype=np.float64)
        risk_count = 0
        risk_vectors: List[np.ndarray] = []
        boundary_idx = idx[boundary_local]
        for gi in boundary_idx.tolist():
            nb = nn_idx_use[gi]
            hetero = nb[y[nb] != cls_i]
            if hetero.size <= 0:
                continue
            use = hetero[: int(max(1, cfg.hetero_k))]
            nu_i = np.mean(X[use], axis=0)
            dv = X[gi] - nu_i
            S_risk += np.outer(dv, dv)
            risk_count += 1
            risk_vectors.append((nu_i - X[gi]).astype(np.float64))
        if risk_count > 0:
            S_risk /= float(risk_count)

        terms[cls_i] = {
            "mu_y": mu_y,
            "S_expand": S_expand,
            "S_B": S_B,
            "S_W": S_W,
            "S_risk": S_risk,
            "class_count": int(len(idx)),
            "class_weight": float(len(idx) / total),
            "interior_count": int(np.sum(interior_local)),
            "boundary_count": int(np.sum(boundary_local)),
            "risk_count": int(risk_count),
            "boundary_to_hetero_vectors": (
                np.vstack(risk_vectors).astype(np.float64)
                if risk_vectors
                else np.empty((0, d), dtype=np.float64)
            ),
            "purity_summary": _summary_stats(purity_y),
            "boundary_purity_summary": _summary_stats(purity_y[boundary_local]),
            "interior_purity_summary": _summary_stats(purity_y[interior_local]),
        }

    meta = {
        "knn_k": int(k_eff),
        "interior_quantile": float(cfg.interior_quantile),
        "boundary_quantile": float(cfg.boundary_quantile),
        "hetero_k": int(cfg.hetero_k),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "purity_summary": _summary_stats(purity),
        "classes": [int(c) for c in classes],
    }
    return terms, meta


def compute_fisher_pia_scores(
    direction_bank: np.ndarray,
    class_terms: Dict[int, Dict[str, object]],
    *,
    cfg: FisherPIAConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    W = np.asarray(direction_bank, dtype=np.float64)
    class_rows: List[Dict[str, object]] = []
    global_rows: List[Dict[str, object]] = []

    classes = sorted(class_terms.keys())
    class_weights = np.asarray([float(class_terms[c]["class_weight"]) for c in classes], dtype=np.float64)
    class_weights = class_weights / max(1e-12, float(np.sum(class_weights)))

    for i in range(W.shape[0]):
        w = W[i]
        local_rows: List[Dict[str, object]] = []
        for cls in classes:
            t = class_terms[cls]
            expand = float(w @ t["S_expand"] @ w)
            between = float(w @ t["S_B"] @ w)
            within = float(w @ t["S_W"] @ w)
            risk = float(w @ t["S_risk"] @ w)
            numer = float(cfg.alpha * expand + cfg.beta * between)
            denom = float(cfg.gamma * within + cfg.eta * risk + cfg.rho)
            score = numer / max(denom, 1e-12)
            row = {
                "direction_id": int(i),
                "class_id": int(cls),
                "expand_score": expand,
                "between_score": between,
                "within_score": within,
                "risk_score": risk,
                "fisher_pia_score": float(score),
                "fisher_pia_numerator": numer,
                "fisher_pia_denominator": denom,
                "class_weight": float(t["class_weight"]),
                "class_count": int(t["class_count"]),
                "interior_count": int(t["interior_count"]),
                "boundary_count": int(t["boundary_count"]),
                "risk_count": int(t["risk_count"]),
            }
            local_rows.append(row)
            class_rows.append(row)

        if local_rows:
            df_i = pd.DataFrame(local_rows)
            global_rows.append(
                {
                    "direction_id": int(i),
                    "class_id": "__global__",
                    "expand_score": float(np.sum(df_i["expand_score"].to_numpy(dtype=np.float64) * class_weights)),
                    "between_score": float(np.sum(df_i["between_score"].to_numpy(dtype=np.float64) * class_weights)),
                    "within_score": float(np.sum(df_i["within_score"].to_numpy(dtype=np.float64) * class_weights)),
                    "risk_score": float(np.sum(df_i["risk_score"].to_numpy(dtype=np.float64) * class_weights)),
                    "fisher_pia_score": float(np.sum(df_i["fisher_pia_score"].to_numpy(dtype=np.float64) * class_weights)),
                    "fisher_pia_numerator": float(np.sum(df_i["fisher_pia_numerator"].to_numpy(dtype=np.float64) * class_weights)),
                    "fisher_pia_denominator": float(np.sum(df_i["fisher_pia_denominator"].to_numpy(dtype=np.float64) * class_weights)),
                    "direction_norm": float(np.linalg.norm(w)),
                }
            )

    class_df = pd.DataFrame(class_rows).sort_values(["direction_id", "class_id"]).reset_index(drop=True)
    global_df = pd.DataFrame(global_rows).sort_values(["direction_id"]).reset_index(drop=True)
    return class_df, global_df


def compute_score_correlations(direction_df: pd.DataFrame) -> pd.DataFrame:
    if direction_df.empty:
        return pd.DataFrame(
            columns=[
                "score_name",
                "metric_name",
                "sign_expectation",
                "pearson_r",
                "spearman_rho",
            ]
        )

    checks = [
        ("fisher_pia_score", "accept_rate", "positive"),
        ("fisher_pia_score", "usage", "positive"),
        ("fisher_pia_score", "flip_rate", "negative"),
        ("fisher_pia_score", "margin_drop_median", "positive"),
        ("fisher_pia_score", "intrusion", "negative"),
        ("risk_score", "flip_rate", "positive"),
        ("risk_score", "intrusion", "positive"),
        ("risk_score", "accept_rate", "negative"),
        ("between_score", "accept_rate", "positive"),
        ("between_score", "intrusion", "negative"),
    ]
    if "gate3_reject_rate_i" in direction_df.columns:
        checks.extend(
            [
                ("fisher_pia_score", "gate3_reject_rate_i", "negative"),
                ("risk_score", "gate3_reject_rate_i", "positive"),
            ]
        )

    rows: List[Dict[str, object]] = []
    for score_name, metric_name, sign in checks:
        if score_name not in direction_df.columns or metric_name not in direction_df.columns:
            continue
        pearson, spearman = _safe_corr(
            direction_df[score_name].to_numpy(dtype=np.float64),
            direction_df[metric_name].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "score_name": score_name,
                "metric_name": metric_name,
                "sign_expectation": sign,
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )
    return pd.DataFrame(rows)


def compute_generic_score_correlations(
    direction_df: pd.DataFrame,
    *,
    score_name: str,
    extra_pairs: List[Tuple[str, str]] | None = None,
) -> pd.DataFrame:
    if direction_df.empty or score_name not in direction_df.columns:
        return pd.DataFrame(
            columns=[
                "score_name",
                "metric_name",
                "sign_expectation",
                "pearson_r",
                "spearman_rho",
            ]
        )
    pairs: List[Tuple[str, str]] = [
        ("accept_rate", "positive"),
        ("usage", "positive"),
        ("flip_rate", "negative"),
        ("margin_drop_median", "positive"),
        ("intrusion", "negative"),
        ("gate3_reject_rate_i", "negative"),
    ]
    if extra_pairs:
        pairs.extend(extra_pairs)

    rows: List[Dict[str, object]] = []
    for metric_name, sign in pairs:
        if metric_name not in direction_df.columns:
            continue
        pearson, spearman = _safe_corr(
            direction_df[score_name].to_numpy(dtype=np.float64),
            direction_df[metric_name].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "score_name": score_name,
                "metric_name": metric_name,
                "sign_expectation": sign,
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )
    return pd.DataFrame(rows)


def summarize_score_signal(direction_df: pd.DataFrame) -> Dict[str, object]:
    if direction_df.empty or int(direction_df.shape[0]) < 2:
        return {
            "n_directions": int(direction_df.shape[0]),
            "score_signal_pass": False,
            "risk_signal_pass": False,
        }

    df = direction_df.sort_values("fisher_pia_score", ascending=False).reset_index(drop=True)
    top_n = max(1, int(np.ceil(df.shape[0] / 2.0)))
    top = df.iloc[:top_n]
    bottom = df.iloc[-top_n:]

    delta_accept = None
    if "accept_rate" in df.columns:
        delta_accept = float(top["accept_rate"].mean() - bottom["accept_rate"].mean())
    delta_flip = None
    if "flip_rate" in df.columns:
        delta_flip = float(top["flip_rate"].mean() - bottom["flip_rate"].mean())
    delta_intrusion = None
    if "intrusion" in df.columns:
        delta_intrusion = float(top["intrusion"].mean() - bottom["intrusion"].mean())
    delta_margin = None
    if "margin_drop_median" in df.columns:
        delta_margin = float(top["margin_drop_median"].mean() - bottom["margin_drop_median"].mean())

    risk_top = None
    risk_bottom = None
    if "risk_score" in df.columns:
        risk_top = float(top["risk_score"].mean())
        risk_bottom = float(bottom["risk_score"].mean())

    score_signal_pass = bool(
        (delta_accept is not None and delta_accept > 0.0)
        and (
            (delta_intrusion is not None and delta_intrusion < 0.0)
            or (delta_flip is not None and delta_flip < 0.0)
        )
    )
    risk_signal_pass = bool(
        (risk_top is not None and risk_bottom is not None and risk_top < risk_bottom)
        and (
            (delta_intrusion is not None and delta_intrusion < 0.0)
            or (delta_flip is not None and delta_flip < 0.0)
        )
    )

    return {
        "n_directions": int(df.shape[0]),
        "score_signal_pass": score_signal_pass,
        "risk_signal_pass": risk_signal_pass,
        "top_direction_id": int(df.iloc[0]["direction_id"]),
        "bottom_direction_id": int(df.iloc[-1]["direction_id"]),
        "delta_accept_top_vs_bottom": delta_accept,
        "delta_flip_top_vs_bottom": delta_flip,
        "delta_intrusion_top_vs_bottom": delta_intrusion,
        "delta_margin_top_vs_bottom": delta_margin,
        "top_mean_risk_score": risk_top,
        "bottom_mean_risk_score": risk_bottom,
    }


def summarize_generic_score_signal(
    direction_df: pd.DataFrame,
    *,
    score_name: str,
    lower_is_better_metrics: List[str] | None = None,
) -> Dict[str, object]:
    if direction_df.empty or int(direction_df.shape[0]) < 2 or score_name not in direction_df.columns:
        return {
            "n_directions": int(direction_df.shape[0]),
            "score_signal_pass": False,
        }
    lower_is_better = set(lower_is_better_metrics or ["flip_rate", "intrusion"])
    df = direction_df.sort_values(score_name, ascending=False).reset_index(drop=True)
    top_n = max(1, int(np.ceil(df.shape[0] / 2.0)))
    top = df.iloc[:top_n]
    bottom = df.iloc[-top_n:]

    deltas: Dict[str, float | None] = {}
    signal_checks: List[bool] = []
    for metric in ["accept_rate", "intrusion", "flip_rate", "margin_drop_median", "usage", "gate3_reject_rate_i"]:
        if metric not in df.columns:
            deltas[f"delta_{metric}_top_vs_bottom"] = None
            continue
        delta = float(top[metric].mean() - bottom[metric].mean())
        deltas[f"delta_{metric}_top_vs_bottom"] = delta
        if metric == "accept_rate":
            signal_checks.append(delta > 0.0)
        elif metric in lower_is_better:
            signal_checks.append(delta < 0.0)
        elif metric == "margin_drop_median":
            signal_checks.append(delta > 0.0)

    return {
        "n_directions": int(df.shape[0]),
        "score_signal_pass": bool(all(signal_checks)) if signal_checks else False,
        "top_direction_id": int(df.iloc[0]["direction_id"]),
        "bottom_direction_id": int(df.iloc[-1]["direction_id"]),
        **deltas,
    }


def compute_safe_axis_scores(
    direction_bank: np.ndarray,
    class_terms: Dict[int, Dict[str, object]],
    *,
    beta: float,
    gamma: float = 0.0,
    include_approach: bool = False,
    direction_score_mode: str = "axis_level",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    W = np.asarray(direction_bank, dtype=np.float64)
    classes = sorted(class_terms.keys())
    class_weights = np.asarray([float(class_terms[c]["class_weight"]) for c in classes], dtype=np.float64)
    class_weights = class_weights / max(1e-12, float(np.sum(class_weights)))
    mu_by_class = {int(c): np.asarray(class_terms[c]["mu_y"], dtype=np.float64) for c in classes}

    class_rows: List[Dict[str, object]] = []
    global_raw_rows: List[Dict[str, object]] = []
    for i in range(W.shape[0]):
        w = W[i]
        local_rows: List[Dict[str, object]] = []
        for cls in classes:
            t = class_terms[cls]
            expand = float(w @ t["S_expand"] @ w)
            risk = float(w @ t["S_risk"] @ w)
            if include_approach:
                app = 0.0
                mu_y = mu_by_class[int(cls)]
                for other in classes:
                    if int(other) == int(cls):
                        continue
                    app += max(0.0, float(np.dot(w, mu_by_class[int(other)] - mu_y)))
            else:
                app = 0.0
            row = {
                "direction_id": int(i),
                "class_id": int(cls),
                "direction_score_mode": direction_score_mode,
                "expand_score": expand,
                "risk_score": risk,
                "approach_score": float(app) if include_approach else None,
                "class_weight": float(t["class_weight"]),
            }
            class_rows.append(row)
            local_rows.append(row)

        df_i = pd.DataFrame(local_rows)
        global_raw_rows.append(
            {
                "direction_id": int(i),
                "class_id": "__global__",
                "direction_score_mode": direction_score_mode,
                "expand_score": float(np.sum(df_i["expand_score"].to_numpy(dtype=np.float64) * class_weights)),
                "risk_score": float(np.sum(df_i["risk_score"].to_numpy(dtype=np.float64) * class_weights)),
                "approach_score": (
                    float(np.sum(df_i["approach_score"].fillna(0.0).to_numpy(dtype=np.float64) * class_weights))
                    if include_approach
                    else None
                ),
                "direction_norm": float(np.linalg.norm(w)),
            }
        )

    global_df = pd.DataFrame(global_raw_rows).sort_values("direction_id").reset_index(drop=True)
    global_df["expand_tilde"] = _minmax_norm(global_df["expand_score"].to_numpy(dtype=np.float64))
    global_df["risk_tilde"] = _minmax_norm(global_df["risk_score"].to_numpy(dtype=np.float64))
    if include_approach:
        global_df["approach_tilde"] = _minmax_norm(global_df["approach_score"].fillna(0.0).to_numpy(dtype=np.float64))
    else:
        global_df["approach_tilde"] = None

    revised = global_df["expand_tilde"].to_numpy(dtype=np.float64) - float(beta) * global_df["risk_tilde"].to_numpy(dtype=np.float64)
    if include_approach:
        revised = revised - float(gamma) * global_df["approach_tilde"].to_numpy(dtype=np.float64)
    global_df["revised_score"] = revised.astype(np.float64)

    class_df = pd.DataFrame(class_rows).sort_values(["direction_id", "class_id"]).reset_index(drop=True)
    class_df["expand_tilde"] = 0.0
    class_df["risk_tilde"] = 0.0
    class_df["approach_tilde"] = None if not include_approach else 0.0
    class_df["revised_score"] = 0.0
    for cls in classes:
        mask = class_df["class_id"] == int(cls)
        expand_norm = _minmax_norm(class_df.loc[mask, "expand_score"].to_numpy(dtype=np.float64))
        risk_norm = _minmax_norm(class_df.loc[mask, "risk_score"].to_numpy(dtype=np.float64))
        class_df.loc[mask, "expand_tilde"] = expand_norm
        class_df.loc[mask, "risk_tilde"] = risk_norm
        revised_cls = expand_norm - float(beta) * risk_norm
        if include_approach:
            app_norm = _minmax_norm(class_df.loc[mask, "approach_score"].fillna(0.0).to_numpy(dtype=np.float64))
            class_df.loc[mask, "approach_tilde"] = app_norm
            revised_cls = revised_cls - float(gamma) * app_norm
        class_df.loc[mask, "revised_score"] = revised_cls
    return class_df, global_df


def compute_safe_directed_scores(
    direction_bank: np.ndarray,
    class_terms: Dict[int, Dict[str, object]],
    *,
    beta: float,
    gamma: float = 0.0,
    include_approach: bool = False,
    direction_score_mode: str = "directed_plus_minus",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    W = np.asarray(direction_bank, dtype=np.float64)
    classes = sorted(class_terms.keys())
    class_weights = np.asarray([float(class_terms[c]["class_weight"]) for c in classes], dtype=np.float64)
    class_weights = class_weights / max(1e-12, float(np.sum(class_weights)))
    mu_by_class = {int(c): np.asarray(class_terms[c]["mu_y"], dtype=np.float64) for c in classes}

    class_rows: List[Dict[str, object]] = []
    global_rows: List[Dict[str, object]] = []
    signs = [("plus", 1.0), ("minus", -1.0)]

    for i in range(W.shape[0]):
        w_axis = W[i]
        expand_axis_per_class: Dict[int, float] = {}
        for cls in classes:
            t = class_terms[cls]
            expand_axis_per_class[int(cls)] = float(w_axis @ t["S_expand"] @ w_axis)

        for sign_name, sign_value in signs:
            w_dir = float(sign_value) * w_axis
            local_rows: List[Dict[str, object]] = []
            for cls in classes:
                t = class_terms[cls]
                expand = float(expand_axis_per_class[int(cls)])
                risk_vectors = np.asarray(t.get("boundary_to_hetero_vectors", np.empty((0, W.shape[1]))), dtype=np.float64)
                if risk_vectors.size:
                    risk = float(np.mean(np.maximum(0.0, risk_vectors @ w_dir)))
                else:
                    risk = 0.0
                if include_approach:
                    mu_y = mu_by_class[int(cls)]
                    app = 0.0
                    for other in classes:
                        other_i = int(other)
                        if other_i == int(cls):
                            continue
                        app += max(0.0, float(np.dot(w_dir, mu_by_class[other_i] - mu_y)))
                else:
                    app = 0.0
                row = {
                    "direction_id": int(i),
                    "sign": sign_name,
                    "class_id": int(cls),
                    "direction_score_mode": direction_score_mode,
                    "expand_score": expand,
                    "risk_score": risk,
                    "approach_score": float(app) if include_approach else None,
                    "class_weight": float(t["class_weight"]),
                }
                class_rows.append(row)
                local_rows.append(row)

            df_local = pd.DataFrame(local_rows)
            global_rows.append(
                {
                    "direction_id": int(i),
                    "sign": sign_name,
                    "class_id": "__global__",
                    "direction_score_mode": direction_score_mode,
                    "expand_score": float(np.sum(df_local["expand_score"].to_numpy(dtype=np.float64) * class_weights)),
                    "risk_score": float(np.sum(df_local["risk_score"].to_numpy(dtype=np.float64) * class_weights)),
                    "approach_score": (
                        float(np.sum(df_local["approach_score"].fillna(0.0).to_numpy(dtype=np.float64) * class_weights))
                        if include_approach
                        else None
                    ),
                    "direction_norm": float(np.linalg.norm(w_axis)),
                }
            )

    global_df = pd.DataFrame(global_rows).sort_values(["direction_id", "sign"]).reset_index(drop=True)
    global_df["expand_tilde"] = _minmax_norm(global_df["expand_score"].to_numpy(dtype=np.float64))
    global_df["risk_tilde"] = _minmax_norm(global_df["risk_score"].to_numpy(dtype=np.float64))
    if include_approach:
        global_df["approach_tilde"] = _minmax_norm(global_df["approach_score"].fillna(0.0).to_numpy(dtype=np.float64))
    else:
        global_df["approach_tilde"] = None
    revised = global_df["expand_tilde"].to_numpy(dtype=np.float64) - float(beta) * global_df["risk_tilde"].to_numpy(dtype=np.float64)
    if include_approach:
        revised = revised - float(gamma) * global_df["approach_tilde"].to_numpy(dtype=np.float64)
    global_df["revised_score"] = revised.astype(np.float64)

    class_df = pd.DataFrame(class_rows).sort_values(["direction_id", "sign", "class_id"]).reset_index(drop=True)
    class_df["expand_tilde"] = 0.0
    class_df["risk_tilde"] = 0.0
    class_df["approach_tilde"] = None if not include_approach else 0.0
    class_df["revised_score"] = 0.0
    for cls in classes:
        mask = class_df["class_id"] == int(cls)
        expand_norm = _minmax_norm(class_df.loc[mask, "expand_score"].to_numpy(dtype=np.float64))
        risk_norm = _minmax_norm(class_df.loc[mask, "risk_score"].to_numpy(dtype=np.float64))
        class_df.loc[mask, "expand_tilde"] = expand_norm
        class_df.loc[mask, "risk_tilde"] = risk_norm
        revised_cls = expand_norm - float(beta) * risk_norm
        if include_approach:
            app_norm = _minmax_norm(class_df.loc[mask, "approach_score"].fillna(0.0).to_numpy(dtype=np.float64))
            class_df.loc[mask, "approach_tilde"] = app_norm
            revised_cls = revised_cls - float(gamma) * app_norm
        class_df.loc[mask, "revised_score"] = revised_cls
    return class_df, global_df
