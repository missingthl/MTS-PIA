from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from transforms.whiten_color_bridge import logvec_to_spd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_embedding_methods(requested: str) -> List[str]:
    methods = [tok.strip().lower() for tok in str(requested).split(",") if tok.strip()]
    if not methods or methods == ["auto"]:
        methods = ["umap", "pca"]
    out: List[str] = []
    for method in methods:
        if method == "umap":
            try:
                import umap  # type: ignore  # noqa: F401

                out.append("umap")
            except Exception:
                pass
        elif method == "pca":
            out.append("pca")
    if not out:
        out = ["pca"]
    return out


def _fit_embedding(X_all: np.ndarray, *, method: str, seed: int) -> np.ndarray:
    x = np.asarray(X_all, dtype=np.float64)
    if str(method).lower() == "umap":
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(5, x.shape[0] - 1)),
            min_dist=0.10,
            random_state=int(seed),
        )
        return np.asarray(reducer.fit_transform(x), dtype=np.float64)

    reducer = PCA(n_components=2, random_state=int(seed))
    return np.asarray(reducer.fit_transform(x), dtype=np.float64)


def build_embedding_maps(
    X_by_operator: Mapping[str, np.ndarray],
    *,
    method: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    names = list(X_by_operator.keys())
    sizes = [int(np.asarray(X_by_operator[name]).shape[0]) for name in names]
    X_all = np.concatenate([np.asarray(X_by_operator[name], dtype=np.float64) for name in names], axis=0)
    emb = _fit_embedding(X_all, method=str(method), seed=int(seed))
    out: Dict[str, np.ndarray] = {}
    start = 0
    for name, size in zip(names, sizes):
        out[str(name)] = np.asarray(emb[start : start + size], dtype=np.float64)
        start += size
    return out


def _cross_class_neighbor_ratio(coords: np.ndarray, y: np.ndarray, *, k: int) -> float:
    x = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    n = int(x.shape[0])
    if n <= 2:
        return 0.0
    n_neighbors = min(int(k) + 1, n)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(x)
    idx = nn.kneighbors(return_distance=False)
    ratios: List[float] = []
    for i in range(n):
        nbrs = [int(j) for j in idx[i].tolist() if int(j) != i]
        if not nbrs:
            continue
        cross = float(np.mean(labels[np.asarray(nbrs, dtype=np.int64)] != labels[i]))
        ratios.append(cross)
    return float(np.mean(ratios)) if ratios else 0.0


def compute_projection_summary(
    coords: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
) -> Dict[str, float]:
    xy = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    uniq = sorted(set(int(v) for v in labels.tolist()))
    compactness_vals: List[float] = []
    centroids: List[np.ndarray] = []
    local_density_vals: List[float] = []
    for cls in uniq:
        mask = labels == int(cls)
        pts = xy[mask]
        if pts.shape[0] == 0:
            continue
        centroid = np.mean(pts, axis=0)
        centroids.append(centroid)
        compactness_vals.append(float(np.mean(np.linalg.norm(pts - centroid[None, :], axis=1))))
        if pts.shape[0] >= 2:
            k_local = min(max(1, int(k)), pts.shape[0] - 1)
            dists = pairwise_distances(pts)
            np.fill_diagonal(dists, np.inf)
            nearest = np.partition(dists, kth=k_local - 1, axis=1)[:, :k_local]
            local_density_vals.append(float(np.mean(nearest)))
    if len(centroids) >= 2:
        cent_arr = np.stack(centroids, axis=0)
        pdist = pairwise_distances(cent_arr)
        interclass = float(np.mean(pdist[np.triu_indices_from(pdist, k=1)]))
    else:
        interclass = 0.0
    overlap = _cross_class_neighbor_ratio(xy, labels, k=int(k))
    return {
        "classwise_compactness": float(np.mean(compactness_vals)) if compactness_vals else 0.0,
        "interclass_separation": float(interclass),
        "overlap_proxy": float(overlap),
        "local_density_proxy": float(np.mean(local_density_vals)) if local_density_vals else 0.0,
    }


def _largest_cc_ratio(num_nodes: int, edges: Mapping[int, Sequence[int]]) -> float:
    if num_nodes <= 1:
        return 1.0
    visited = np.zeros((num_nodes,), dtype=bool)
    best = 0
    for start in range(num_nodes):
        if visited[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = True
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in edges.get(cur, []):
                j = int(nxt)
                if not visited[j]:
                    visited[j] = True
                    q.append(j)
        best = max(best, size)
    return float(best / max(1, num_nodes))


def compute_neighborhood_summary(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
) -> Dict[str, float]:
    z = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    n = int(z.shape[0])
    if n <= 1:
        return {
            "intra_class_nn_distance": 0.0,
            "inter_class_nn_distance": 0.0,
            "cross_class_neighbor_ratio": 0.0,
            "connectivity_proxy": 1.0,
        }

    dmat = pairwise_distances(z)
    np.fill_diagonal(dmat, np.inf)
    intra_vals: List[float] = []
    inter_vals: List[float] = []
    for i in range(n):
        same_mask = labels == labels[i]
        same_mask[i] = False
        diff_mask = labels != labels[i]
        if np.any(same_mask):
            intra_vals.append(float(np.min(dmat[i, same_mask])))
        if np.any(diff_mask):
            inter_vals.append(float(np.min(dmat[i, diff_mask])))

    nn = NearestNeighbors(n_neighbors=min(int(k) + 1, n))
    nn.fit(z)
    nbr_idx = nn.kneighbors(return_distance=False)
    cross_ratios: List[float] = []
    for i in range(n):
        nbrs = [int(j) for j in nbr_idx[i].tolist() if int(j) != i]
        if not nbrs:
            continue
        cross_ratios.append(float(np.mean(labels[np.asarray(nbrs, dtype=np.int64)] != labels[i])))

    uniq = sorted(set(int(v) for v in labels.tolist()))
    cc_ratios: List[float] = []
    for cls in uniq:
        idx_cls = np.where(labels == int(cls))[0]
        pts = z[idx_cls]
        n_cls = int(pts.shape[0])
        if n_cls <= 1:
            cc_ratios.append(1.0)
            continue
        k_cls = min(max(1, int(k)), n_cls - 1)
        nn_cls = NearestNeighbors(n_neighbors=k_cls + 1)
        nn_cls.fit(pts)
        cls_idx = nn_cls.kneighbors(return_distance=False)
        edges: Dict[int, List[int]] = {}
        for i in range(n_cls):
            nbrs = [int(j) for j in cls_idx[i].tolist() if int(j) != i]
            edges.setdefault(i, []).extend(nbrs)
            for j in nbrs:
                edges.setdefault(j, []).append(i)
        cc_ratios.append(_largest_cc_ratio(n_cls, edges))

    return {
        "intra_class_nn_distance": float(np.mean(intra_vals)) if intra_vals else 0.0,
        "inter_class_nn_distance": float(np.mean(inter_vals)) if inter_vals else 0.0,
        "cross_class_neighbor_ratio": float(np.mean(cross_ratios)) if cross_ratios else 0.0,
        "connectivity_proxy": float(np.mean(cc_ratios)) if cc_ratios else 0.0,
    }


def _cov_ellipse_params(cov: np.ndarray, n_std: float = 2.0) -> Tuple[float, float, float]:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width = 2.0 * n_std * math.sqrt(max(float(vals[0]), 1e-12))
    height = 2.0 * n_std * math.sqrt(max(float(vals[1]), 1e-12))
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle


def plot_projection_panels(
    path: str,
    *,
    coords_by_operator: Mapping[str, np.ndarray],
    y: np.ndarray,
    dataset: str,
    seed: int,
    method: str,
) -> None:
    labels = np.asarray(y, dtype=np.int64)
    uniq = sorted(set(int(v) for v in labels.tolist()))
    cmap = plt.cm.get_cmap("tab10", len(uniq))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    order = ["orig", "vector", "logeuclidean", "overlay"]
    for ax, title in zip(axes, order):
        if title != "overlay":
            coords = np.asarray(coords_by_operator[title], dtype=np.float64)
            for i, cls in enumerate(uniq):
                mask = labels == int(cls)
                ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.45, color=cmap(i), label=f"class {cls}")
        else:
            markers = {"orig": "o", "vector": "^", "logeuclidean": "s"}
            alphas = {"orig": 0.20, "vector": 0.28, "logeuclidean": 0.28}
            for op_name in ["orig", "vector", "logeuclidean"]:
                coords = np.asarray(coords_by_operator[op_name], dtype=np.float64)
                for i, cls in enumerate(uniq):
                    mask = labels == int(cls)
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        s=12,
                        alpha=alphas[op_name],
                        color=cmap(i),
                        marker=markers[op_name],
                    )
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(8, len(handles)))
    fig.suptitle(f"{dataset} manifold projection ({method}, seed={seed})")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_neighborhood_summary(
    path: str,
    *,
    stats_by_operator: Mapping[str, Mapping[str, float]],
    dataset: str,
    seed: int,
) -> None:
    operators = ["orig", "vector", "logeuclidean"]
    metrics = [
        "intra_class_nn_distance",
        "inter_class_nn_distance",
        "cross_class_neighbor_ratio",
        "connectivity_proxy",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.2), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        vals = [float(stats_by_operator[op][metric]) for op in operators]
        ax.bar(np.arange(len(operators)), vals, color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_xticks(np.arange(len(operators)))
        ax.set_xticklabels(operators, rotation=20)
        ax.set_title(metric)
    fig.suptitle(f"{dataset} neighborhood diagnostics (seed={seed})")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _mean_spd_from_z(X: np.ndarray, mean_log_train: np.ndarray) -> np.ndarray:
    mats = [logvec_to_spd(v, mean_log_train) for v in np.asarray(X, dtype=np.float64)]
    return np.mean(np.stack(mats, axis=0), axis=0).astype(np.float64)


def compute_ellipsoid_summary(
    X_by_operator: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    mean_log_train: np.ndarray,
) -> List[Dict[str, object]]:
    labels = np.asarray(y, dtype=np.int64)
    uniq = sorted(set(int(v) for v in labels.tolist()))
    orig_by_class: Dict[int, np.ndarray] = {}
    orig_topvec: Dict[int, np.ndarray] = {}
    rows: List[Dict[str, object]] = []

    for cls in uniq:
        mask = labels == int(cls)
        spd_orig = _mean_spd_from_z(np.asarray(X_by_operator["orig"], dtype=np.float64)[mask], mean_log_train)
        vals_o, vecs_o = np.linalg.eigh(0.5 * (spd_orig + spd_orig.T))
        order_o = np.argsort(vals_o)[::-1]
        vals_o = vals_o[order_o]
        vecs_o = vecs_o[:, order_o]
        orig_by_class[int(cls)] = spd_orig
        orig_topvec[int(cls)] = vecs_o[:, 0]

    for op_name, X_op in X_by_operator.items():
        X_arr = np.asarray(X_op, dtype=np.float64)
        for cls in uniq:
            mask = labels == int(cls)
            spd = _mean_spd_from_z(X_arr[mask], mean_log_train)
            vals, vecs = np.linalg.eigh(0.5 * (spd + spd.T))
            order = np.argsort(vals)[::-1]
            vals = np.maximum(vals[order], 1e-12)
            vecs = vecs[:, order]
            topk = [float(math.sqrt(v)) for v in vals[: min(3, vals.shape[0])].tolist()]
            anisotropy = float(vals[0] / max(vals[1], 1e-12)) if vals.shape[0] >= 2 else 1.0
            if str(op_name) == "orig":
                orient_shift = 0.0
            else:
                cosine = float(abs(np.dot(vecs[:, 0], orig_topvec[int(cls)])))
                cosine = min(1.0, max(0.0, cosine))
                orient_shift = float(np.degrees(np.arccos(cosine)))
            rows.append(
                {
                    "operator_type": str(op_name),
                    "class_id": int(cls),
                    "mean_axis_length_topk": json.dumps([round(v, 6) for v in topk], ensure_ascii=False),
                    "anisotropy_ratio": float(anisotropy),
                    "orientation_shift_proxy": float(orient_shift),
                    "notes": "",
                }
            )
    return rows


def plot_ellipsoid_summary(
    path: str,
    *,
    X_by_operator: Mapping[str, np.ndarray],
    y: np.ndarray,
    mean_log_train: np.ndarray,
    dataset: str,
    seed: int,
) -> None:
    labels = np.asarray(y, dtype=np.int64)
    uniq = sorted(set(int(v) for v in labels.tolist()))
    operators = ["orig", "vector", "logeuclidean"]
    fig, axes = plt.subplots(len(uniq), len(operators), figsize=(10.2, max(3.0, 2.9 * len(uniq))), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(uniq):
        mask = labels == int(cls)
        for c, op_name in enumerate(operators):
            spd = _mean_spd_from_z(np.asarray(X_by_operator[op_name], dtype=np.float64)[mask], mean_log_train)
            vals = np.sort(np.maximum(np.linalg.eigvalsh(0.5 * (spd + spd.T)), 1e-12))[::-1]
            a = math.sqrt(float(vals[0]))
            b = math.sqrt(float(vals[1])) if vals.shape[0] >= 2 else 1e-6
            t = np.linspace(0.0, 2.0 * np.pi, 256)
            x = a * np.cos(t)
            yv = b * np.sin(t)
            ax = axes[r, c]
            ax.plot(x, yv, lw=2.0)
            ax.axhline(0.0, color="grey", lw=0.5)
            ax.axvline(0.0, color="grey", lw=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"class {cls} {op_name}")
            ax.set_xlabel("eig-1")
            ax.set_ylabel("eig-2")
            lim = max(a, b) * 1.2
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
    fig.suptitle(f"{dataset} mean SPD ellipsoids (seed={seed})")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

