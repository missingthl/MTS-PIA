#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    load_trials_for_dataset,
)
from route_b_unified import PIACore, PIACoreConfig  # noqa: E402
from scripts.route_b.run_route_b_pia_core_config_sweep import _build_rep_state_from_trials  # noqa: E402
from transforms.whiten_color_bridge import logvec_to_spd  # noqa: E402


def _parse_csv_list(text: str) -> List[str]:
    out = [tok.strip().lower() for tok in str(text).split(",") if tok.strip()]
    if not out:
        raise ValueError("csv list cannot be empty")
    return out


def _parse_seed_list(text: str) -> List[int]:
    out = sorted(set(int(tok.strip()) for tok in str(text).split(",") if tok.strip()))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_official_split_trials(dataset: str, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    all_trials = load_trials_for_dataset(
        dataset=dataset,
        natops_root=str(args.natops_root),
        selfregulationscp1_root=str(args.selfregulationscp1_root),
    )
    train_trials = [t for t in all_trials if str(t.get("split", "")).lower() == "train"]
    test_trials = [t for t in all_trials if str(t.get("split", "")).lower() == "test"]
    if not train_trials or not test_trials:
        raise RuntimeError(f"expected official train/test split trials for dataset={dataset}")
    return train_trials, test_trials


def _dataset_second_axis_scale(dataset: str, args: argparse.Namespace) -> float:
    ds = str(dataset).lower()
    if ds == "natops":
        return float(args.natops_second_axis_scale)
    if ds == "selfregulationscp1":
        return float(args.scp1_second_axis_scale)
    raise ValueError(f"unsupported dataset: {dataset}")


def _build_embedding(
    X_orig: np.ndarray,
    X_aug: np.ndarray,
    *,
    method: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    x_all = np.concatenate([np.asarray(X_orig, dtype=np.float64), np.asarray(X_aug, dtype=np.float64)], axis=0)
    chosen = str(method).lower()
    meta: Dict[str, object] = {}
    if chosen == "auto":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=int(seed))
            emb = reducer.fit_transform(x_all)
            chosen = "umap"
            meta["embedding_method"] = "umap"
        except Exception:
            reducer = PCA(n_components=2, random_state=int(seed))
            emb = reducer.fit_transform(x_all)
            chosen = "pca"
            meta["embedding_method"] = "pca_fallback"
    elif chosen == "umap":
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=int(seed))
        emb = reducer.fit_transform(x_all)
        meta["embedding_method"] = "umap"
    else:
        reducer = PCA(n_components=2, random_state=int(seed))
        emb = reducer.fit_transform(x_all)
        meta["embedding_method"] = "pca"

    n_orig = int(X_orig.shape[0])
    meta["embedding_dim"] = 2
    meta["embedding_method_requested"] = str(method)
    return emb[:n_orig], emb[n_orig:], meta


def _cov_ellipse_params(cov: np.ndarray, n_std: float = 2.0) -> Tuple[float, float, float]:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width = 2.0 * n_std * math.sqrt(max(float(vals[0]), 1e-12))
    height = 2.0 * n_std * math.sqrt(max(float(vals[1]), 1e-12))
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle


def _plot_scatter_panels(
    path: str,
    *,
    coords_orig: np.ndarray,
    coords_aug: np.ndarray,
    y: np.ndarray,
    dataset: str,
    title_suffix: str,
) -> None:
    uniq = sorted(set(int(v) for v in np.asarray(y).tolist()))
    cmap = plt.cm.get_cmap("tab10", len(uniq))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    panels = [
        ("orig_z", coords_orig, 0.45),
        ("aug_z", coords_aug, 0.45),
        ("overlay", None, 0.30),
    ]
    for ax, (title, coords, alpha) in zip(axes, panels):
        if title != "overlay":
            for i, cls in enumerate(uniq):
                mask = np.asarray(y) == int(cls)
                ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=alpha, color=cmap(i), label=f"class {cls}")
        else:
            for i, cls in enumerate(uniq):
                mask = np.asarray(y) == int(cls)
                ax.scatter(coords_orig[mask, 0], coords_orig[mask, 1], s=14, alpha=alpha, color=cmap(i), marker="o")
                ax.scatter(coords_aug[mask, 0], coords_aug[mask, 1], s=14, alpha=alpha, color=cmap(i), marker="^")
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(handles)))
    fig.suptitle(f"{dataset} manifold scatter ({title_suffix})")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_embedding_ellipses(
    path: str,
    *,
    coords_orig: np.ndarray,
    coords_aug: np.ndarray,
    y: np.ndarray,
    dataset: str,
    title_suffix: str,
) -> None:
    uniq = sorted(set(int(v) for v in np.asarray(y).tolist()))
    cmap = plt.cm.get_cmap("tab10", len(uniq))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for i, cls in enumerate(uniq):
        mask = np.asarray(y) == int(cls)
        xy_o = coords_orig[mask]
        xy_a = coords_aug[mask]
        mean_o = np.mean(xy_o, axis=0)
        mean_a = np.mean(xy_a, axis=0)
        cov_o = np.cov(xy_o.T) if xy_o.shape[0] >= 2 else np.eye(2, dtype=np.float64) * 1e-6
        cov_a = np.cov(xy_a.T) if xy_a.shape[0] >= 2 else np.eye(2, dtype=np.float64) * 1e-6
        wo, ho, ao = _cov_ellipse_params(cov_o)
        wa, ha, aa = _cov_ellipse_params(cov_a)
        ax.add_patch(Ellipse(mean_o, wo, ho, angle=ao, fill=False, lw=2.0, ls="-", color=cmap(i)))
        ax.add_patch(Ellipse(mean_a, wa, ha, angle=aa, fill=False, lw=2.0, ls="--", color=cmap(i)))
        ax.scatter(mean_o[0], mean_o[1], color=cmap(i), marker="o", s=45, label=f"class {cls} orig")
        ax.scatter(mean_a[0], mean_a[1], color=cmap(i), marker="^", s=45, label=f"class {cls} aug")
    ax.set_title(f"{dataset} class covariance ellipses ({title_suffix})")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq_pairs = list(dict(zip(labels, handles)).items())
        ax.legend([h for _, h in uniq_pairs], [l for l, _ in uniq_pairs], fontsize=8, loc="best")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _spd_mean_from_z(X: np.ndarray, mean_log_train: np.ndarray) -> np.ndarray:
    mats = [logvec_to_spd(v, mean_log_train) for v in np.asarray(X, dtype=np.float64)]
    return np.mean(np.stack(mats, axis=0), axis=0).astype(np.float64)


def _plot_spd_ellipsoids(
    path: str,
    *,
    X_orig: np.ndarray,
    X_aug: np.ndarray,
    y: np.ndarray,
    mean_log_train: np.ndarray,
    dataset: str,
) -> List[Dict[str, object]]:
    uniq = sorted(set(int(v) for v in np.asarray(y).tolist()))
    n_cls = len(uniq)
    fig, axes = plt.subplots(n_cls, 2, figsize=(8.0, max(3.0, 2.8 * n_cls)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    rows: List[Dict[str, object]] = []
    for r, cls in enumerate(uniq):
        mask = np.asarray(y) == int(cls)
        spd_orig = _spd_mean_from_z(X_orig[mask], mean_log_train)
        spd_aug = _spd_mean_from_z(X_aug[mask], mean_log_train)
        for c, (name, spd) in enumerate([("orig", spd_orig), ("aug", spd_aug)]):
            vals, _ = np.linalg.eigh(0.5 * (spd + spd.T))
            vals = np.sort(np.maximum(vals, 1e-12))[::-1]
            a = math.sqrt(float(vals[0]))
            b = math.sqrt(float(vals[1])) if vals.size > 1 else 1e-6
            t = np.linspace(0.0, 2.0 * np.pi, 256)
            x = a * np.cos(t)
            yv = b * np.sin(t)
            ax = axes[r, c]
            ax.plot(x, yv, lw=2.0)
            ax.axhline(0.0, color="grey", lw=0.5)
            ax.axvline(0.0, color="grey", lw=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"class {cls} {name}")
            ax.set_xlabel("eig-1")
            ax.set_ylabel("eig-2")
            lim = max(a, b) * 1.2
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        rows.append(
            {
                "class_id": int(cls),
                "orig_trace": float(np.trace(spd_orig)),
                "aug_trace": float(np.trace(spd_aug)),
                "orig_top1_eig": float(np.sort(np.linalg.eigvalsh(spd_orig))[::-1][0]),
                "orig_top2_eig": float(np.sort(np.linalg.eigvalsh(spd_orig))[::-1][1]),
                "aug_top1_eig": float(np.sort(np.linalg.eigvalsh(spd_aug))[::-1][0]),
                "aug_top2_eig": float(np.sort(np.linalg.eigvalsh(spd_aug))[::-1][1]),
            }
        )
    fig.suptitle(f"{dataset} mean SPD ellipsoids (top-2 spectral view)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Visual probe for manifold geometry: embedding scatter + ellipsoids.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/route_b_manifold_visual_probe_20260329")
    p.add_argument("--embedding-method", type=str, default="auto", choices=["auto", "umap", "pca"])
    p.add_argument("--gamma-main", type=float, default=0.10)
    p.add_argument("--natops-second-axis-scale", type=float, default=0.80)
    p.add_argument("--scp1-second-axis-scale", type=float, default=0.70)
    p.add_argument("--pullback-alpha", type=float, default=0.90)
    p.add_argument("--axis-count", type=int, default=2)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--pia-r-dimension", type=int, default=3)
    p.add_argument("--pia-n-iters", type=int, default=3)
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--pia-activation", type=str, default="sine")
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--pia-orthogonalize", type=int, default=1)
    p.add_argument("--natops-root", type=str, default=DEFAULT_NATOPS_ROOT)
    p.add_argument("--selfregulationscp1-root", type=str, default=DEFAULT_SELFREGULATIONSCP1_ROOT)
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)
    _ensure_dir(args.out_root)

    summary_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []
    spd_rows: List[Dict[str, object]] = []

    pia_cfg = PIACoreConfig(
        r_dimension=int(args.pia_r_dimension),
        n_iters=int(args.pia_n_iters),
        C_repr=float(args.pia_c_repr),
        activation=str(args.pia_activation),
        bias_update_mode=str(args.pia_bias_update_mode),
        orthogonalize=bool(int(args.pia_orthogonalize)),
    )

    for dataset in datasets:
        ds_dir = os.path.join(args.out_root, str(dataset))
        _ensure_dir(ds_dir)
        for seed in seeds:
            train_trials, test_trials = _load_official_split_trials(dataset, args)
            rep_state = _build_rep_state_from_trials(
                dataset=dataset,
                seed=int(seed),
                train_trials=train_trials,
                val_trials=[],
                test_trials=test_trials,
                spd_eps=float(args.spd_eps),
                protocol_type="fixed_split_official",
                protocol_note="dataset-provided official TRAIN/TEST split",
            )
            pia_core = PIACore(replace(pia_cfg, seed=int(seed))).fit(rep_state.X_train)
            ranked_axis_ids, rank_meta = pia_core.rank_axes_by_energy(rep_state.X_train)
            axis_ids = ranked_axis_ids[: int(args.axis_count)]
            second_axis_scale = _dataset_second_axis_scale(dataset, args)
            gamma_vec, gamma_meta = pia_core.build_two_axis_gamma_vector(
                axis_ids=axis_ids,
                gamma_main=float(args.gamma_main),
                second_axis_scale=float(second_axis_scale),
            )
            op_result = pia_core.apply_affine(
                rep_state.X_train,
                gamma_vector=gamma_vec,
                axis_ids=axis_ids,
                pullback_alpha=float(args.pullback_alpha),
            )

            coords_orig, coords_aug, emb_meta = _build_embedding(
                rep_state.X_train,
                op_result.X_aug,
                method=str(args.embedding_method),
                seed=int(seed),
            )

            title_suffix = f"seed={seed}, method={emb_meta['embedding_method']}"
            scatter_path = os.path.join(ds_dir, f"seed{int(seed)}_embedding_scatter.png")
            ellipse_path = os.path.join(ds_dir, f"seed{int(seed)}_embedding_ellipses.png")
            spd_path = os.path.join(ds_dir, f"seed{int(seed)}_spd_ellipsoids.png")
            _plot_scatter_panels(
                scatter_path,
                coords_orig=coords_orig,
                coords_aug=coords_aug,
                y=rep_state.y_train,
                dataset=str(dataset),
                title_suffix=title_suffix,
            )
            _plot_embedding_ellipses(
                ellipse_path,
                coords_orig=coords_orig,
                coords_aug=coords_aug,
                y=rep_state.y_train,
                dataset=str(dataset),
                title_suffix=title_suffix,
            )
            spd_class_rows = _plot_spd_ellipsoids(
                spd_path,
                X_orig=rep_state.X_train,
                X_aug=op_result.X_aug,
                y=rep_state.y_train,
                mean_log_train=rep_state.mean_log_train,
                dataset=str(dataset),
            )

            for i in range(coords_orig.shape[0]):
                point_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "source": "orig",
                        "class_id": int(rep_state.y_train[i]),
                        "x1": float(coords_orig[i, 0]),
                        "x2": float(coords_orig[i, 1]),
                    }
                )
                point_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "source": "aug",
                        "class_id": int(rep_state.y_train[i]),
                        "x1": float(coords_aug[i, 0]),
                        "x2": float(coords_aug[i, 1]),
                    }
                )

            for row in spd_class_rows:
                spd_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        **row,
                    }
                )

            summary_rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "embedding_method": str(emb_meta["embedding_method"]),
                    "z_dim": int(rep_state.X_train.shape[1]),
                    "train_count": int(rep_state.X_train.shape[0]),
                    "axis_ids": json.dumps(axis_ids, ensure_ascii=False),
                    "ranked_axis_ids": json.dumps(ranked_axis_ids, ensure_ascii=False),
                    "rank_meta": json.dumps(rank_meta, ensure_ascii=False),
                    "gamma_main": float(args.gamma_main),
                    "second_axis_scale": float(second_axis_scale),
                    "pullback_alpha": float(args.pullback_alpha),
                    "gamma_vector": json.dumps(gamma_vec, ensure_ascii=False),
                    "gamma_meta": json.dumps(gamma_meta, ensure_ascii=False),
                    "scatter_png": os.path.abspath(scatter_path),
                    "ellipse_png": os.path.abspath(ellipse_path),
                    "spd_ellipsoid_png": os.path.abspath(spd_path),
                }
            )

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_root, "manifold_visual_probe_summary.csv"), index=False)
    pd.DataFrame(point_rows).to_csv(os.path.join(args.out_root, "manifold_visual_probe_points.csv"), index=False)
    pd.DataFrame(spd_rows).to_csv(os.path.join(args.out_root, "manifold_visual_probe_spd_summary.csv"), index=False)

    conclusion_path = os.path.join(args.out_root, "manifold_visual_probe_conclusion.md")
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("# Manifold Visual Probe Conclusion\n\n")
        f.write("更新时间：2026-03-29\n\n")
        f.write("当前 probe 只停在 `representation + PIA Core augmentation`，不经过 bridge 与 raw MiniROCKET。\n\n")
        f.write("输出包含：\n\n")
        f.write("- embedding scatter：orig / aug / overlay\n")
        f.write("- class covariance ellipses：在 2D embedding 中比较类内形状\n")
        f.write("- mean SPD ellipsoids：比较各类 mean SPD 的 top-2 spectral ellipse\n\n")
        f.write("如果环境中没有 `umap-learn`，当前会自动回退到 `PCA` 投影。\n")


if __name__ == "__main__":
    main()
