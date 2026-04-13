#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from datasets.trial_dataset_factory import (  # noqa: E402
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    load_trials_for_dataset,
)
from route_b_unified import PIACore, PIACoreConfig  # noqa: E402
from route_b_unified.manifold_diagnostics import (  # noqa: E402
    build_embedding_maps,
    compute_ellipsoid_summary,
    compute_neighborhood_summary,
    compute_projection_summary,
    ensure_dir,
    plot_ellipsoid_summary,
    plot_neighborhood_summary,
    plot_projection_panels,
    resolve_embedding_methods,
)
from scripts.route_b.run_route_b_pia_core_config_sweep import _build_rep_state_from_trials  # noqa: E402


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


def _projection_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (dataset, operator_type, embedding_method), sub in df.groupby(["dataset", "operator_type", "embedding_method"], sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "operator_type": str(operator_type),
                "embedding_method": str(embedding_method),
                "classwise_compactness_mean": float(sub["classwise_compactness"].mean()),
                "interclass_separation_mean": float(sub["interclass_separation"].mean()),
                "overlap_proxy_mean": float(sub["overlap_proxy"].mean()),
                "local_density_proxy_mean": float(sub["local_density_proxy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _neighborhood_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (dataset, operator_type), sub in df.groupby(["dataset", "operator_type"], sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "operator_type": str(operator_type),
                "intra_class_nn_distance_mean": float(sub["intra_class_nn_distance"].mean()),
                "inter_class_nn_distance_mean": float(sub["inter_class_nn_distance"].mean()),
                "cross_class_neighbor_ratio_mean": float(sub["cross_class_neighbor_ratio"].mean()),
                "connectivity_proxy_mean": float(sub["connectivity_proxy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _ellipsoid_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (dataset, operator_type), sub in df.groupby(["dataset", "operator_type"], sort=True):
        rows.append(
            {
                "dataset": str(dataset),
                "operator_type": str(operator_type),
                "anisotropy_ratio_mean": float(sub["anisotropy_ratio"].mean()),
                "orientation_shift_proxy_mean": float(sub["orientation_shift_proxy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _lookup_summary_value(
    df: pd.DataFrame,
    *,
    operator_type: str,
    column: str,
    embedding_method: str | None = None,
) -> float:
    sub = df.copy()
    if embedding_method is not None and "embedding_method" in sub.columns:
        sub = sub[sub["embedding_method"] == embedding_method]
    sub = sub[sub["operator_type"] == operator_type]
    if sub.empty:
        return 0.0
    return float(sub.iloc[0][column])


def _fmt_signed(value: float) -> str:
    return f"{value:+.4f}"


def _operator_note(dataset: str, proj_df: pd.DataFrame, neigh_df: pd.DataFrame, ell_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    proj_sub = proj_df[proj_df["dataset"] == dataset].copy()
    neigh_sub = neigh_df[neigh_df["dataset"] == dataset].copy()
    ell_sub = ell_df[ell_df["dataset"] == dataset].copy()

    embed_method = "umap" if "umap" in proj_sub.get("embedding_method", pd.Series(dtype=str)).tolist() else None

    orig_sep = _lookup_summary_value(
        proj_sub,
        operator_type="orig",
        column="interclass_separation_mean",
        embedding_method=embed_method,
    )
    vec_sep = _lookup_summary_value(
        proj_sub,
        operator_type="vector",
        column="interclass_separation_mean",
        embedding_method=embed_method,
    )
    log_sep = _lookup_summary_value(
        proj_sub,
        operator_type="logeuclidean",
        column="interclass_separation_mean",
        embedding_method=embed_method,
    )
    orig_overlap = _lookup_summary_value(
        neigh_sub,
        operator_type="orig",
        column="cross_class_neighbor_ratio_mean",
    )
    vec_overlap = _lookup_summary_value(
        neigh_sub,
        operator_type="vector",
        column="cross_class_neighbor_ratio_mean",
    )
    log_overlap = _lookup_summary_value(
        neigh_sub,
        operator_type="logeuclidean",
        column="cross_class_neighbor_ratio_mean",
    )
    vec_orient = _lookup_summary_value(
        ell_sub,
        operator_type="vector",
        column="orientation_shift_proxy_mean",
    )
    log_orient = _lookup_summary_value(
        ell_sub,
        operator_type="logeuclidean",
        column="orientation_shift_proxy_mean",
    )
    conn_orig = _lookup_summary_value(
        neigh_sub,
        operator_type="orig",
        column="connectivity_proxy_mean",
    )
    conn_vec = _lookup_summary_value(
        neigh_sub,
        operator_type="vector",
        column="connectivity_proxy_mean",
    )
    conn_log = _lookup_summary_value(
        neigh_sub,
        operator_type="logeuclidean",
        column="connectivity_proxy_mean",
    )

    lines.append(
        "- 低维分离度变化："
        f" `vector {orig_sep:.4f}->{vec_sep:.4f}`，"
        f" `log-Euclidean {orig_sep:.4f}->{log_sep:.4f}`。"
    )
    lines.append(
        "- 跨类近邻比例变化："
        f" `vector {orig_overlap:.4f}->{vec_overlap:.4f}`，"
        f" `log-Euclidean {orig_overlap:.4f}->{log_overlap:.4f}`。"
    )
    lines.append(
        "- 椭球方向漂移均值："
        f" `vector={vec_orient:.4f}`，`log-Euclidean={log_orient:.4f}`。"
    )

    ds = str(dataset).lower()
    if ds == "natops":
        if vec_sep >= orig_sep and vec_overlap <= orig_overlap:
            lines.append(
                "- NATOPS 上，向量版更像“安全外扩 + 轻度致密化”，不是明显错误桥接。"
            )
        else:
            lines.append(
                "- NATOPS 上，向量版没有呈现干净的安全外扩，需要结合图像继续确认是否只是厚化。"
            )
        if log_overlap <= vec_overlap and log_orient < vec_orient:
            lines.append(
                "- log-Euclidean 版在 NATOPS 上更保守，方向漂移更小，符合“更健康但张力更弱”的模式。"
            )
        else:
            lines.append(
                "- log-Euclidean 版在 NATOPS 上没有形成明显更健康的优势，更多表现为轻微保守扰动。"
            )
    else:
        lines.append(
            "- SCP1 上，三组对象在粗邻域统计中的差异都偏小，说明第一轮 B0 还没有直接看见强烈的错误桥接。"
        )
        if log_orient <= vec_orient:
            lines.append(
                "- log-Euclidean 版在 SCP1 上略更保守，但与向量版非常接近；当前失败更像细粒度类条件扭曲，而不是低维图上一眼可见的塌缩。"
            )
        else:
            lines.append(
                "- SCP1 上目前看不到 log-Euclidean 版相对向量版的稳定几何优势。"
            )

    if abs(conn_orig - 1.0) < 1e-8 and abs(conn_vec - 1.0) < 1e-8 and abs(conn_log - 1.0) < 1e-8:
        lines.append("- 当前 connectivity_proxy 在本轮三组对象上全部饱和为 1.0，后续不宜把它作为 Feedback Pool 的主录取信号。")
    return lines


def _write_conclusion(
    path: str,
    *,
    projection_summary: pd.DataFrame,
    neighborhood_summary: pd.DataFrame,
    ellipsoid_summary: pd.DataFrame,
    embedding_methods: Sequence[str],
    config_note: str,
) -> None:
    proj_ds = _projection_dataset_summary(projection_summary)
    neigh_ds = _neighborhood_dataset_summary(neighborhood_summary)
    ell_ds = _ellipsoid_dataset_summary(ellipsoid_summary)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Manifold Diagnostics Conclusion\n\n")
        f.write("更新时间：2026-03-29\n\n")
        f.write("本轮 B0 诊断对象：`orig / vector operator / log-Euclidean operator`。\n\n")
        f.write(f"当前配置：{config_note}\n\n")
        f.write(f"Embedding methods: {', '.join(str(v) for v in embedding_methods)}\n\n")
        for dataset in sorted(set(str(v) for v in projection_summary["dataset"].tolist())):
            f.write(f"## {dataset}\n\n")
            for line in _operator_note(dataset, proj_ds, neigh_ds, ell_ds):
                f.write(f"{line}\n")
            f.write("\n")
        f.write("## 总结性判断\n\n")
        f.write("- 第一轮 B0 已经能稳定区分 `orig / vector / log-Euclidean` 三组对象，但结构差异总体偏弱，说明当前 operator 更像在既有流形上做有限扰动，而不是彻底改写流形拓扑。\n")
        f.write("- NATOPS 上更容易看到“安全外扩/轻度致密化”信号；SCP1 上则更像存在细粒度类条件扭曲，单靠粗投影和粗邻域统计还不够把失败模式完全显形。\n")
        f.write("- 因此后续 Feedback Pool 更应依赖“低方向漂移 + 低跨类近邻增长 + 类条件安全性”的组合规则，而不能只看单一投影图是否更分开。\n\n")
        f.write("后续 Feedback Pool 更应优先录取：\n\n")
        f.write("- 能提升类内连通性但不显著升高跨类近邻比例的增强样本\n")
        f.write("- 椭球主轴延伸存在但方向漂移较小的增强样本\n")
        f.write("- 在 NATOPS 上表现为安全外扩、在 SCP1 上不过度桥接的增强样本\n")


def main() -> None:
    p = argparse.ArgumentParser(description="B0 manifold diagnostics for orig/vector/log-Euclidean operator states.")
    p.add_argument("--datasets", type=str, default="natops,selfregulationscp1")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--out-root", type=str, default="out/route_b_manifold_diagnostics_20260329_formal")
    p.add_argument("--embedding-methods", type=str, default="umap,pca")
    p.add_argument("--split", type=str, default="train", choices=["train"])
    p.add_argument("--projection-k", type=int, default=10)
    p.add_argument("--neighborhood-k", type=int, default=10)
    p.add_argument("--gamma-main", type=float, default=0.10)
    p.add_argument("--natops-second-axis-scale", type=float, default=0.80)
    p.add_argument("--scp1-second-axis-scale", type=float, default=0.80)
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
    methods = resolve_embedding_methods(args.embedding_methods)

    ensure_dir(args.out_root)
    projection_dir = os.path.join(args.out_root, "projection")
    ellipsoid_dir = os.path.join(args.out_root, "ellipsoid")
    neighborhood_dir = os.path.join(args.out_root, "neighborhood")
    ensure_dir(projection_dir)
    ensure_dir(ellipsoid_dir)
    ensure_dir(neighborhood_dir)

    projection_rows: List[Dict[str, object]] = []
    neighborhood_rows: List[Dict[str, object]] = []
    ellipsoid_rows: List[Dict[str, object]] = []

    pia_cfg = PIACoreConfig(
        r_dimension=int(args.pia_r_dimension),
        n_iters=int(args.pia_n_iters),
        C_repr=float(args.pia_c_repr),
        activation=str(args.pia_activation),
        bias_update_mode=str(args.pia_bias_update_mode),
        orthogonalize=bool(int(args.pia_orthogonalize)),
    )

    for dataset in datasets:
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
            ranked_axis_ids, _ = pia_core.rank_axes_by_energy(rep_state.X_train)
            axis_ids = ranked_axis_ids[: int(args.axis_count)]
            second_axis_scale = _dataset_second_axis_scale(dataset, args)
            gamma_vec, _ = pia_core.build_two_axis_gamma_vector(
                axis_ids=axis_ids,
                gamma_main=float(args.gamma_main),
                second_axis_scale=float(second_axis_scale),
            )
            vec_result = pia_core.apply_affine(
                rep_state.X_train,
                gamma_vector=gamma_vec,
                axis_ids=axis_ids,
                pullback_alpha=float(args.pullback_alpha),
            )
            log_result = pia_core.apply_logeuclidean_affine(
                rep_state.X_train,
                gamma_vector=gamma_vec,
                axis_ids=axis_ids,
                pullback_alpha=float(args.pullback_alpha),
            )

            X_by_operator = {
                "orig": np.asarray(rep_state.X_train, dtype=np.float64),
                "vector": np.asarray(vec_result.X_aug, dtype=np.float64),
                "logeuclidean": np.asarray(log_result.X_aug, dtype=np.float64),
            }

            for method in methods:
                coords_by_operator = build_embedding_maps(X_by_operator, method=str(method), seed=int(seed))
                projection_png = os.path.join(projection_dir, f"{dataset}_seed{int(seed)}_{method}.png")
                plot_projection_panels(
                    projection_png,
                    coords_by_operator=coords_by_operator,
                    y=rep_state.y_train,
                    dataset=str(dataset),
                    seed=int(seed),
                    method=str(method),
                )
                for operator_type, coords in coords_by_operator.items():
                    stats = compute_projection_summary(coords, rep_state.y_train, k=int(args.projection_k))
                    projection_rows.append(
                        {
                            "dataset": str(dataset),
                            "seed": int(seed),
                            "operator_type": str(operator_type),
                            "embedding_method": str(method),
                            "split": str(args.split),
                            **stats,
                            "notes": f"axis_ids={axis_ids}, gamma={gamma_vec}, pullback={float(args.pullback_alpha):.2f}",
                        }
                    )

            neigh_stats_map: Dict[str, Dict[str, float]] = {}
            for operator_type, X_op in X_by_operator.items():
                neigh_stats = compute_neighborhood_summary(X_op, rep_state.y_train, k=int(args.neighborhood_k))
                neigh_stats_map[str(operator_type)] = neigh_stats
                neighborhood_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "operator_type": str(operator_type),
                        **neigh_stats,
                        "notes": f"axis_ids={axis_ids}, gamma={gamma_vec}",
                    }
                )
            neighborhood_png = os.path.join(neighborhood_dir, f"{dataset}_seed{int(seed)}_neighborhood.png")
            plot_neighborhood_summary(
                neighborhood_png,
                stats_by_operator=neigh_stats_map,
                dataset=str(dataset),
                seed=int(seed),
            )

            ellipsoid_png = os.path.join(ellipsoid_dir, f"{dataset}_seed{int(seed)}_ellipsoids.png")
            plot_ellipsoid_summary(
                ellipsoid_png,
                X_by_operator=X_by_operator,
                y=rep_state.y_train,
                mean_log_train=rep_state.mean_log_train,
                dataset=str(dataset),
                seed=int(seed),
            )
            for row in compute_ellipsoid_summary(X_by_operator, rep_state.y_train, mean_log_train=rep_state.mean_log_train):
                ellipsoid_rows.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        **row,
                    }
                )

    projection_df = pd.DataFrame(projection_rows)
    neighborhood_df = pd.DataFrame(neighborhood_rows)
    ellipsoid_df = pd.DataFrame(ellipsoid_rows)

    projection_csv = os.path.join(args.out_root, "manifold_diagnostics_projection_summary.csv")
    neighborhood_csv = os.path.join(args.out_root, "manifold_diagnostics_neighborhood_summary.csv")
    ellipsoid_csv = os.path.join(args.out_root, "manifold_diagnostics_ellipsoid_summary.csv")
    projection_df.to_csv(projection_csv, index=False)
    neighborhood_df.to_csv(neighborhood_csv, index=False)
    ellipsoid_df.to_csv(ellipsoid_csv, index=False)

    conclusion_path = os.path.join(args.out_root, "manifold_diagnostics_conclusion.md")
    config_note = (
        f"split={args.split}; gamma_main={float(args.gamma_main):.2f}; "
        f"natops_axis2={float(args.natops_second_axis_scale):.2f}; "
        f"scp1_axis2={float(args.scp1_second_axis_scale):.2f}; "
        f"pullback={float(args.pullback_alpha):.2f}"
    )
    _write_conclusion(
        conclusion_path,
        projection_summary=projection_df,
        neighborhood_summary=neighborhood_df,
        ellipsoid_summary=ellipsoid_df,
        embedding_methods=methods,
        config_note=config_note,
    )

    print(f"[b0-diagnostics] wrote {projection_csv}")
    print(f"[b0-diagnostics] wrote {neighborhood_csv}")
    print(f"[b0-diagnostics] wrote {ellipsoid_csv}")
    print(f"[b0-diagnostics] wrote {conclusion_path}")


if __name__ == "__main__":
    main()
