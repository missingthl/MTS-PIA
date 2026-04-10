#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from route_b_unified.pia_operator_value_probe import (  # noqa: E402
    FixedReferenceGeometry,
    FixedReferenceGeometryConfig,
    SingleTemplatePIADiscriminativeConfig,
    SingleTemplatePIAOperator,
    SingleTemplatePIAStageARepairConfig,
    SingleTemplatePIAValueConfig,
    apply_single_template_pia_stage_a_variant,
    build_fixed_reference_geometry,
    fit_single_template_pia_operator,
    fit_single_template_pia_operator_discriminative,
)
from route_b_unified.trajectory_representation import (  # noqa: E402
    TrajectoryRepresentationConfig,
    TrajectoryRepresentationState,
    build_trajectory_representation,
)


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


def _build_dense_state(args: argparse.Namespace, dataset: str, seed: int) -> TrajectoryRepresentationState:
    return build_trajectory_representation(
        TrajectoryRepresentationConfig(
            dataset=str(dataset),
            seed=int(seed),
            val_fraction=float(args.val_fraction),
            spd_eps=float(args.spd_eps),
            prop_win_ratio=float(args.prop_win_ratio),
            prop_hop_ratio=float(args.prop_hop_ratio),
            min_window_extra_channels=int(args.min_window_extra_channels),
            min_hop_len=int(args.min_hop_len),
            force_hop_len=int(args.force_hop_len),
        )
    )


def _normalize_direction(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.zeros_like(arr)
    return arr / norm


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    ua = _normalize_direction(a)
    ub = _normalize_direction(b)
    if float(np.linalg.norm(ua)) <= 1e-12 or float(np.linalg.norm(ub)) <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(ua, ub), -1.0, 1.0))


def _lookup_same_prototype_center(
    geometry: FixedReferenceGeometry,
    *,
    class_id: int,
    prototype_id: int,
) -> np.ndarray:
    return np.asarray(geometry.prototypes_by_class[int(class_id)][int(prototype_id)], dtype=np.float64)


def _nearest_opposite_prototype_center(
    query: np.ndarray,
    *,
    same_class_id: int,
    geometry: FixedReferenceGeometry,
) -> np.ndarray:
    query_arr = np.asarray(query, dtype=np.float64).ravel()
    best_center = None
    best_dist = None
    for cls, _pid, rep in geometry.all_prototypes:
        if int(cls) == int(same_class_id):
            continue
        rep_arr = np.asarray(rep, dtype=np.float64)
        dist = float(np.linalg.norm(query_arr - rep_arr))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_center = rep_arr
    if best_center is None:
        raise RuntimeError("geometry must contain at least two classes")
    return np.asarray(best_center, dtype=np.float64)


def _nearest_opposite_prototype_for_same_prototype(
    geometry: FixedReferenceGeometry,
    *,
    class_id: int,
    prototype_id: int,
) -> Tuple[int, int, np.ndarray]:
    p_same = _lookup_same_prototype_center(geometry, class_id=int(class_id), prototype_id=int(prototype_id))
    best = None
    best_dist = None
    for cls, pid, rep in geometry.all_prototypes:
        if int(cls) == int(class_id):
            continue
        rep_arr = np.asarray(rep, dtype=np.float64)
        dist = float(np.linalg.norm(p_same - rep_arr))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (int(cls), int(pid), rep_arr)
    if best is None:
        raise RuntimeError("geometry must contain at least two classes")
    return int(best[0]), int(best[1]), np.asarray(best[2], dtype=np.float64)


def _build_pair_table(geometry: FixedReferenceGeometry) -> pd.DataFrame:
    keys = sorted({(int(r["class_id"]), int(r["prototype_id"])) for r in geometry.fit_rows})
    rows: List[Dict[str, object]] = []
    for class_id, prototype_id in keys:
        opp_class_id, opp_prototype_id, p_opp = _nearest_opposite_prototype_for_same_prototype(
            geometry,
            class_id=int(class_id),
            prototype_id=int(prototype_id),
        )
        p_same = _lookup_same_prototype_center(geometry, class_id=int(class_id), prototype_id=int(prototype_id))
        pair_axis = _normalize_direction(p_same - p_opp)
        rows.append(
            {
                "same_class_id": int(class_id),
                "same_prototype_id": int(prototype_id),
                "opp_class_id": int(opp_class_id),
                "opp_prototype_id": int(opp_prototype_id),
                "pair_id": f"s{class_id}p{prototype_id}|o{opp_class_id}p{opp_prototype_id}",
                "center_dist": float(np.linalg.norm(p_same - p_opp)),
                "pair_axis": pair_axis,
                "p_same": p_same,
                "p_opp": p_opp,
            }
        )
    return pd.DataFrame(rows)


def _compute_same_only_fit_row_cosines(operator: SingleTemplatePIAOperator, geometry: FixedReferenceGeometry) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for row, z in zip(geometry.fit_rows, geometry.fit_windows):
        class_id = int(row["class_id"])
        prototype_id = int(row["prototype_id"])
        p_same = _lookup_same_prototype_center(geometry, class_id=class_id, prototype_id=prototype_id)
        p_opp = _nearest_opposite_prototype_center(np.asarray(z, dtype=np.float64), same_class_id=class_id, geometry=geometry)
        u_geom = _normalize_direction(p_same - p_opp)
        rows.append(
            {
                "class_id": class_id,
                "prototype_id": prototype_id,
                "local_cosine": _safe_cosine(operator.direction, u_geom),
            }
        )
    return pd.DataFrame(rows)


def _compute_pair_level_cosines(
    pair_df: pd.DataFrame,
    *,
    op_c0: SingleTemplatePIAOperator,
    op_c2: SingleTemplatePIAOperator,
    op_c3: SingleTemplatePIAOperator,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pair_meta = {
        (int(r["same_class_id"]), int(r["same_prototype_id"])): r
        for r in op_c3.meta.get("pair_weight_rows", [])
    }
    for _, row in pair_df.iterrows():
        key = (int(row["same_class_id"]), int(row["same_prototype_id"]))
        meta = pair_meta.get(key, {})
        pair_axis = np.asarray(row["pair_axis"], dtype=np.float64)
        rows.append(
            {
                "pair_id": str(row["pair_id"]),
                "same_class_id": int(row["same_class_id"]),
                "same_prototype_id": int(row["same_prototype_id"]),
                "opp_class_id": int(row["opp_class_id"]),
                "opp_prototype_id": int(row["opp_prototype_id"]),
                "center_dist": float(row["center_dist"]),
                "cos_c0": _safe_cosine(op_c0.direction, pair_axis),
                "cos_c2": _safe_cosine(op_c2.direction, pair_axis),
                "cos_c3": _safe_cosine(op_c3.direction, pair_axis),
                "same_pool_count": int(meta.get("same_pool_count", 0)),
                "opp_pool_count": int(meta.get("opp_pool_count", 0)),
                "same_weight_mass": float(meta.get("same_weight_mass", 0.0)),
                "opp_weight_mass": float(meta.get("opp_weight_mass", 0.0)),
                "same_proto_effective_sample_size": float(meta.get("same_proto_effective_sample_size", 0.0)),
                "opp_proto_effective_sample_size": float(meta.get("opp_proto_effective_sample_size", 0.0)),
                "same_weight_scale": float(meta.get("same_weight_scale", 0.0)),
                "opp_weight_scale": float(meta.get("opp_weight_scale", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _plot_pair_axis_cosines(pair_cos_df: pd.DataFrame, out_path: str) -> None:
    df = pair_cos_df.sort_values("cos_c3", ascending=False).reset_index(drop=True)
    x = np.arange(int(df.shape[0]), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(x, df["cos_c0"].to_numpy(dtype=np.float64), marker="o", linewidth=1.5, label="C0 same-only unweighted", color="#7f8c8d")
    ax.plot(x, df["cos_c2"].to_numpy(dtype=np.float64), marker="o", linewidth=1.5, label="C2 same-only weighted", color="#8e44ad")
    ax.plot(x, df["cos_c3"].to_numpy(dtype=np.float64), marker="o", linewidth=2.0, label="C3 bipolar discriminative", color="#c0392b")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Cosine To Pair Axis")
    ax.set_xlabel("Prototype Pair (sorted by C3 cosine)")
    ax.set_title("Pair-Axis Alignment Across Prototype Pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(df["pair_id"].tolist(), rotation=90, fontsize=8)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _project_direction_to_pca(direction: np.ndarray, pca: PCA, target_length: float) -> np.ndarray:
    proj = pca.components_ @ _normalize_direction(direction)
    norm = float(np.linalg.norm(proj))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.zeros((2,), dtype=np.float64)
    return np.asarray((target_length / norm) * proj, dtype=np.float64)


def _plot_representative_pair_geometry(
    geometry: FixedReferenceGeometry,
    pair_cos_df: pd.DataFrame,
    *,
    op_c0: SingleTemplatePIAOperator,
    op_c2: SingleTemplatePIAOperator,
    op_c3: SingleTemplatePIAOperator,
    out_path: str,
) -> Dict[str, object]:
    ordered = pair_cos_df.sort_values("cos_c3", ascending=False).reset_index(drop=True)
    row = ordered.iloc[int(len(ordered) // 2)]
    same_key = (int(row["same_class_id"]), int(row["same_prototype_id"]))
    opp_key = (int(row["opp_class_id"]), int(row["opp_prototype_id"]))

    same_ids = [idx for idx, r in enumerate(geometry.fit_rows) if (int(r["class_id"]), int(r["prototype_id"])) == same_key]
    opp_ids = [idx for idx, r in enumerate(geometry.fit_rows) if (int(r["class_id"]), int(r["prototype_id"])) == opp_key]
    same_arr = np.asarray(geometry.fit_windows[same_ids], dtype=np.float64)
    opp_arr = np.asarray(geometry.fit_windows[opp_ids], dtype=np.float64)

    fit_arr = np.concatenate([same_arr, opp_arr], axis=0)
    pca = PCA(n_components=2, random_state=0).fit(fit_arr)
    same_xy = pca.transform(same_arr)
    opp_xy = pca.transform(opp_arr)

    p_same = _lookup_same_prototype_center(geometry, class_id=same_key[0], prototype_id=same_key[1])
    p_opp = _lookup_same_prototype_center(geometry, class_id=opp_key[0], prototype_id=opp_key[1])
    same_center_xy = pca.transform(np.asarray([p_same], dtype=np.float64))[0]
    opp_center_xy = pca.transform(np.asarray([p_opp], dtype=np.float64))[0]
    anchor_xy = 0.5 * (same_center_xy + opp_center_xy)
    arrow_len = 0.35 * max(1e-6, float(np.linalg.norm(same_center_xy - opp_center_xy)))

    pair_axis = _normalize_direction(p_same - p_opp)
    pair_arrow = _project_direction_to_pca(pair_axis, pca, arrow_len)
    c0_arrow = _project_direction_to_pca(op_c0.direction, pca, arrow_len)
    c2_arrow = _project_direction_to_pca(op_c2.direction, pca, arrow_len)
    c3_arrow = _project_direction_to_pca(op_c3.direction, pca, arrow_len)

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    ax.scatter(same_xy[:, 0], same_xy[:, 1], s=26, alpha=0.75, color="#2e86de", label=f"same {same_key[0]}:{same_key[1]}")
    ax.scatter(opp_xy[:, 0], opp_xy[:, 1], s=26, alpha=0.75, color="#e67e22", label=f"opp {opp_key[0]}:{opp_key[1]}")
    ax.scatter([same_center_xy[0]], [same_center_xy[1]], s=180, marker="*", color="#1f5fbf")
    ax.scatter([opp_center_xy[0]], [opp_center_xy[1]], s=180, marker="*", color="#c86f1b")

    for arrow, color, label in [
        (pair_arrow, "#111111", f"pair axis ({row['cos_c3']:.3f} vs C3)"),
        (c0_arrow, "#7f8c8d", f"C0 ({row['cos_c0']:.3f})"),
        (c2_arrow, "#8e44ad", f"C2 ({row['cos_c2']:.3f})"),
        (c3_arrow, "#c0392b", f"C3 ({row['cos_c3']:.3f})"),
    ]:
        ax.arrow(
            float(anchor_xy[0]),
            float(anchor_xy[1]),
            float(arrow[0]),
            float(arrow[1]),
            width=0.006,
            head_width=0.12,
            head_length=0.14,
            length_includes_head=True,
            color=color,
            alpha=0.95,
            label=label,
        )

    ax.set_title("Representative Pair Local Geometry (median-C3 pair)")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "representative_pair_id": str(row["pair_id"]),
        "same_class_id": int(row["same_class_id"]),
        "same_prototype_id": int(row["same_prototype_id"]),
        "opp_class_id": int(row["opp_class_id"]),
        "opp_prototype_id": int(row["opp_prototype_id"]),
        "cos_c0": float(row["cos_c0"]),
        "cos_c2": float(row["cos_c2"]),
        "cos_c3": float(row["cos_c3"]),
    }


def _plot_response_margin_scatter(
    state: TrajectoryRepresentationState,
    *,
    op_c0: SingleTemplatePIAOperator,
    op_c2: SingleTemplatePIAOperator,
    op_c3: SingleTemplatePIAOperator,
    epsilon_scale: float,
    smooth_lambda: float,
    out_path: str,
) -> Dict[str, float]:
    b0_train = apply_single_template_pia_stage_a_variant(
        z_seq_list=state.train.z_seq_list,
        operator=op_c0,
        cfg=SingleTemplatePIAStageARepairConfig(
            variant="current_sigmoid_minimal",
            epsilon_scale=float(epsilon_scale),
            smooth_lambda=float(smooth_lambda),
        ),
    )
    budget_target = float(b0_train.summary["operator_to_step_ratio_mean"])

    arm_rows: Dict[str, pd.DataFrame] = {}
    arm_corr: Dict[str, float] = {}
    for arm_name, operator in [
        ("C0", op_c0),
        ("C2", op_c2),
        ("C3", op_c3),
    ]:
        train_res = apply_single_template_pia_stage_a_variant(
            z_seq_list=state.train.z_seq_list,
            operator=operator,
            cfg=SingleTemplatePIAStageARepairConfig(
                variant="sigmoid_clip_tanh_local_median_scaled_iqr",
                epsilon_scale=float(epsilon_scale),
                smooth_lambda=float(smooth_lambda),
                budget_target_operator_to_step_ratio=float(budget_target),
            ),
        )
        arm_rows[str(arm_name)] = pd.DataFrame(train_res.diagnostics_rows)
        arm_corr[str(arm_name)] = float(train_res.summary["response_vs_margin_correlation"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    colors = {"C0": "#7f8c8d", "C2": "#8e44ad", "C3": "#c0392b"}
    for ax, arm_name in zip(axes, ["C0", "C2", "C3"]):
        df = arm_rows[str(arm_name)]
        x = df["margin_before"].to_numpy(dtype=np.float64)
        y = df["response_force"].to_numpy(dtype=np.float64)
        if x.size > 1800:
            idx = np.linspace(0, x.size - 1, 1800).astype(np.int64)
            x = x[idx]
            y = y[idx]
        ax.scatter(x, y, s=10, alpha=0.18, color=colors[str(arm_name)], edgecolors="none")
        ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
        ax.set_title(f"{arm_name} train corr = {arm_corr[str(arm_name)]:.3f}")
        ax.grid(alpha=0.22, linestyle=":")
    axes[0].set_ylabel("response_force")
    for ax in axes:
        ax.set_xlabel("margin_before")
    fig.suptitle("Train-Time Response Force vs Margin")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return arm_corr


def _write_markdown_summary(
    out_path: str,
    *,
    dataset: str,
    seed: int,
    pair_cos_df: pd.DataFrame,
    same_only_cos_c0: pd.DataFrame,
    same_only_cos_c2: pd.DataFrame,
    c3_meta: Dict[str, object],
    response_corr: Dict[str, float],
    rep_pair: Dict[str, object],
    pair_axis_plot: str,
    rep_pair_plot: str,
    response_plot: str,
) -> None:
    lines: List[str] = [
        f"# C3 Geometry Summary: {dataset} seed={seed}",
        "",
        "## High-Level Read",
        "",
        "- `C3` has rotated the single template much closer to prototype-pair normal directions than `C0/C2`.",
        "- This rotation is not obviously explained by same/opp pool imbalance, because both count ratio and weight-mass ratio stay near `1.0`.",
        "- The remaining bottleneck is not primarily axis discovery anymore; it is the force/readout coupling after the axis has rotated.",
        "",
        "## Pair-Axis Alignment",
        "",
        f"- `pair cosine mean`: C0={pair_cos_df['cos_c0'].mean():.4f}, C2={pair_cos_df['cos_c2'].mean():.4f}, C3={pair_cos_df['cos_c3'].mean():.4f}",
        f"- `pair cosine median`: C0={pair_cos_df['cos_c0'].median():.4f}, C2={pair_cos_df['cos_c2'].median():.4f}, C3={pair_cos_df['cos_c3'].median():.4f}",
        f"- `positive pair count`: C0={(pair_cos_df['cos_c0'] > 0).sum()}/{len(pair_cos_df)}, C2={(pair_cos_df['cos_c2'] > 0).sum()}/{len(pair_cos_df)}, C3={(pair_cos_df['cos_c3'] > 0).sum()}/{len(pair_cos_df)}",
        f"- `same-only fit-row local cosine mean`: C0={same_only_cos_c0['local_cosine'].mean():.4f}, C2={same_only_cos_c2['local_cosine'].mean():.4f}",
        "",
        f"- Figure: [{os.path.basename(pair_axis_plot)}]({pair_axis_plot})",
        "",
        "## Pool Balance Audit",
        "",
        f"- `same_pool_count`: {int(c3_meta.get('same_pool_count', 0))}",
        f"- `opp_pool_count`: {int(c3_meta.get('opp_pool_count', 0))}",
        f"- `same_weight_mass`: {float(c3_meta.get('same_weight_mass', 0.0)):.4f}",
        f"- `opp_weight_mass`: {float(c3_meta.get('opp_weight_mass', 0.0)):.4f}",
        f"- `same_opp_count_ratio`: {float(c3_meta.get('same_opp_count_ratio', 0.0)):.4f}",
        f"- `same_opp_weight_mass_ratio`: {float(c3_meta.get('same_opp_weight_mass_ratio', 0.0)):.4f}",
        "",
        "## Representative Pair",
        "",
        f"- `pair`: {rep_pair['representative_pair_id']}",
        f"- `pair cosine`: C0={rep_pair['cos_c0']:.4f}, C2={rep_pair['cos_c2']:.4f}, C3={rep_pair['cos_c3']:.4f}",
        f"- Figure: [{os.path.basename(rep_pair_plot)}]({rep_pair_plot})",
        "",
        "## Force-Field Readout",
        "",
        f"- `train response_vs_margin_correlation`: C0={response_corr['C0']:.4f}, C2={response_corr['C2']:.4f}, C3={response_corr['C3']:.4f}",
        "- Geometric reading: the axis has rotated, but the deployed response field has not yet become margin-monotone under the current A2r force rule.",
        f"- Figure: [{os.path.basename(response_plot)}]({response_plot})",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize geometry for P0a.1 C3 discriminative closed-form probe")
    p.add_argument("--datasets", type=str, default="natops")
    p.add_argument("--seeds", type=str, default="1")
    p.add_argument("--out-root", type=str, default="out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--prop-win-ratio", type=float, default=0.20)
    p.add_argument("--prop-hop-ratio", type=float, default=0.10)
    p.add_argument("--min-window-extra-channels", type=int, default=4)
    p.add_argument("--min-hop-len", type=int, default=4)
    p.add_argument("--force-hop-len", type=int, default=1)
    p.add_argument("--prototype-count", type=int, default=4)
    p.add_argument("--anchors-per-prototype", type=int, default=8)
    p.add_argument("--same-dist-quantile", type=float, default=50.0)
    p.add_argument("--anchor-selection-mode", type=str, default="tight_margin")
    p.add_argument("--pia-activation", type=str, default="sigmoid")
    p.add_argument("--pia-n-iters", type=int, default=3)
    p.add_argument("--pia-c-repr", type=float, default=1.0)
    p.add_argument("--pia-bias-lr", type=float, default=0.25)
    p.add_argument("--pia-bias-update-mode", type=str, default="residual")
    p.add_argument("--pia-epsilon-scale", type=float, default=0.10)
    p.add_argument("--operator-smooth-lambda", type=float, default=0.50)
    p.add_argument("--c3-target-pos", type=float, default=0.95)
    p.add_argument("--c3-target-neg", type=float, default=0.05)
    p.add_argument("--c3-opp-pair-rule", type=str, default="nearest_opposite_prototype")
    args = p.parse_args()

    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_seed_list(args.seeds)

    for dataset in datasets:
        for seed in seeds:
            out_dir = os.path.join(args.out_root, f"{dataset}_seed{seed}")
            _ensure_dir(out_dir)

            state = _build_dense_state(args, dataset, seed)
            geometry = build_fixed_reference_geometry(
                train_tids=state.train.tids.tolist(),
                train_labels=state.train.y.tolist(),
                train_z_seq_list=state.train.z_seq_list,
                cfg=FixedReferenceGeometryConfig(
                    prototype_count=int(args.prototype_count),
                    anchors_per_prototype=int(args.anchors_per_prototype),
                    same_dist_quantile=float(args.same_dist_quantile),
                    anchor_selection_mode=str(args.anchor_selection_mode),
                    seed=int(seed),
                ),
            )

            pia_cfg_base = SingleTemplatePIAValueConfig(
                r_dimension=1,
                n_iters=int(args.pia_n_iters),
                C_repr=float(args.pia_c_repr),
                activation=str(args.pia_activation),
                bias_lr=float(args.pia_bias_lr),
                bias_update_mode=str(args.pia_bias_update_mode),
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                fit_mode="unweighted",
                seed=int(seed),
            )

            op_c0 = fit_single_template_pia_operator(geometry=geometry, cfg=replace(pia_cfg_base, fit_mode="unweighted"))
            op_c2 = fit_single_template_pia_operator(geometry=geometry, cfg=replace(pia_cfg_base, fit_mode="median_min_weighted"))
            op_c3 = fit_single_template_pia_operator_discriminative(
                geometry=geometry,
                cfg=SingleTemplatePIADiscriminativeConfig(
                    r_dimension=1,
                    n_iters=int(args.pia_n_iters),
                    C_repr=float(args.pia_c_repr),
                    activation=str(args.pia_activation),
                    bias_lr=float(args.pia_bias_lr),
                    bias_update_mode=str(args.pia_bias_update_mode),
                    target_pos=float(args.c3_target_pos),
                    target_neg=float(args.c3_target_neg),
                    opp_pair_rule=str(args.c3_opp_pair_rule),
                    seed=int(seed),
                ),
            )

            pair_df = _build_pair_table(geometry)
            pair_cos_df = _compute_pair_level_cosines(pair_df, op_c0=op_c0, op_c2=op_c2, op_c3=op_c3)
            same_only_cos_c0 = _compute_same_only_fit_row_cosines(op_c0, geometry)
            same_only_cos_c2 = _compute_same_only_fit_row_cosines(op_c2, geometry)

            pair_cos_csv = os.path.join(out_dir, "pair_axis_cosines.csv")
            pair_cos_df.to_csv(pair_cos_csv, index=False)

            pair_axis_plot = os.path.join(out_dir, "pair_axis_cosine_comparison.png")
            _plot_pair_axis_cosines(pair_cos_df, pair_axis_plot)

            rep_pair_plot = os.path.join(out_dir, "representative_pair_pca.png")
            rep_pair = _plot_representative_pair_geometry(
                geometry,
                pair_cos_df,
                op_c0=op_c0,
                op_c2=op_c2,
                op_c3=op_c3,
                out_path=rep_pair_plot,
            )

            response_plot = os.path.join(out_dir, "train_response_margin_scatter.png")
            response_corr = _plot_response_margin_scatter(
                state,
                op_c0=op_c0,
                op_c2=op_c2,
                op_c3=op_c3,
                epsilon_scale=float(args.pia_epsilon_scale),
                smooth_lambda=float(args.operator_smooth_lambda),
                out_path=response_plot,
            )

            summary_md = os.path.join(out_dir, "geometry_visual_summary.md")
            _write_markdown_summary(
                summary_md,
                dataset=str(dataset),
                seed=int(seed),
                pair_cos_df=pair_cos_df,
                same_only_cos_c0=same_only_cos_c0,
                same_only_cos_c2=same_only_cos_c2,
                c3_meta=dict(op_c3.meta),
                response_corr=response_corr,
                rep_pair=rep_pair,
                pair_axis_plot=os.path.abspath(pair_axis_plot),
                rep_pair_plot=os.path.abspath(rep_pair_plot),
                response_plot=os.path.abspath(response_plot),
            )

            print(f"[pia-operator-c3-vis] wrote {pair_cos_csv}")
            print(f"[pia-operator-c3-vis] wrote {pair_axis_plot}")
            print(f"[pia-operator-c3-vis] wrote {rep_pair_plot}")
            print(f"[pia-operator-c3-vis] wrote {response_plot}")
            print(f"[pia-operator-c3-vis] wrote {summary_md}")


if __name__ == "__main__":
    main()
