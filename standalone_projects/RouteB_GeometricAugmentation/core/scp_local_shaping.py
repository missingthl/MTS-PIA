from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


def _window_local_step_mean(z_seq: np.ndarray, idx: int) -> float:
    arr = np.asarray(z_seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError("z_seq must be [K, D] with K>0")
    k = int(arr.shape[0])
    if k <= 1:
        return 0.0
    prev_idx = max(0, int(idx) - 1)
    next_idx = min(k - 1, int(idx) + 1)
    prev_step = float(np.linalg.norm(arr[int(idx)] - arr[int(prev_idx)]))
    next_step = float(np.linalg.norm(arr[int(next_idx)] - arr[int(idx)]))
    return float(0.5 * (prev_step + next_step))


def _fit_classwise_kmeans_prototypes(
    rows: Sequence[Dict[str, object]],
    *,
    prototype_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise ImportError("SCP local shaping requires scikit-learn in the active environment.") from e

    x = np.stack([np.asarray(r["z_window"], dtype=np.float64) for r in rows], axis=0).astype(np.float64)
    k = int(max(1, min(int(prototype_count), int(x.shape[0]))))
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
    labels = km.fit_predict(x)
    return np.asarray(km.cluster_centers_, dtype=np.float64), np.asarray(labels, dtype=np.int64)


@dataclass(frozen=True)
class SCPLocalShapingConfig:
    prototype_count: int = 4
    anchors_per_prototype: int = 16
    anchor_selection_mode: str = "nearest"  # nearest | tight_margin
    same_dist_quantile: float = 50.0
    beta: float = 0.5
    epsilon_scale: float = 0.10
    seed: int | None = None


@dataclass
class SCPLocalShapingResult:
    shaped_train_z_seq_list: List[np.ndarray]
    anchor_rows: List[Dict[str, object]]
    shaping_rows: List[Dict[str, object]]
    summary: Dict[str, object] = field(default_factory=dict)


def apply_scp_local_shaping(
    *,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    cfg: SCPLocalShapingConfig,
) -> SCPLocalShapingResult:
    tids = [str(v) for v in train_tids]
    labels = [int(v) for v in train_labels]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    if len(tids) != len(labels) or len(tids) != len(seqs):
        raise ValueError("train inputs must align")
    if not seqs:
        raise ValueError("train_z_seq_list cannot be empty")
    anchor_selection_mode = str(cfg.anchor_selection_mode).strip().lower()
    if anchor_selection_mode not in {"nearest", "tight_margin"}:
        raise ValueError("anchor_selection_mode must be one of: nearest, tight_margin")

    all_rows: List[Dict[str, object]] = []
    rows_by_class: Dict[int, List[Dict[str, object]]] = {}
    for tid, cls, seq in zip(tids, labels, seqs):
        for window_index in range(int(seq.shape[0])):
            row = {
                "trial_id": str(tid),
                "label": int(cls),
                "window_index": int(window_index),
                "z_window": np.asarray(seq[int(window_index)], dtype=np.float32),
            }
            all_rows.append(row)
            rows_by_class.setdefault(int(cls), []).append(row)

    prototype_rows: Dict[int, np.ndarray] = {}
    assignment_rows: Dict[int, np.ndarray] = {}
    for cls, rows in rows_by_class.items():
        reps, assign = _fit_classwise_kmeans_prototypes(
            rows,
            prototype_count=int(cfg.prototype_count),
            seed=int(0 if cfg.seed is None else cfg.seed) + int(cls),
        )
        prototype_rows[int(cls)] = np.asarray(reps, dtype=np.float64)
        assignment_rows[int(cls)] = np.asarray(assign, dtype=np.int64)

    shaped_seqs = [np.asarray(seq, dtype=np.float32).copy() for seq in seqs]
    seq_index_by_tid = {str(tid): idx for idx, tid in enumerate(tids)}
    all_prototypes: List[tuple[int, int, np.ndarray]] = []
    for cls, reps in prototype_rows.items():
        for proto_id, rep in enumerate(np.asarray(reps, dtype=np.float64)):
            all_prototypes.append((int(cls), int(proto_id), np.asarray(rep, dtype=np.float64)))

    anchor_rows: List[Dict[str, object]] = []
    shaping_rows: List[Dict[str, object]] = []

    for cls, rows in rows_by_class.items():
        reps = np.asarray(prototype_rows[int(cls)], dtype=np.float64)
        assign = np.asarray(assignment_rows[int(cls)], dtype=np.int64)
        if reps.ndim != 2 or reps.shape[0] <= 0:
            continue

        for proto_id in range(int(reps.shape[0])):
            member_ids = np.where(assign == int(proto_id))[0]
            if member_ids.size <= 0:
                continue
            rep = np.asarray(reps[int(proto_id)], dtype=np.float64)
            member_rows = [rows[int(i)] for i in member_ids.tolist()]
            member_dists = np.asarray(
                [float(np.linalg.norm(np.asarray(r["z_window"], dtype=np.float64) - rep)) for r in member_rows],
                dtype=np.float64,
            )
            if anchor_selection_mode == "nearest":
                order = np.argsort(member_dists, kind="mergesort")
                keep = order[: int(max(1, min(int(cfg.anchors_per_prototype), int(order.size))))]
            else:
                same_dist_threshold = float(np.percentile(member_dists, float(cfg.same_dist_quantile)))
                eligible = np.where(member_dists <= same_dist_threshold)[0]
                if eligible.size <= 0:
                    eligible = np.arange(member_dists.size, dtype=np.int64)
                opp_margin_vals = []
                for idx in eligible.tolist():
                    z = np.asarray(member_rows[int(idx)]["z_window"], dtype=np.float64)
                    opp_rows = [
                        (other_cls, other_proto_id, other_rep)
                        for other_cls, other_proto_id, other_rep in all_prototypes
                        if int(other_cls) != int(cls)
                    ]
                    if not opp_rows:
                        opp_margin_vals.append((int(idx), float("inf"), float(member_dists[int(idx)])))
                        continue
                    opp_dists = np.asarray(
                        [float(np.linalg.norm(z - np.asarray(other_rep, dtype=np.float64))) for _c, _pid, other_rep in opp_rows],
                        dtype=np.float64,
                    )
                    opp_dist = float(np.min(opp_dists)) if opp_dists.size else float("inf")
                    margin_val = float(opp_dist - float(member_dists[int(idx)]))
                    opp_margin_vals.append((int(idx), float(margin_val), float(member_dists[int(idx)])))
                opp_margin_vals = sorted(opp_margin_vals, key=lambda x: (x[1], x[2], x[0]))
                keep_ids = [int(row[0]) for row in opp_margin_vals[: int(max(1, min(int(cfg.anchors_per_prototype), len(opp_margin_vals))))]]
                keep = np.asarray(keep_ids, dtype=np.int64)
            admitted_rows = [member_rows[int(i)] for i in keep.tolist()]
            prototype_member_count = int(member_ids.size)
            admitted_anchor_count = int(len(admitted_rows))
            anchor_coverage_ratio = float(admitted_anchor_count / max(1, prototype_member_count))

            for row in admitted_rows:
                z = np.asarray(row["z_window"], dtype=np.float64)
                trial_id = str(row["trial_id"])
                trial_pos = int(seq_index_by_tid[trial_id])
                window_index = int(row["window_index"])

                opp_rows = [
                    (other_cls, other_proto_id, other_rep)
                    for other_cls, other_proto_id, other_rep in all_prototypes
                    if int(other_cls) != int(cls)
                ]
                if not opp_rows:
                    continue
                opp_dists = np.asarray(
                    [float(np.linalg.norm(z - np.asarray(other_rep, dtype=np.float64))) for _c, _pid, other_rep in opp_rows],
                    dtype=np.float64,
                )
                best_opp_idx = int(np.argmin(opp_dists))
                opp_cls, opp_proto_id, p_opp = opp_rows[best_opp_idx]
                p_same = np.asarray(rep, dtype=np.float64)
                same_dist = float(np.linalg.norm(z - p_same))
                opp_dist = float(np.linalg.norm(z - np.asarray(p_opp, dtype=np.float64)))

                direction_raw = (p_same - z) + float(cfg.beta) * (z - np.asarray(p_opp, dtype=np.float64))
                direction_norm = float(np.linalg.norm(direction_raw))
                if not np.isfinite(direction_norm) or direction_norm <= 1e-12:
                    continue
                direction = np.asarray(direction_raw / direction_norm, dtype=np.float64)
                epsilon_local = float(float(cfg.epsilon_scale) * min(same_dist, opp_dist))
                z_shaped = np.asarray(z + epsilon_local * direction, dtype=np.float32)

                orig_local_step = float(_window_local_step_mean(seqs[int(trial_pos)], int(window_index)))
                shaped_seqs[int(trial_pos)][int(window_index)] = z_shaped
                shaped_local_step = float(_window_local_step_mean(shaped_seqs[int(trial_pos)], int(window_index)))
                step_distortion_ratio = float(shaped_local_step / max(1e-6, orig_local_step))

                margin_before = float(opp_dist - same_dist)
                margin_after = float(
                    np.linalg.norm(np.asarray(z_shaped, dtype=np.float64) - np.asarray(p_opp, dtype=np.float64))
                    - np.linalg.norm(np.asarray(z_shaped, dtype=np.float64) - p_same)
                )

                anchor_rows.append(
                    {
                        "class_id": int(cls),
                        "prototype_id": int(proto_id),
                        "trial_id": str(trial_id),
                        "window_index": int(window_index),
                        "prototype_member_count": int(prototype_member_count),
                        "admitted_anchor_count": int(admitted_anchor_count),
                        "anchor_coverage_ratio": float(anchor_coverage_ratio),
                        "anchor_selection_mode": str(anchor_selection_mode),
                        "admitted_margin_before": float(margin_before),
                        "admitted_same_dist_before": float(same_dist),
                    }
                )
                shaping_rows.append(
                    {
                        "class_id": int(cls),
                        "prototype_id": int(proto_id),
                        "trial_id": str(trial_id),
                        "window_index": int(window_index),
                        "opp_class_id": int(opp_cls),
                        "opp_prototype_id": int(opp_proto_id),
                        "same_dist": float(same_dist),
                        "opp_dist": float(opp_dist),
                        "epsilon_local": float(epsilon_local),
                        "margin_before": float(margin_before),
                        "margin_after": float(margin_after),
                        "margin_gain": float(margin_after - margin_before),
                        "orig_local_step": float(orig_local_step),
                        "shaped_local_step": float(shaped_local_step),
                        "local_step_distortion_ratio": float(step_distortion_ratio),
                    }
                )

    eps_vals = np.asarray([float(r["epsilon_local"]) for r in shaping_rows], dtype=np.float64)
    distort_vals = np.asarray([float(r["local_step_distortion_ratio"]) for r in shaping_rows], dtype=np.float64)
    margin_gain_vals = np.asarray([float(r["margin_gain"]) for r in shaping_rows], dtype=np.float64)
    summary = {
        "shaped_window_count": int(len(shaping_rows)),
        "shaped_window_ratio": float(len(shaping_rows) / max(1, len(all_rows))),
        "epsilon_local_mean": float(np.mean(eps_vals)) if eps_vals.size else 0.0,
        "epsilon_local_p95": float(np.percentile(eps_vals, 95.0)) if eps_vals.size else 0.0,
        "local_step_distortion_ratio_mean": float(np.mean(distort_vals)) if distort_vals.size else 1.0,
        "local_step_distortion_ratio_p95": float(np.percentile(distort_vals, 95.0)) if distort_vals.size else 1.0,
        "margin_gain_mean": float(np.mean(margin_gain_vals)) if margin_gain_vals.size else 0.0,
        "admitted_margin_mean_before": float(np.mean([float(r["admitted_margin_before"]) for r in anchor_rows]))
        if anchor_rows
        else 0.0,
        "admitted_same_dist_mean_before": float(np.mean([float(r["admitted_same_dist_before"]) for r in anchor_rows]))
        if anchor_rows
        else 0.0,
        "beta": float(cfg.beta),
        "epsilon_scale": float(cfg.epsilon_scale),
        "anchors_per_prototype": int(cfg.anchors_per_prototype),
        "anchor_selection_mode": str(anchor_selection_mode),
        "same_dist_quantile": float(cfg.same_dist_quantile),
        "scope_note": (
            "SCP-Branch v1 performs train-only local margin shaping on admitted prototype-member windows. "
            "Validation and test trajectories remain original; there is no replay, curriculum, neighborhood propagation, "
            "or test-time routing."
        ),
    }
    return SCPLocalShapingResult(
        shaped_train_z_seq_list=shaped_seqs,
        anchor_rows=anchor_rows,
        shaping_rows=shaping_rows,
        summary=summary,
    )
