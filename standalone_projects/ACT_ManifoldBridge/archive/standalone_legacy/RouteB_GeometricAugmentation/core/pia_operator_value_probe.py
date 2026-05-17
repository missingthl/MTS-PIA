from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from PIA.telm2 import TELM2Artifacts, TELM2Config, TELM2Transformer


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


def _normalize_direction(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size <= 0:
        raise ValueError("direction vector cannot be empty")
    nrm = float(np.linalg.norm(arr))
    if not np.isfinite(nrm) or nrm <= 1e-12:
        out = np.zeros_like(arr, dtype=np.float64)
        out[0] = 1.0
        return out
    return np.asarray(arr / nrm, dtype=np.float64)


def _smooth_delta(delta: np.ndarray, smooth_lambda: float) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("delta must be [K, D]")
    lam = float(smooth_lambda)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("smooth_lambda must be in [0, 1]")
    if arr.shape[0] <= 1 or lam <= 0.0:
        return np.asarray(arr, dtype=np.float64)

    padded = np.pad(arr, ((1, 1), (0, 0)), mode="edge")
    prev_delta = padded[:-2]
    cur_delta = padded[1:-1]
    next_delta = padded[2:]
    smoothed = (1.0 - lam) * cur_delta + 0.5 * lam * (prev_delta + next_delta)
    return np.asarray(smoothed, dtype=np.float64)


def _apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    kind = str(activation).strip().lower()
    arr = np.asarray(x, dtype=np.float64)
    if kind == "sigmoid":
        out = np.empty_like(arr, dtype=np.float64)
        pos = arr >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
        exp_arr = np.exp(arr[~pos])
        out[~pos] = exp_arr / (1.0 + exp_arr)
        return out
    if kind == "sine":
        return np.sin(arr)
    raise ValueError(f"Unsupported activation={activation}; use sigmoid or sine")


def _activation_inverse(y: np.ndarray, activation: str) -> np.ndarray:
    kind = str(activation).strip().lower()
    arr = np.asarray(y, dtype=np.float64)
    eps = 1e-6
    if kind == "sigmoid":
        arr_clip = np.clip(arr, eps, 1.0 - eps)
        return np.log(arr_clip / (1.0 - arr_clip))
    if kind == "sine":
        arr_clip = np.clip(arr, -1.0 + eps, 1.0 - eps)
        return np.arcsin(arr_clip)
    raise ValueError(f"Unsupported activation={activation}; use sigmoid or sine")


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=np.float64).ravel()
    xb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(xa))
    nb = float(np.linalg.norm(xb))
    if na <= 1e-12 or nb <= 1e-12:
        return 1.0
    return float(np.dot(xa, xb) / (na * nb + 1e-12))


def _mean_direction_vector(direction_seq_list: Sequence[np.ndarray]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for seq in direction_seq_list:
        arr = np.asarray(seq, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            continue
        keep = np.linalg.norm(arr, axis=1) > 1e-12
        if np.any(keep):
            rows.append(arr[keep])
    if not rows:
        return np.zeros((0,), dtype=np.float64)
    pooled = np.concatenate(rows, axis=0).astype(np.float64)
    return _normalize_direction(np.mean(pooled, axis=0).astype(np.float64))


def _fit_classwise_kmeans_prototypes(
    rows: Sequence[Dict[str, object]],
    *,
    prototype_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise ImportError("PIA operator value probe requires scikit-learn in the active environment.") from e

    x = np.stack([np.asarray(r["z_window"], dtype=np.float64) for r in rows], axis=0).astype(np.float64)
    k = int(max(1, min(int(prototype_count), int(x.shape[0]))))
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
    labels = km.fit_predict(x)
    return np.asarray(km.cluster_centers_, dtype=np.float64), np.asarray(labels, dtype=np.int64)


@dataclass(frozen=True)
class FixedReferenceGeometryConfig:
    prototype_count: int = 4
    anchors_per_prototype: int = 8
    same_dist_quantile: float = 50.0
    anchor_selection_mode: str = "tight_margin"
    seed: int | None = None


@dataclass
class FixedReferenceGeometry:
    prototypes_by_class: Dict[int, np.ndarray]
    all_prototypes: List[tuple[int, int, np.ndarray]]
    anchor_rows: List[Dict[str, object]]
    fit_rows: List[Dict[str, object]]
    fit_windows: np.ndarray
    fit_window_count: int
    fit_trial_count: int
    z_dim: int
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CenterBasedOperatorConfig:
    epsilon_scale: float = 0.10
    smooth_lambda: float = 0.50


@dataclass(frozen=True)
class SingleTemplatePIAValueConfig:
    r_dimension: int = 1
    n_iters: int = 3
    C_repr: float = 1.0
    activation: str = "sigmoid"
    bias_lr: float = 0.25
    orthogonalize: bool = True
    enable_repr_learning: bool = True
    bias_update_mode: str = "residual"
    epsilon_scale: float = 0.10
    smooth_lambda: float = 0.50
    fit_mode: str = "unweighted"
    template_readout_mode: str = "mean_committee"
    seed: int | None = None


@dataclass(frozen=True)
class SingleTemplatePIADiscriminativeConfig:
    r_dimension: int = 1
    n_iters: int = 3
    C_repr: float = 1.0
    activation: str = "sigmoid"
    bias_lr: float = 0.25
    orthogonalize: bool = True
    enable_repr_learning: bool = True
    bias_update_mode: str = "residual"
    target_pos: float = 0.95
    target_neg: float = 0.05
    target_mode: str = "logit_soft"
    template_readout_mode: str = "mean_committee"
    opp_pair_rule: str = "nearest_opposite_prototype"
    seed: int | None = None


@dataclass
class SingleTemplatePIAOperator:
    telm_artifacts: TELM2Artifacts
    activation: str
    direction: np.ndarray
    readout_w: np.ndarray
    readout_b: float
    template_directions: np.ndarray
    response_mean: float
    response_std: float
    response_scale: float
    response_scale_iqr: float
    preactivation_clip_lower: float
    preactivation_clip_upper: float
    pooled_window_count: int
    fit_trial_count: int
    z_dim: int
    geometry: FixedReferenceGeometry
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class OperatorApplyResult:
    z_seq_list: List[np.ndarray]
    summary: Dict[str, object]
    diagnostics_rows: List[Dict[str, object]]
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SingleTemplatePIAStageARepairConfig:
    variant: str = "current_sigmoid_minimal"
    epsilon_scale: float = 0.10
    smooth_lambda: float = 0.50
    budget_target_operator_to_step_ratio: float | None = None
    budget_scale_factor: float | None = None


@dataclass(frozen=True)
class ContinuousGeometricCouplingConfig:
    epsilon_scale: float = 0.10
    smooth_lambda: float = 0.50
    response_variant: str = "sigmoid_clip_tanh_local_median_scaled_iqr"
    budget_target_operator_to_step_ratio: float | None = None
    budget_scale_factor: float | None = None


def build_fixed_reference_geometry(
    *,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    cfg: FixedReferenceGeometryConfig,
) -> FixedReferenceGeometry:
    tids = [str(v) for v in train_tids]
    labels = [int(v) for v in train_labels]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    if len(tids) != len(labels) or len(tids) != len(seqs):
        raise ValueError("train inputs must align")
    if not seqs:
        raise ValueError("train_z_seq_list cannot be empty")

    selection_mode = str(cfg.anchor_selection_mode).strip().lower()
    if selection_mode not in {"nearest", "tight_margin"}:
        raise ValueError("anchor_selection_mode must be one of: nearest, tight_margin")

    rows_by_class: Dict[int, List[Dict[str, object]]] = {}
    for tid, cls, seq in zip(tids, labels, seqs):
        for window_index in range(int(seq.shape[0])):
            row = {
                "trial_id": str(tid),
                "label": int(cls),
                "window_index": int(window_index),
                "z_window": np.asarray(seq[int(window_index)], dtype=np.float32),
            }
            rows_by_class.setdefault(int(cls), []).append(row)

    prototypes_by_class: Dict[int, np.ndarray] = {}
    assignments_by_class: Dict[int, np.ndarray] = {}
    for cls, rows in rows_by_class.items():
        reps, assign = _fit_classwise_kmeans_prototypes(
            rows,
            prototype_count=int(cfg.prototype_count),
            seed=int(0 if cfg.seed is None else cfg.seed) + int(cls),
        )
        prototypes_by_class[int(cls)] = np.asarray(reps, dtype=np.float64)
        assignments_by_class[int(cls)] = np.asarray(assign, dtype=np.int64)

    all_prototypes: List[tuple[int, int, np.ndarray]] = []
    for cls, reps in prototypes_by_class.items():
        for proto_id, rep in enumerate(np.asarray(reps, dtype=np.float64)):
            all_prototypes.append((int(cls), int(proto_id), np.asarray(rep, dtype=np.float64)))

    anchor_rows: List[Dict[str, object]] = []
    fit_windows: List[np.ndarray] = []
    fit_trials: set[str] = set()

    for cls, rows in rows_by_class.items():
        reps = np.asarray(prototypes_by_class[int(cls)], dtype=np.float64)
        assign = np.asarray(assignments_by_class[int(cls)], dtype=np.int64)
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
            if selection_mode == "nearest":
                order = np.argsort(member_dists, kind="mergesort")
                keep = order[: int(max(1, min(int(cfg.anchors_per_prototype), int(order.size))))]
            else:
                same_dist_threshold = float(np.percentile(member_dists, float(cfg.same_dist_quantile)))
                eligible = np.where(member_dists <= same_dist_threshold)[0]
                if eligible.size <= 0:
                    eligible = np.arange(member_dists.size, dtype=np.int64)
                scored: List[tuple[int, float, float]] = []
                for idx in eligible.tolist():
                    z = np.asarray(member_rows[int(idx)]["z_window"], dtype=np.float64)
                    opp_rows = [
                        (other_cls, other_proto_id, other_rep)
                        for other_cls, other_proto_id, other_rep in all_prototypes
                        if int(other_cls) != int(cls)
                    ]
                    if not opp_rows:
                        scored.append((int(idx), float("inf"), float(member_dists[int(idx)])))
                        continue
                    opp_dists = np.asarray(
                        [float(np.linalg.norm(z - np.asarray(other_rep, dtype=np.float64))) for _c, _pid, other_rep in opp_rows],
                        dtype=np.float64,
                    )
                    opp_dist = float(np.min(opp_dists)) if opp_dists.size else float("inf")
                    margin_val = float(opp_dist - float(member_dists[int(idx)]))
                    scored.append((int(idx), float(margin_val), float(member_dists[int(idx)])))
                scored = sorted(scored, key=lambda x: (x[1], x[2], x[0]))
                keep_ids = [int(row[0]) for row in scored[: int(max(1, min(int(cfg.anchors_per_prototype), len(scored))))]]
                keep = np.asarray(keep_ids, dtype=np.int64)

            admitted_rows = [member_rows[int(i)] for i in keep.tolist()]
            prototype_member_count = int(member_ids.size)
            admitted_anchor_count = int(len(admitted_rows))
            anchor_coverage_ratio = float(admitted_anchor_count / max(1, prototype_member_count))
            admitted_payloads: List[tuple[Dict[str, object], np.ndarray, float, float, float]] = []
            for row in admitted_rows:
                z = np.asarray(row["z_window"], dtype=np.float32)
                same_dist = float(np.linalg.norm(np.asarray(z, dtype=np.float64) - rep))
                opp_rows = [
                    (other_cls, other_proto_id, other_rep)
                    for other_cls, other_proto_id, other_rep in all_prototypes
                    if int(other_cls) != int(cls)
                ]
                if opp_rows:
                    opp_dists = np.asarray(
                        [float(np.linalg.norm(np.asarray(z, dtype=np.float64) - np.asarray(other_rep, dtype=np.float64))) for _c, _pid, other_rep in opp_rows],
                        dtype=np.float64,
                    )
                    opp_dist = float(np.min(opp_dists))
                    margin_before = float(opp_dist - same_dist)
                else:
                    opp_dist = same_dist
                    margin_before = 0.0
                admitted_payloads.append((row, np.asarray(z, dtype=np.float32), float(same_dist), float(opp_dist), float(margin_before)))
            admitted_same_dist_mean = float(np.mean([float(v[2]) for v in admitted_payloads])) if admitted_payloads else 0.0
            admitted_margin_mean = float(np.mean([float(v[4]) for v in admitted_payloads])) if admitted_payloads else 0.0
            for row, z, same_dist, opp_dist, margin_before in admitted_payloads:
                fit_windows.append(np.asarray(z, dtype=np.float32))
                fit_trials.add(str(row["trial_id"]))
                anchor_rows.append(
                    {
                        "class_id": int(cls),
                        "prototype_id": int(proto_id),
                        "trial_id": str(row["trial_id"]),
                        "window_index": int(row["window_index"]),
                        "prototype_member_count": int(prototype_member_count),
                        "admitted_anchor_count": int(admitted_anchor_count),
                        "anchor_coverage_ratio": float(anchor_coverage_ratio),
                        "admitted_margin_before": float(margin_before),
                        "admitted_same_dist_before": float(same_dist),
                        "admitted_opp_dist_before": float(opp_dist),
                        "prototype_admitted_same_dist_mean": float(admitted_same_dist_mean),
                        "prototype_admitted_margin_mean": float(admitted_margin_mean),
                    }
                )

    if not fit_windows:
        raise RuntimeError("fixed reference geometry did not admit any fit windows")

    fit_arr = np.stack([np.asarray(v, dtype=np.float32) for v in fit_windows], axis=0).astype(np.float32)
    return FixedReferenceGeometry(
        prototypes_by_class=prototypes_by_class,
        all_prototypes=all_prototypes,
        anchor_rows=anchor_rows,
        fit_rows=list(anchor_rows),
        fit_windows=np.asarray(fit_arr, dtype=np.float32),
        fit_window_count=int(fit_arr.shape[0]),
        fit_trial_count=int(len(fit_trials)),
        z_dim=int(fit_arr.shape[1]),
        meta={
            "anchor_selection_mode": str(selection_mode),
            "prototype_count": int(cfg.prototype_count),
            "anchors_per_prototype": int(cfg.anchors_per_prototype),
            "same_dist_quantile": float(cfg.same_dist_quantile),
        },
    )


def rebuild_fixed_reference_geometry_with_frozen_identities(
    *,
    geometry: FixedReferenceGeometry,
    train_tids: Sequence[str],
    train_z_seq_list: Sequence[np.ndarray],
) -> FixedReferenceGeometry:
    """
    Rebuild prototype coordinates on post-fast states while freezing the original
    object identities carried by (class_id, prototype_id, trial_id, window_index).

    This is the minimal R0 control used by P0b-lite:
    - no new object discovery
    - no new anchor admission
    - only recenter the original prototype identities on transformed windows
    """
    tids = [str(v) for v in train_tids]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    if len(tids) != len(seqs):
        raise ValueError("train_tids and train_z_seq_list must align")
    if not seqs:
        raise ValueError("train_z_seq_list cannot be empty")

    seq_by_tid = {str(tid): np.asarray(seq, dtype=np.float32) for tid, seq in zip(tids, seqs)}
    if len(seq_by_tid) != len(tids):
        raise ValueError("train_tids must be unique for frozen-identity geometry rebuild")

    updated_windows: List[np.ndarray] = []
    updated_rows: List[Dict[str, object]] = []
    proto_windows: Dict[tuple[int, int], List[np.ndarray]] = {}
    fit_trials: set[str] = set()

    for base_row in geometry.fit_rows:
        tid = str(base_row["trial_id"])
        if tid not in seq_by_tid:
            raise KeyError(f"trial_id={tid} missing from transformed train states")
        seq = np.asarray(seq_by_tid[tid], dtype=np.float32)
        window_index = int(base_row["window_index"])
        if window_index < 0 or window_index >= int(seq.shape[0]):
            raise IndexError(f"window_index={window_index} out of range for trial_id={tid}")
        z_window = np.asarray(seq[window_index], dtype=np.float32)
        key = (int(base_row["class_id"]), int(base_row["prototype_id"]))
        proto_windows.setdefault(key, []).append(np.asarray(z_window, dtype=np.float64))
        updated_windows.append(np.asarray(z_window, dtype=np.float32))
        updated_rows.append(
            {
                **dict(base_row),
                "z_window": np.asarray(z_window, dtype=np.float32),
            }
        )
        fit_trials.add(str(tid))

    prototypes_by_class: Dict[int, np.ndarray] = {}
    for class_id, reps in geometry.prototypes_by_class.items():
        arr = np.asarray(reps, dtype=np.float64).copy()
        for proto_id in range(int(arr.shape[0])):
            key = (int(class_id), int(proto_id))
            if key in proto_windows and proto_windows[key]:
                arr[int(proto_id)] = np.mean(np.stack(proto_windows[key], axis=0), axis=0).astype(np.float64)
        prototypes_by_class[int(class_id)] = np.asarray(arr, dtype=np.float64)

    all_prototypes: List[tuple[int, int, np.ndarray]] = []
    for class_id, reps in prototypes_by_class.items():
        for proto_id, rep in enumerate(np.asarray(reps, dtype=np.float64)):
            all_prototypes.append((int(class_id), int(proto_id), np.asarray(rep, dtype=np.float64)))

    proto_stats: Dict[tuple[int, int], Dict[str, float]] = {}
    for (class_id, prototype_id), win_list in proto_windows.items():
        same_center = np.asarray(prototypes_by_class[int(class_id)][int(prototype_id)], dtype=np.float64)
        same_dists: List[float] = []
        margins: List[float] = []
        for z_window in win_list:
            z = np.asarray(z_window, dtype=np.float64)
            same_dist = float(np.linalg.norm(z - same_center))
            opp_rows = [
                (other_cls, other_pid, other_rep)
                for other_cls, other_pid, other_rep in all_prototypes
                if int(other_cls) != int(class_id)
            ]
            if opp_rows:
                opp_dists = np.asarray(
                    [float(np.linalg.norm(z - np.asarray(other_rep, dtype=np.float64))) for _c, _p, other_rep in opp_rows],
                    dtype=np.float64,
                )
                opp_dist = float(np.min(opp_dists))
            else:
                opp_dist = float(same_dist)
            same_dists.append(float(same_dist))
            margins.append(float(opp_dist - same_dist))
        proto_stats[(int(class_id), int(prototype_id))] = {
            "same_dist_mean": float(np.mean(np.asarray(same_dists, dtype=np.float64))) if same_dists else 0.0,
            "margin_mean": float(np.mean(np.asarray(margins, dtype=np.float64))) if margins else 0.0,
        }

    for row in updated_rows:
        class_id = int(row["class_id"])
        prototype_id = int(row["prototype_id"])
        z = np.asarray(row["z_window"], dtype=np.float64)
        p_same = np.asarray(prototypes_by_class[int(class_id)][int(prototype_id)], dtype=np.float64)
        same_dist = float(np.linalg.norm(z - p_same))
        opp_rows = [
            (other_cls, other_pid, other_rep)
            for other_cls, other_pid, other_rep in all_prototypes
            if int(other_cls) != int(class_id)
        ]
        if opp_rows:
            opp_dists = np.asarray(
                [float(np.linalg.norm(z - np.asarray(other_rep, dtype=np.float64))) for _c, _p, other_rep in opp_rows],
                dtype=np.float64,
            )
            opp_dist = float(np.min(opp_dists))
        else:
            opp_dist = float(same_dist)
        stat_row = proto_stats[(int(class_id), int(prototype_id))]
        row["admitted_same_dist_before"] = float(same_dist)
        row["admitted_opp_dist_before"] = float(opp_dist)
        row["admitted_margin_before"] = float(opp_dist - same_dist)
        row["prototype_admitted_same_dist_mean"] = float(stat_row["same_dist_mean"])
        row["prototype_admitted_margin_mean"] = float(stat_row["margin_mean"])

    fit_arr = np.stack(updated_windows, axis=0).astype(np.float32)
    return FixedReferenceGeometry(
        prototypes_by_class=prototypes_by_class,
        all_prototypes=all_prototypes,
        anchor_rows=[dict(v) for v in updated_rows],
        fit_rows=[dict(v) for v in updated_rows],
        fit_windows=np.asarray(fit_arr, dtype=np.float32),
        fit_window_count=int(fit_arr.shape[0]),
        fit_trial_count=int(len(fit_trials)),
        z_dim=int(fit_arr.shape[1]),
        meta={
            **dict(geometry.meta),
            "geometry_rebuild_mode": "frozen_identity_post_fast_refit",
        },
    )


def _lookup_same_prototype_center(
    geometry: FixedReferenceGeometry,
    *,
    class_id: int,
    prototype_id: int,
) -> np.ndarray:
    reps = np.asarray(geometry.prototypes_by_class[int(class_id)], dtype=np.float64)
    if int(prototype_id) < 0 or int(prototype_id) >= int(reps.shape[0]):
        raise KeyError(f"prototype ({class_id}, {prototype_id}) missing from fixed reference geometry")
    return np.asarray(reps[int(prototype_id)], dtype=np.float64)


def _nearest_opposite_prototype_center(
    z: np.ndarray,
    *,
    class_id: int,
    geometry: FixedReferenceGeometry,
) -> tuple[np.ndarray, float]:
    query = np.asarray(z, dtype=np.float64)
    opp_rows = [
        (other_cls, other_proto_id, other_rep)
        for other_cls, other_proto_id, other_rep in geometry.all_prototypes
        if int(other_cls) != int(class_id)
    ]
    if not opp_rows:
        raise RuntimeError("fixed reference geometry requires at least two classes")
    opp_dists = np.asarray(
        [float(np.linalg.norm(query - np.asarray(other_rep, dtype=np.float64))) for _cls, _pid, other_rep in opp_rows],
        dtype=np.float64,
    )
    opp_idx = int(np.argmin(opp_dists))
    _opp_cls, _opp_pid, p_opp = opp_rows[opp_idx]
    return np.asarray(p_opp, dtype=np.float64), float(opp_dists[opp_idx])


def _nearest_opposite_prototype_for_same_prototype(
    geometry: FixedReferenceGeometry,
    *,
    class_id: int,
    prototype_id: int,
) -> tuple[int, int, np.ndarray, float]:
    p_same = _lookup_same_prototype_center(
        geometry,
        class_id=int(class_id),
        prototype_id=int(prototype_id),
    )
    opp_rows = [
        (other_cls, other_proto_id, other_rep)
        for other_cls, other_proto_id, other_rep in geometry.all_prototypes
        if int(other_cls) != int(class_id)
    ]
    if not opp_rows:
        raise RuntimeError("fixed reference geometry requires at least two classes")
    opp_dists = np.asarray(
        [float(np.linalg.norm(np.asarray(p_same, dtype=np.float64) - np.asarray(other_rep, dtype=np.float64))) for other_cls, other_proto_id, other_rep in opp_rows],
        dtype=np.float64,
    )
    opp_idx = int(np.argmin(opp_dists))
    opp_cls, opp_pid, p_opp = opp_rows[opp_idx]
    return int(opp_cls), int(opp_pid), np.asarray(p_opp, dtype=np.float64), float(opp_dists[opp_idx])


def _median_min_side_weights(dist_arr: np.ndarray) -> tuple[np.ndarray, float, bool]:
    arr = np.asarray(dist_arr, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        raise ValueError("distance array cannot be empty")
    if arr.size <= 3:
        return np.ones_like(arr, dtype=np.float64), 0.0, True
    d_min = float(np.min(arr))
    d_med = float(np.median(arr))
    weight_scale = max(1e-8, float(d_med - d_min))
    w_arr = np.exp(-(arr - float(d_min)) / float(weight_scale))
    return np.asarray(w_arr, dtype=np.float64), float(weight_scale), False


def _resolve_operator_readout(
    w: np.ndarray,
    b: np.ndarray,
    *,
    readout_mode: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    w_arr = np.asarray(w, dtype=np.float64)
    if w_arr.ndim != 2:
        w_arr = np.asarray(w_arr, dtype=np.float64).reshape(1, -1)
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if b_arr.size <= 0:
        b_arr = np.zeros((w_arr.shape[0],), dtype=np.float64)
    if b_arr.size == 1 and w_arr.shape[0] > 1:
        b_arr = np.repeat(b_arr, int(w_arr.shape[0])).astype(np.float64)
    if b_arr.size != w_arr.shape[0]:
        raise ValueError("bias dimension must match template count")

    mode = str(readout_mode).strip().lower()
    if mode not in {"first_row", "mean_committee"}:
        raise ValueError("template_readout_mode must be one of: first_row, mean_committee")

    template_directions = np.stack([_normalize_direction(row) for row in w_arr], axis=0).astype(np.float64)
    if mode == "first_row":
        readout_w = np.asarray(w_arr[0], dtype=np.float64)
        readout_b = float(b_arr[0])
    else:
        readout_w = np.asarray(np.mean(w_arr, axis=0), dtype=np.float64)
        readout_b = float(np.mean(b_arr))
    return np.asarray(readout_w, dtype=np.float64), float(readout_b), np.asarray(template_directions, dtype=np.float64)


def _stage_a_response_bundle(
    arr: np.ndarray,
    *,
    operator: SingleTemplatePIAOperator,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_preactivation = np.asarray(arr, dtype=np.float64) @ np.asarray(operator.readout_w, dtype=np.float64) + float(operator.readout_b)
    clipped_preactivation = np.clip(
        raw_preactivation,
        float(operator.preactivation_clip_lower),
        float(operator.preactivation_clip_upper),
    )
    response_scale = max(1e-6, float(operator.response_scale))
    response_scale_iqr = max(1e-6, float(operator.response_scale_iqr))
    if variant == "current_sigmoid_minimal":
        activation_driver = (_apply_activation(raw_preactivation, str(operator.activation)) - float(operator.response_mean)) / float(operator.response_std)
    elif variant == "sigmoid_clip_tanh":
        activation_driver = np.asarray(clipped_preactivation, dtype=np.float64)
    elif variant == "sigmoid_clip_tanh_local_median":
        local_median = float(np.median(clipped_preactivation)) if clipped_preactivation.size else 0.0
        activation_driver = np.asarray(clipped_preactivation - local_median, dtype=np.float64)
    elif variant == "sigmoid_clip_tanh_scaled":
        activation_driver = np.asarray(clipped_preactivation / response_scale, dtype=np.float64)
    elif variant == "sigmoid_clip_tanh_local_median_scaled":
        local_median = float(np.median(clipped_preactivation)) if clipped_preactivation.size else 0.0
        activation_driver = np.asarray((clipped_preactivation - local_median) / response_scale, dtype=np.float64)
    elif variant == "sigmoid_clip_tanh_scaled_iqr":
        activation_driver = np.asarray(clipped_preactivation / response_scale_iqr, dtype=np.float64)
    elif variant == "sigmoid_clip_tanh_local_median_scaled_iqr":
        local_median = float(np.median(clipped_preactivation)) if clipped_preactivation.size else 0.0
        activation_driver = np.asarray((clipped_preactivation - local_median) / response_scale_iqr, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported response variant={variant}")
    response_force = np.tanh(np.asarray(activation_driver, dtype=np.float64))
    clip_mask = (raw_preactivation < float(operator.preactivation_clip_lower)) | (
        raw_preactivation > float(operator.preactivation_clip_upper)
    )
    return (
        np.asarray(raw_preactivation, dtype=np.float64),
        np.asarray(clipped_preactivation, dtype=np.float64),
        np.asarray(activation_driver, dtype=np.float64),
        np.asarray(response_force, dtype=np.float64),
    )


def _build_single_template_operator_from_fit(
    *,
    fit_windows: np.ndarray,
    fit_rows: Sequence[Dict[str, object]],
    fit_trial_count: int,
    geometry: FixedReferenceGeometry,
    activation: str,
    telm_artifacts: TELM2Artifacts,
    meta: Dict[str, object],
    response_stats_windows: np.ndarray | None = None,
    response_stats_mode: str = "fit_pool",
) -> SingleTemplatePIAOperator:
    fit_arr = np.asarray(fit_windows, dtype=np.float64)
    stats_arr = fit_arr if response_stats_windows is None else np.asarray(response_stats_windows, dtype=np.float64)
    if stats_arr.ndim != 2 or stats_arr.shape[1] != fit_arr.shape[1]:
        raise ValueError("response_stats_windows must be [N, D] and match fit window dimensionality")
    w = np.asarray(telm_artifacts.W, dtype=np.float64)
    if w.ndim != 2:
        w = np.asarray(w, dtype=np.float64).reshape(1, -1)
    b = np.asarray(telm_artifacts.b, dtype=np.float64).ravel()
    readout_mode = str(meta.get("template_readout_mode", "first_row" if int(w.shape[0]) == 1 else "mean_committee"))
    readout_w, readout_b, template_directions = _resolve_operator_readout(
        w,
        b,
        readout_mode=str(readout_mode),
    )
    direction = _normalize_direction(readout_w)
    fit_preactivation = fit_arr @ readout_w + float(readout_b)
    stats_preactivation = stats_arr @ readout_w + float(readout_b)
    clip_lower = float(np.quantile(stats_preactivation, 0.01))
    clip_upper = float(np.quantile(stats_preactivation, 0.99))
    if not np.isfinite(clip_lower) or not np.isfinite(clip_upper) or clip_upper <= clip_lower:
        clip_lower = float(np.min(stats_preactivation))
        clip_upper = float(np.max(stats_preactivation))
    stats_response = _apply_activation(stats_preactivation, activation)
    response_mean = float(np.mean(stats_response))
    response_std = float(np.std(stats_response))
    if response_std < 1e-6:
        response_std = 1.0
    response_scale = float(np.std(stats_preactivation))
    if response_scale < 1e-6:
        response_scale = 1.0
    q25 = float(np.quantile(stats_preactivation, 0.25))
    q75 = float(np.quantile(stats_preactivation, 0.75))
    response_scale_iqr = float((q75 - q25) / 1.349)
    if response_scale_iqr < 1e-6 or not np.isfinite(response_scale_iqr):
        response_scale_iqr = float(response_scale)

    template_geom_cosines: List[float] = []
    for row, z in zip(list(fit_rows), fit_arr):
        if "u_geom_vector" in row:
            u_geom = _normalize_direction(np.asarray(row["u_geom_vector"], dtype=np.float64))
        else:
            p_same = _lookup_same_prototype_center(
                geometry,
                class_id=int(row["class_id"]),
                prototype_id=int(row["prototype_id"]),
            )
            p_opp, _opp_dist = _nearest_opposite_prototype_center(
                z,
                class_id=int(row["class_id"]),
                geometry=geometry,
            )
            u_geom = _normalize_direction(p_same - p_opp)
        template_geom_cosines.append(float(_safe_cosine(direction, u_geom)))
    template_mean_direction_cosine = float(np.mean(np.asarray(template_geom_cosines, dtype=np.float64))) if template_geom_cosines else 0.0

    return SingleTemplatePIAOperator(
        telm_artifacts=telm_artifacts,
        activation=str(activation),
        direction=np.asarray(direction, dtype=np.float64),
        readout_w=np.asarray(readout_w, dtype=np.float64),
        readout_b=float(readout_b),
        template_directions=np.asarray(template_directions, dtype=np.float64),
        response_mean=float(response_mean),
        response_std=float(response_std),
        response_scale=float(response_scale),
        response_scale_iqr=float(response_scale_iqr),
        preactivation_clip_lower=float(clip_lower),
        preactivation_clip_upper=float(clip_upper),
        pooled_window_count=int(fit_arr.shape[0]),
        fit_trial_count=int(fit_trial_count),
        z_dim=int(fit_arr.shape[1]),
        geometry=geometry,
        meta={
            "activation": str(activation),
            "preactivation_clip_lower": float(clip_lower),
            "preactivation_clip_upper": float(clip_upper),
            "fit_preactivation_mean": float(np.mean(fit_preactivation)),
            "fit_preactivation_std": float(np.std(fit_preactivation)),
            "fit_preactivation_clipped_std": float(np.std(fit_preactivation)),
            "fit_preactivation_iqr_scale": float((float(np.quantile(fit_preactivation, 0.75)) - float(np.quantile(fit_preactivation, 0.25))) / 1.349) if fit_preactivation.size else 0.0,
            "response_stats_mode": str(response_stats_mode),
            "response_stats_window_count": int(stats_arr.shape[0]),
            "response_stats_preactivation_mean": float(np.mean(stats_preactivation)),
            "response_stats_preactivation_std": float(np.std(stats_preactivation)),
            "response_stats_clipped_std": float(response_scale),
            "response_stats_iqr_scale": float(response_scale_iqr),
            "template_count": int(w.shape[0]),
            "template_readout_mode": str(readout_mode),
            "template_mean_direction_cosine": float(template_mean_direction_cosine),
            "geometry_alignment_cosine_mean": float(template_mean_direction_cosine),
            **dict(meta),
        },
    )


def _build_stage_c_weight_bundle(
    geometry: FixedReferenceGeometry,
    *,
    fit_mode: str,
) -> tuple[np.ndarray | None, Dict[str, object]]:
    if len(geometry.fit_rows) != int(geometry.fit_windows.shape[0]):
        raise RuntimeError("fit_rows must align one-to-one with fit_windows for Stage-C weighted fitting")
    fit_mode_in = str(fit_mode).strip().lower()
    fit_mode_alias = {
        "unweighted": "unweighted",
        "soft_weighted": "mean_dist_weighted",
        "mean_dist_weighted": "mean_dist_weighted",
        "median_min_weighted": "median_min_weighted",
    }
    if fit_mode_in not in fit_mode_alias:
        raise ValueError("fit_mode must be one of: unweighted, soft_weighted, mean_dist_weighted, median_min_weighted")
    fit_mode_canon = str(fit_mode_alias[fit_mode_in])

    weights = np.ones((int(geometry.fit_windows.shape[0]),), dtype=np.float64)
    proto_row_index: Dict[tuple[int, int], List[int]] = {}
    for idx, row in enumerate(geometry.fit_rows):
        key = (int(row["class_id"]), int(row["prototype_id"]))
        proto_row_index.setdefault(key, []).append(int(idx))

    prototype_rows: List[Dict[str, object]] = []
    proto_effective_sizes: List[float] = []
    proto_scales: List[float] = []
    small_proto_fallback_count = 0

    for (class_id, prototype_id), row_ids in sorted(proto_row_index.items()):
        same_arr = np.asarray(
            [float(geometry.fit_rows[int(idx)]["admitted_same_dist_before"]) for idx in row_ids],
            dtype=np.float64,
        )
        margin_arr = np.asarray(
            [float(geometry.fit_rows[int(idx)]["admitted_margin_before"]) for idx in row_ids],
            dtype=np.float64,
        )
        fit_anchor_count = int(len(row_ids))
        fallback_unweighted = False
        if fit_mode_canon == "unweighted":
            kernel_name = "identity"
            w_arr = np.ones((fit_anchor_count,), dtype=np.float64)
            weight_scale = 0.0
        elif fit_mode_canon == "mean_dist_weighted":
            kernel_name = "exp_distance_over_mean"
            weight_scale = max(1e-8, float(np.mean(same_arr))) if same_arr.size else 1.0
            w_arr = np.exp(-same_arr / float(weight_scale))
        else:
            kernel_name = "exp_relative_distance_over_median_minus_min"
            if fit_anchor_count <= 3:
                fallback_unweighted = True
                small_proto_fallback_count += 1
                weight_scale = 0.0
                w_arr = np.ones((fit_anchor_count,), dtype=np.float64)
            else:
                d_min = float(np.min(same_arr)) if same_arr.size else 0.0
                d_med = float(np.median(same_arr)) if same_arr.size else 0.0
                weight_scale = max(1e-8, float(d_med - d_min))
                w_arr = np.exp(-(same_arr - float(d_min)) / float(weight_scale))
        if w_arr.size <= 0 or not np.all(np.isfinite(w_arr)):
            raise RuntimeError("stage-c weights must be finite and non-empty")
        for idx, w_val in zip(row_ids, w_arr.tolist()):
            weights[int(idx)] = float(w_val)
        proto_eff_num = float(np.sum(w_arr))
        proto_eff_den = float(np.sum(np.square(w_arr)))
        proto_eff_size = float((proto_eff_num * proto_eff_num) / max(1e-8, proto_eff_den))
        proto_effective_sizes.append(float(proto_eff_size))
        if float(weight_scale) > 0.0:
            proto_scales.append(float(weight_scale))
        prototype_rows.append(
            {
                "class_id": int(class_id),
                "prototype_id": int(prototype_id),
                "fit_anchor_count": int(fit_anchor_count),
                "fit_anchor_margin_mean": float(np.mean(margin_arr)) if margin_arr.size else 0.0,
                "fit_anchor_same_dist_mean": float(np.mean(same_arr)) if same_arr.size else 0.0,
                "weight_mean": float(np.mean(w_arr)) if w_arr.size else 0.0,
                "weight_min": float(np.min(w_arr)) if w_arr.size else 0.0,
                "weight_max": float(np.max(w_arr)) if w_arr.size else 0.0,
                "effective_sample_size": float(proto_eff_size),
                "effective_sample_ratio": float(proto_eff_size / max(1, fit_anchor_count)),
                "weight_scale": float(weight_scale),
                "weight_kernel_name": str(kernel_name),
                "fallback_unweighted": bool(fallback_unweighted),
            }
        )

    weight_arr = np.asarray(weights, dtype=np.float64)
    if weight_arr.size <= 0 or not np.all(np.isfinite(weight_arr)):
        raise RuntimeError("stage-c weights must be finite and non-empty")
    eff_num = float(np.sum(weight_arr))
    eff_den = float(np.sum(np.square(weight_arr)))
    effective_sample_size = float((eff_num * eff_num) / max(1e-8, eff_den))

    if fit_mode_canon == "unweighted":
        sample_weights = None
        weight_kernel_name = "identity"
    else:
        sample_weights = np.asarray(weight_arr, dtype=np.float64)
        weight_kernel_name = str(prototype_rows[0]["weight_kernel_name"]) if prototype_rows else fit_mode_canon

    return sample_weights, {
        "fit_mode": str(fit_mode_canon),
        "weight_kernel_name": str(weight_kernel_name),
        "weight_kernel": str(weight_kernel_name),
        "effective_sample_size": float(effective_sample_size),
        "effective_sample_ratio": float(effective_sample_size / max(1, int(weight_arr.size))),
        "fit_anchor_margin_mean": float(np.mean([float(r["admitted_margin_before"]) for r in geometry.fit_rows])) if geometry.fit_rows else 0.0,
        "fit_anchor_same_dist_mean": float(np.mean([float(r["admitted_same_dist_before"]) for r in geometry.fit_rows])) if geometry.fit_rows else 0.0,
        "min_proto_effective_sample_size": float(np.min(np.asarray(proto_effective_sizes, dtype=np.float64))) if proto_effective_sizes else 0.0,
        "median_proto_effective_sample_size": float(np.median(np.asarray(proto_effective_sizes, dtype=np.float64))) if proto_effective_sizes else 0.0,
        "proto_weight_scale_mean": float(np.mean(np.asarray(proto_scales, dtype=np.float64))) if proto_scales else 0.0,
        "proto_weight_scale_min": float(np.min(np.asarray(proto_scales, dtype=np.float64))) if proto_scales else 0.0,
        "small_proto_fallback_count": int(small_proto_fallback_count),
        "prototype_weight_rows": prototype_rows,
    }


def fit_single_template_pia_operator(
    *,
    geometry: FixedReferenceGeometry,
    cfg: SingleTemplatePIAValueConfig,
) -> SingleTemplatePIAOperator:
    activation = str(cfg.activation).strip().lower()
    if activation in {"linear", "identity"}:
        raise ValueError("single-template PIA probe requires an explicitly nonlinear activation")

    fit_mode = str(cfg.fit_mode).strip().lower()
    sample_weights, fit_mode_meta = _build_stage_c_weight_bundle(
        geometry,
        fit_mode=str(fit_mode),
    )
    fit_arr = np.asarray(geometry.fit_windows, dtype=np.float64)

    telm = TELM2Transformer(
        TELM2Config(
            r_dimension=int(cfg.r_dimension),
            n_iters=int(cfg.n_iters),
            C_repr=float(cfg.C_repr),
            activation=str(activation),
            bias_lr=float(cfg.bias_lr),
            orthogonalize=bool(cfg.orthogonalize),
            enable_repr_learning=bool(cfg.enable_repr_learning),
            bias_update_mode=str(cfg.bias_update_mode),
            seed=None if cfg.seed is None else int(cfg.seed),
        )
    ).fit(fit_arr, sample_weights=sample_weights)
    arts = telm.get_artifacts()
    return _build_single_template_operator_from_fit(
        fit_windows=fit_arr,
        fit_rows=list(geometry.fit_rows),
        fit_trial_count=int(geometry.fit_trial_count),
        geometry=geometry,
        activation=str(activation),
        telm_artifacts=arts,
        meta={
            "r_dimension": int(cfg.r_dimension),
            "template_readout_mode": str(cfg.template_readout_mode),
            "fit_target_mode": "auto_associative",
            "pool_mode": "same_only",
            "opp_pair_rule": "",
            "fit_mode": str(fit_mode_meta.get("fit_mode", fit_mode)),
            "weight_kernel_name": str(fit_mode_meta.get("weight_kernel_name", fit_mode_meta.get("weight_kernel", "identity"))),
            "n_iters": int(cfg.n_iters),
            "C_repr": float(cfg.C_repr),
            "bias_lr": float(cfg.bias_lr),
            "bias_update_mode": str(cfg.bias_update_mode),
            "recon_err_last": float(arts.recon_err[-1]) if arts.recon_err else 0.0,
            **dict(fit_mode_meta),
        },
    )


def fit_single_template_pia_operator_discriminative(
    *,
    geometry: FixedReferenceGeometry,
    cfg: SingleTemplatePIADiscriminativeConfig,
    response_stats_windows: np.ndarray | None = None,
    response_stats_mode: str = "fit_pool",
) -> SingleTemplatePIAOperator:
    activation = str(cfg.activation).strip().lower()
    if activation != "sigmoid":
        raise ValueError("C3 first version currently supports sigmoid activation only")
    pair_rule = str(cfg.opp_pair_rule).strip().lower()
    if pair_rule != "nearest_opposite_prototype":
        raise ValueError("opp_pair_rule must be nearest_opposite_prototype")
    target_mode = str(cfg.target_mode).strip().lower()
    if target_mode not in {"logit_soft", "linear_pm1"}:
        raise ValueError("target_mode must be one of: logit_soft, linear_pm1")

    by_proto: Dict[tuple[int, int], List[int]] = {}
    for idx, row in enumerate(geometry.fit_rows):
        key = (int(row["class_id"]), int(row["prototype_id"]))
        by_proto.setdefault(key, []).append(int(idx))

    if target_mode == "logit_soft":
        pos_scalar = float(_activation_inverse(np.asarray([[float(cfg.target_pos)]], dtype=np.float64), activation)[0, 0])
        neg_scalar = float(_activation_inverse(np.asarray([[float(cfg.target_neg)]], dtype=np.float64), activation)[0, 0])
        fit_target_mode = "hetero_associative_discriminative_logit_soft"
    else:
        pos_scalar = 1.0
        neg_scalar = -1.0
        fit_target_mode = "hetero_associative_discriminative_linear_pm1"

    fit_blocks: List[np.ndarray] = []
    target_blocks: List[np.ndarray] = []
    weight_blocks: List[np.ndarray] = []
    fit_rows: List[Dict[str, object]] = []
    fit_trials: set[str] = set()
    pair_weight_rows: List[Dict[str, object]] = []
    same_pool_count_total = 0
    opp_pool_count_total = 0
    same_weight_mass_total = 0.0
    opp_weight_mass_total = 0.0
    same_proto_eff_sizes: List[float] = []
    opp_proto_eff_sizes: List[float] = []

    for class_id, prototype_id in sorted(by_proto.keys()):
        same_row_ids = list(by_proto[(int(class_id), int(prototype_id))])
        if not same_row_ids:
            continue
        opp_class_id, opp_prototype_id, p_opp, _pair_dist = _nearest_opposite_prototype_for_same_prototype(
            geometry,
            class_id=int(class_id),
            prototype_id=int(prototype_id),
        )
        opp_row_ids = list(by_proto.get((int(opp_class_id), int(opp_prototype_id)), []))
        if not opp_row_ids:
            continue

        p_same = _lookup_same_prototype_center(
            geometry,
            class_id=int(class_id),
            prototype_id=int(prototype_id),
        )
        pair_axis = _normalize_direction(np.asarray(p_same, dtype=np.float64) - np.asarray(p_opp, dtype=np.float64))

        same_arr = np.asarray(geometry.fit_windows[same_row_ids], dtype=np.float64)
        opp_arr = np.asarray(geometry.fit_windows[opp_row_ids], dtype=np.float64)
        same_dists = np.linalg.norm(same_arr - np.asarray(p_same, dtype=np.float64)[None, :], axis=1)
        opp_dists = np.linalg.norm(opp_arr - np.asarray(p_opp, dtype=np.float64)[None, :], axis=1)

        same_w_raw, same_scale, same_fallback = _median_min_side_weights(same_dists)
        opp_w_raw, opp_scale, opp_fallback = _median_min_side_weights(opp_dists)
        same_w = np.asarray(same_w_raw / (float(np.sum(same_w_raw)) + 1e-8), dtype=np.float64)
        opp_w = np.asarray(opp_w_raw / (float(np.sum(opp_w_raw)) + 1e-8), dtype=np.float64)

        same_eff = float((np.sum(same_w) ** 2) / max(1e-8, float(np.sum(np.square(same_w)))))
        opp_eff = float((np.sum(opp_w) ** 2) / max(1e-8, float(np.sum(np.square(opp_w)))))
        same_proto_eff_sizes.append(float(same_eff))
        opp_proto_eff_sizes.append(float(opp_eff))

        same_targets = np.asarray(pos_scalar * pair_axis[None, :], dtype=np.float64).repeat(int(same_arr.shape[0]), axis=0)
        opp_targets = np.asarray(neg_scalar * pair_axis[None, :], dtype=np.float64).repeat(int(opp_arr.shape[0]), axis=0)

        fit_blocks.extend([same_arr, opp_arr])
        target_blocks.extend([same_targets, opp_targets])
        weight_blocks.extend([same_w, opp_w])

        same_pool_count_total += int(same_arr.shape[0])
        opp_pool_count_total += int(opp_arr.shape[0])
        same_weight_mass_total += float(np.sum(same_w))
        opp_weight_mass_total += float(np.sum(opp_w))

        for idx in same_row_ids:
            base_row = dict(geometry.fit_rows[int(idx)])
            fit_trials.add(str(base_row["trial_id"]))
            fit_rows.append(
                {
                    **base_row,
                    "sample_side": "same",
                    "pair_same_class_id": int(class_id),
                    "pair_same_prototype_id": int(prototype_id),
                    "pair_opp_class_id": int(opp_class_id),
                    "pair_opp_prototype_id": int(opp_prototype_id),
                    "u_geom_vector": np.asarray(pair_axis, dtype=np.float64),
                }
            )
        for idx in opp_row_ids:
            base_row = dict(geometry.fit_rows[int(idx)])
            fit_trials.add(str(base_row["trial_id"]))
            fit_rows.append(
                {
                    **base_row,
                    "sample_side": "opp",
                    "pair_same_class_id": int(class_id),
                    "pair_same_prototype_id": int(prototype_id),
                    "pair_opp_class_id": int(opp_class_id),
                    "pair_opp_prototype_id": int(opp_prototype_id),
                    "u_geom_vector": np.asarray(pair_axis, dtype=np.float64),
                }
            )

        pair_weight_rows.append(
            {
                "same_class_id": int(class_id),
                "same_prototype_id": int(prototype_id),
                "opp_class_id": int(opp_class_id),
                "opp_prototype_id": int(opp_prototype_id),
                "same_pool_count": int(same_arr.shape[0]),
                "opp_pool_count": int(opp_arr.shape[0]),
                "same_weight_mass": float(np.sum(same_w)),
                "opp_weight_mass": float(np.sum(opp_w)),
                "same_opp_count_ratio": float(same_arr.shape[0] / max(1, int(opp_arr.shape[0]))),
                "same_opp_weight_mass_ratio": float(np.sum(same_w) / max(1e-8, float(np.sum(opp_w)))),
                "same_proto_effective_sample_size": float(same_eff),
                "opp_proto_effective_sample_size": float(opp_eff),
                "same_weight_scale": float(same_scale),
                "opp_weight_scale": float(opp_scale),
                "same_fallback_unweighted": bool(same_fallback),
                "opp_fallback_unweighted": bool(opp_fallback),
            }
        )

    if not fit_blocks:
        raise RuntimeError("C3 discriminative fit did not assemble any bipolar fit pool")

    fit_arr = np.concatenate(fit_blocks, axis=0).astype(np.float64)
    target_arr = np.concatenate(target_blocks, axis=0).astype(np.float64)
    weight_arr = np.concatenate(weight_blocks, axis=0).astype(np.float64)
    if fit_arr.shape != target_arr.shape:
        raise RuntimeError("C3 target override must align with bipolar fit pool shape")

    telm = TELM2Transformer(
        TELM2Config(
            r_dimension=int(cfg.r_dimension),
            n_iters=int(cfg.n_iters),
            C_repr=float(cfg.C_repr),
            activation=str(activation),
            bias_lr=float(cfg.bias_lr),
            orthogonalize=bool(cfg.orthogonalize),
            enable_repr_learning=bool(cfg.enable_repr_learning),
            bias_update_mode=str(cfg.bias_update_mode),
            seed=None if cfg.seed is None else int(cfg.seed),
        )
    ).fit(fit_arr, sample_weights=weight_arr, target_override=target_arr)
    arts = telm.get_artifacts()

    effective_sample_size = float((np.sum(weight_arr) ** 2) / max(1e-8, float(np.sum(np.square(weight_arr)))))
    same_pool_count_total_f = int(same_pool_count_total)
    opp_pool_count_total_f = int(opp_pool_count_total)

    return _build_single_template_operator_from_fit(
        fit_windows=fit_arr,
        fit_rows=fit_rows,
        fit_trial_count=int(len(fit_trials)),
        geometry=geometry,
        activation=str(activation),
        telm_artifacts=arts,
        response_stats_windows=response_stats_windows,
        response_stats_mode=str(response_stats_mode),
        meta={
            "r_dimension": int(cfg.r_dimension),
            "template_readout_mode": str(cfg.template_readout_mode),
            "fit_target_mode": str(fit_target_mode),
            "target_mode": str(target_mode),
            "pool_mode": "bipolar_same_opp",
            "opp_pair_rule": str(pair_rule),
            "fit_mode": "bipolar_discriminative_weighted",
            "weight_kernel_name": "bipolar_exp_relative_distance_over_median_minus_min",
            "n_iters": int(cfg.n_iters),
            "C_repr": float(cfg.C_repr),
            "bias_lr": float(cfg.bias_lr),
            "bias_update_mode": str(cfg.bias_update_mode),
            "recon_err_last": float(arts.recon_err[-1]) if arts.recon_err else 0.0,
            "effective_sample_size": float(effective_sample_size),
            "effective_sample_ratio": float(effective_sample_size / max(1, int(weight_arr.size))),
            "same_pool_count": int(same_pool_count_total_f),
            "opp_pool_count": int(opp_pool_count_total_f),
            "same_weight_mass": float(same_weight_mass_total),
            "opp_weight_mass": float(opp_weight_mass_total),
            "same_opp_count_ratio": float(same_pool_count_total_f / max(1, opp_pool_count_total_f)),
            "same_opp_weight_mass_ratio": float(same_weight_mass_total / max(1e-8, opp_weight_mass_total)),
            "same_proto_effective_sample_size": float(np.median(np.asarray(same_proto_eff_sizes, dtype=np.float64))) if same_proto_eff_sizes else 0.0,
            "opp_proto_effective_sample_size": float(np.median(np.asarray(opp_proto_eff_sizes, dtype=np.float64))) if opp_proto_eff_sizes else 0.0,
            "discriminative_target_gap": float(pos_scalar - neg_scalar),
            "prototype_weight_rows": pair_weight_rows,
            "pair_weight_rows": pair_weight_rows,
        },
    )


def _assign_same_opp_from_fixed_geometry(
    z: np.ndarray,
    *,
    geometry: FixedReferenceGeometry,
) -> tuple[int, int, np.ndarray, np.ndarray, float, float]:
    query = np.asarray(z, dtype=np.float64)
    rows = geometry.all_prototypes
    if not rows:
        raise RuntimeError("fixed reference geometry has no prototypes")
    dists = np.asarray(
        [float(np.linalg.norm(query - np.asarray(rep, dtype=np.float64))) for _cls, _pid, rep in rows],
        dtype=np.float64,
    )
    same_idx = int(np.argmin(dists))
    same_cls, same_pid, p_same = rows[same_idx]
    same_dist = float(dists[same_idx])
    opp_candidates = [
        (other_cls, other_pid, other_rep)
        for other_cls, other_pid, other_rep in rows
        if int(other_cls) != int(same_cls)
    ]
    if not opp_candidates:
        raise RuntimeError("fixed reference geometry requires at least two classes")
    opp_dists = np.asarray(
        [float(np.linalg.norm(query - np.asarray(rep, dtype=np.float64))) for _cls, _pid, rep in opp_candidates],
        dtype=np.float64,
    )
    opp_idx = int(np.argmin(opp_dists))
    _opp_cls, _opp_pid, p_opp = opp_candidates[opp_idx]
    opp_dist = float(opp_dists[opp_idx])
    return int(same_cls), int(same_pid), np.asarray(p_same, dtype=np.float64), np.asarray(p_opp, dtype=np.float64), float(same_dist), float(opp_dist)


def _summarize_operator_application(
    *,
    original_z_seq_list: Sequence[np.ndarray],
    z_seq_list: Sequence[np.ndarray],
    delta_seq_list: Sequence[np.ndarray],
    direction_seq_list: Sequence[np.ndarray],
    diagnostics_rows: Sequence[Dict[str, object]],
    fit_window_count: int,
    fit_trial_count: int,
    meta: Dict[str, object],
) -> Dict[str, object]:
    distortion_vals: List[float] = []
    delta_norm_vals: List[float] = []
    delta_to_step_vals: List[float] = []
    direction_stability_vals: List[float] = []
    total_windows = 0
    for seq_orig, seq_aug, delta_seq, dir_seq in zip(original_z_seq_list, z_seq_list, delta_seq_list, direction_seq_list):
        arr_orig = np.asarray(seq_orig, dtype=np.float64)
        arr_aug = np.asarray(seq_aug, dtype=np.float64)
        arr_delta = np.asarray(delta_seq, dtype=np.float64)
        arr_dir = np.asarray(dir_seq, dtype=np.float64)
        for idx in range(int(arr_orig.shape[0])):
            total_windows += 1
            orig_step = float(_window_local_step_mean(arr_orig, idx))
            aug_step = float(_window_local_step_mean(arr_aug, idx))
            delta_norm = float(np.linalg.norm(arr_delta[int(idx)]))
            distortion_vals.append(float(aug_step / max(1e-6, orig_step)))
            delta_norm_vals.append(float(delta_norm))
            delta_to_step_vals.append(float(delta_norm / max(1e-6, orig_step)))
        if int(arr_dir.shape[0]) > 1:
            for d0, d1 in zip(arr_dir[:-1], arr_dir[1:]):
                direction_stability_vals.append(float(_safe_cosine(d0, d1)))

    distort_arr = np.asarray(distortion_vals, dtype=np.float64)
    delta_norm_arr = np.asarray(delta_norm_vals, dtype=np.float64)
    delta_to_step_arr = np.asarray(delta_to_step_vals, dtype=np.float64)
    direction_stability_arr = np.asarray(direction_stability_vals, dtype=np.float64)
    return {
        "fit_window_count": int(fit_window_count),
        "fit_trial_count": int(fit_trial_count),
        "applied_window_count": int(total_windows),
        "accepted_window_ratio": 1.0 if total_windows > 0 else 0.0,
        "shaped_window_ratio": 1.0 if total_windows > 0 else 0.0,
        "operator_norm_mean": float(np.mean(delta_norm_arr)) if delta_norm_arr.size else 0.0,
        "operator_norm_p95": float(np.percentile(delta_norm_arr, 95.0)) if delta_norm_arr.size else 0.0,
        "operator_to_step_ratio_mean": float(np.mean(delta_to_step_arr)) if delta_to_step_arr.size else 0.0,
        "local_step_distortion_ratio_mean": float(np.mean(distort_arr)) if distort_arr.size else 1.0,
        "local_step_distortion_ratio_p95": float(np.percentile(distort_arr, 95.0)) if distort_arr.size else 1.0,
        "operator_direction_stability": float(np.mean(direction_stability_arr)) if direction_stability_arr.size else 1.0,
        "diagnostic_row_count": int(len(diagnostics_rows)),
        **dict(meta),
    }


def apply_center_based_operator(
    *,
    z_seq_list: Sequence[np.ndarray],
    geometry: FixedReferenceGeometry,
    cfg: CenterBasedOperatorConfig,
) -> OperatorApplyResult:
    z_aug_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    direction_list: List[np.ndarray] = []
    diagnostic_rows: List[Dict[str, object]] = []
    for seq_idx, seq in enumerate(z_seq_list):
        arr = np.asarray(seq, dtype=np.float64)
        raw_delta = np.zeros_like(arr, dtype=np.float64)
        raw_dir = np.zeros_like(arr, dtype=np.float64)
        for window_index in range(int(arr.shape[0])):
            z = np.asarray(arr[int(window_index)], dtype=np.float64)
            same_cls, same_pid, p_same, p_opp, same_dist, opp_dist = _assign_same_opp_from_fixed_geometry(
                z,
                geometry=geometry,
            )
            denom = float(same_dist + opp_dist + 1e-6)
            lambda_intra = float(opp_dist / denom)
            lambda_inter = float(same_dist / denom)
            raw_vec = lambda_intra * (p_same - z) + lambda_inter * (z - p_opp)
            raw_norm = float(np.linalg.norm(raw_vec))
            if raw_norm <= 1e-12 or not np.isfinite(raw_norm):
                continue
            unit_dir = np.asarray(raw_vec / raw_norm, dtype=np.float64)
            local_step = float(_window_local_step_mean(arr, int(window_index)))
            ambiguity_gate = float(same_dist / denom)
            raw_delta[int(window_index)] = float(cfg.epsilon_scale) * float(local_step) * float(ambiguity_gate) * unit_dir
            raw_dir[int(window_index)] = unit_dir
            diagnostic_rows.append(
                {
                    "sequence_index": int(seq_idx),
                    "window_index": int(window_index),
                    "assigned_class_id": int(same_cls),
                    "assigned_prototype_id": int(same_pid),
                    "same_dist": float(same_dist),
                    "opp_dist": float(opp_dist),
                    "lambda_intra": float(lambda_intra),
                    "lambda_inter": float(lambda_inter),
                    "ambiguity_gate": float(ambiguity_gate),
                    "delta_norm_raw": float(np.linalg.norm(raw_delta[int(window_index)])),
                }
            )
        delta_smooth = _smooth_delta(raw_delta, float(cfg.smooth_lambda))
        z_aug = arr + delta_smooth
        z_aug_list.append(np.asarray(z_aug, dtype=np.float32))
        delta_list.append(np.asarray(delta_smooth, dtype=np.float32))
        direction_list.append(np.asarray(raw_dir, dtype=np.float32))

    summary = _summarize_operator_application(
        original_z_seq_list=z_seq_list,
        z_seq_list=z_aug_list,
        delta_seq_list=delta_list,
        direction_seq_list=direction_list,
        diagnostics_rows=diagnostic_rows,
        fit_window_count=int(geometry.fit_window_count),
        fit_trial_count=int(geometry.fit_trial_count),
        meta={
            "operator_arm": "mean_centered",
            "epsilon_scale": float(cfg.epsilon_scale),
            "smooth_lambda": float(cfg.smooth_lambda),
        },
    )
    return OperatorApplyResult(
        z_seq_list=z_aug_list,
        summary=summary,
        diagnostics_rows=diagnostic_rows,
        meta={
            "operator_arm": "mean_centered",
            "mean_direction_vector": _mean_direction_vector(direction_list).tolist(),
        },
    )


def apply_single_template_pia_operator(
    *,
    z_seq_list: Sequence[np.ndarray],
    operator: SingleTemplatePIAOperator,
    cfg: SingleTemplatePIAValueConfig,
) -> OperatorApplyResult:
    z_aug_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    direction_list: List[np.ndarray] = []
    diagnostic_rows: List[Dict[str, object]] = []
    raw_w = np.asarray(operator.readout_w, dtype=np.float64)
    readout_b = float(operator.readout_b)
    direction = np.asarray(operator.direction, dtype=np.float64)

    for seq_idx, seq in enumerate(z_seq_list):
        arr = np.asarray(seq, dtype=np.float64)
        raw_delta = np.zeros_like(arr, dtype=np.float64)
        raw_dir = np.zeros_like(arr, dtype=np.float64)
        response = _apply_activation(arr @ raw_w + float(readout_b), str(operator.activation))
        centered_response = (np.asarray(response, dtype=np.float64) - float(operator.response_mean)) / float(operator.response_std)
        for window_index in range(int(arr.shape[0])):
            gate = float(np.tanh(abs(float(centered_response[int(window_index)]))))
            sign = 1.0 if float(centered_response[int(window_index)]) >= 0.0 else -1.0
            local_step = float(_window_local_step_mean(arr, int(window_index)))
            same_cls, same_pid, p_same, p_opp, same_dist, opp_dist = _assign_same_opp_from_fixed_geometry(
                arr[int(window_index)],
                geometry=operator.geometry,
            )
            margin_before = float(opp_dist - same_dist)
            raw_dir[int(window_index)] = float(sign) * direction
            raw_delta[int(window_index)] = float(cfg.epsilon_scale) * float(local_step) * float(gate) * raw_dir[int(window_index)]
            diagnostic_rows.append(
                {
                    "sequence_index": int(seq_idx),
                    "window_index": int(window_index),
                    "assigned_class_id": int(same_cls),
                    "assigned_prototype_id": int(same_pid),
                    "same_dist": float(same_dist),
                    "opp_dist": float(opp_dist),
                    "margin_before": float(margin_before),
                    "response": float(response[int(window_index)]),
                    "centered_response": float(centered_response[int(window_index)]),
                    "response_gate": float(gate),
                    "delta_norm_raw": float(np.linalg.norm(raw_delta[int(window_index)])),
                }
            )
        delta_smooth = _smooth_delta(raw_delta, float(cfg.smooth_lambda))
        z_aug = arr + delta_smooth
        z_aug_list.append(np.asarray(z_aug, dtype=np.float32))
        delta_list.append(np.asarray(delta_smooth, dtype=np.float32))
        direction_list.append(np.asarray(raw_dir, dtype=np.float32))

    response_arr = np.asarray([float(r["response"]) for r in diagnostic_rows], dtype=np.float64)
    centered_response_arr = np.asarray([float(r["centered_response"]) for r in diagnostic_rows], dtype=np.float64)
    margin_arr = np.asarray([float(r["margin_before"]) for r in diagnostic_rows], dtype=np.float64)
    if response_arr.size >= 2 and np.std(response_arr) > 1e-12 and np.std(margin_arr) > 1e-12:
        response_vs_margin_correlation = float(np.corrcoef(response_arr, margin_arr)[0, 1])
    else:
        response_vs_margin_correlation = 0.0
    activation_coverage_ratio = float(np.mean(np.abs(centered_response_arr) >= 1.0)) if centered_response_arr.size else 0.0

    summary = _summarize_operator_application(
        original_z_seq_list=z_seq_list,
        z_seq_list=z_aug_list,
        delta_seq_list=delta_list,
        direction_seq_list=direction_list,
        diagnostics_rows=diagnostic_rows,
        fit_window_count=int(operator.pooled_window_count),
        fit_trial_count=int(operator.fit_trial_count),
        meta={
            "operator_arm": "single_template_pia",
            "epsilon_scale": float(cfg.epsilon_scale),
            "smooth_lambda": float(cfg.smooth_lambda),
            "activation": str(operator.activation),
            "fit_target_mode": str(operator.meta.get("fit_target_mode", "auto_associative")),
            "target_mode": str(operator.meta.get("target_mode", "")),
            "pool_mode": str(operator.meta.get("pool_mode", "same_only")),
            "opp_pair_rule": str(operator.meta.get("opp_pair_rule", "")),
            "template_count": int(operator.meta.get("template_count", 1)),
            "template_readout_mode": str(operator.meta.get("template_readout_mode", "first_row")),
            "response_stats_mode": str(operator.meta.get("response_stats_mode", "fit_pool")),
            "response_stats_window_count": int(operator.meta.get("response_stats_window_count", operator.pooled_window_count)),
            "recon_err_last": float(operator.meta.get("recon_err_last", 0.0)),
            "response_mean": float(operator.response_mean),
            "response_std": float(operator.response_std),
            "response_vs_margin_correlation": float(response_vs_margin_correlation),
            "activation_coverage_ratio": float(activation_coverage_ratio),
            "template_mean_direction_cosine": float(operator.meta.get("template_mean_direction_cosine", 0.0)),
            "geometry_alignment_cosine_mean": float(operator.meta.get("geometry_alignment_cosine_mean", 0.0)),
            "weight_kernel_name": str(operator.meta.get("weight_kernel_name", operator.meta.get("weight_kernel", "identity"))),
            "effective_sample_size": float(operator.meta.get("effective_sample_size", 0.0)),
            "effective_sample_ratio": float(operator.meta.get("effective_sample_ratio", 0.0)),
            "min_proto_effective_sample_size": float(operator.meta.get("min_proto_effective_sample_size", 0.0)),
            "median_proto_effective_sample_size": float(operator.meta.get("median_proto_effective_sample_size", 0.0)),
            "fit_anchor_margin_mean": float(operator.meta.get("fit_anchor_margin_mean", 0.0)),
            "fit_anchor_same_dist_mean": float(operator.meta.get("fit_anchor_same_dist_mean", 0.0)),
            "proto_weight_scale_mean": float(operator.meta.get("proto_weight_scale_mean", 0.0)),
            "proto_weight_scale_min": float(operator.meta.get("proto_weight_scale_min", 0.0)),
            "same_pool_count": int(operator.meta.get("same_pool_count", 0)),
            "opp_pool_count": int(operator.meta.get("opp_pool_count", 0)),
            "same_weight_mass": float(operator.meta.get("same_weight_mass", 0.0)),
            "opp_weight_mass": float(operator.meta.get("opp_weight_mass", 0.0)),
            "same_opp_count_ratio": float(operator.meta.get("same_opp_count_ratio", 0.0)),
            "same_opp_weight_mass_ratio": float(operator.meta.get("same_opp_weight_mass_ratio", 0.0)),
            "same_proto_effective_sample_size": float(operator.meta.get("same_proto_effective_sample_size", 0.0)),
            "opp_proto_effective_sample_size": float(operator.meta.get("opp_proto_effective_sample_size", 0.0)),
            "discriminative_target_gap": float(operator.meta.get("discriminative_target_gap", 0.0)),
        },
    )
    return OperatorApplyResult(
        z_seq_list=z_aug_list,
        summary=summary,
        diagnostics_rows=diagnostic_rows,
        meta={
            "operator_arm": "single_template_pia",
            "mean_direction_vector": _mean_direction_vector(direction_list).tolist(),
            **dict(operator.meta),
        },
    )


def apply_single_template_pia_stage_a_variant(
    *,
    z_seq_list: Sequence[np.ndarray],
    operator: SingleTemplatePIAOperator,
    cfg: SingleTemplatePIAStageARepairConfig,
) -> OperatorApplyResult:
    variant = str(cfg.variant).strip().lower()
    allowed = {
        "current_sigmoid_minimal",
        "sigmoid_clip_tanh",
        "sigmoid_clip_tanh_local_median",
        "sigmoid_clip_tanh_scaled",
        "sigmoid_clip_tanh_local_median_scaled",
        "sigmoid_clip_tanh_scaled_iqr",
        "sigmoid_clip_tanh_local_median_scaled_iqr",
    }
    if variant not in allowed:
        raise ValueError(f"variant must be one of {sorted(allowed)}")

    z_aug_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    direction_list: List[np.ndarray] = []
    diagnostic_rows: List[Dict[str, object]] = []
    direction = np.asarray(operator.direction, dtype=np.float64)

    raw_delta_list: List[np.ndarray] = []
    local_step_vals: List[float] = []

    for seq_idx, seq in enumerate(z_seq_list):
        arr = np.asarray(seq, dtype=np.float64)
        raw_delta = np.zeros_like(arr, dtype=np.float64)
        raw_dir = np.zeros_like(arr, dtype=np.float64)
        raw_preactivation, clipped_preactivation, activation_driver, response_force = _stage_a_response_bundle(
            arr,
            operator=operator,
            variant=str(variant),
        )
        clip_mask = (raw_preactivation < float(operator.preactivation_clip_lower)) | (
            raw_preactivation > float(operator.preactivation_clip_upper)
        )

        for window_index in range(int(arr.shape[0])):
            local_step = float(_window_local_step_mean(arr, int(window_index)))
            same_cls, same_pid, p_same, p_opp, same_dist, opp_dist = _assign_same_opp_from_fixed_geometry(
                arr[int(window_index)],
                geometry=operator.geometry,
            )
            margin_before = float(opp_dist - same_dist)
            raw_dir[int(window_index)] = direction
            raw_delta[int(window_index)] = float(cfg.epsilon_scale) * float(local_step) * float(response_force[int(window_index)]) * direction
            local_step_vals.append(float(local_step))
            diagnostic_rows.append(
                {
                    "sequence_index": int(seq_idx),
                    "window_index": int(window_index),
                    "assigned_class_id": int(same_cls),
                    "assigned_prototype_id": int(same_pid),
                    "same_dist": float(same_dist),
                    "opp_dist": float(opp_dist),
                    "margin_before": float(margin_before),
                    "raw_preactivation": float(raw_preactivation[int(window_index)]),
                    "clipped_preactivation": float(clipped_preactivation[int(window_index)]),
                    "activation_driver": float(activation_driver[int(window_index)]),
                    "response_force": float(response_force[int(window_index)]),
                    "clip_applied": bool(clip_mask[int(window_index)]),
                    "response_gate": float(abs(response_force[int(window_index)])),
                    "delta_norm_raw": float(np.linalg.norm(raw_delta[int(window_index)])),
                }
            )

        raw_delta_list.append(np.asarray(raw_delta, dtype=np.float64))
        direction_list.append(np.asarray(raw_dir, dtype=np.float32))

    smooth_delta_list = [_smooth_delta(v, float(cfg.smooth_lambda)) for v in raw_delta_list]
    if cfg.budget_scale_factor is not None:
        budget_scale = float(cfg.budget_scale_factor)
    elif float(cfg.budget_target_operator_to_step_ratio or 0.0) > 0.0:
        delta_norm_vals = [float(np.linalg.norm(v[i])) for v, seq in zip(smooth_delta_list, z_seq_list) for i in range(int(np.asarray(seq).shape[0]))]
        current_ratio = float(np.mean(np.asarray(delta_norm_vals, dtype=np.float64) / np.maximum(1e-6, np.asarray(local_step_vals, dtype=np.float64)))) if local_step_vals else 0.0
        budget_scale = float(cfg.budget_target_operator_to_step_ratio) / max(1e-8, current_ratio) if current_ratio > 0.0 else 1.0
    else:
        budget_scale = 1.0
    smooth_delta_list = [np.asarray(v * float(budget_scale), dtype=np.float64) for v in smooth_delta_list]

    for seq, delta_smooth in zip(z_seq_list, smooth_delta_list):
        arr = np.asarray(seq, dtype=np.float64)
        z_aug = arr + delta_smooth
        z_aug_list.append(np.asarray(z_aug, dtype=np.float32))
        delta_list.append(np.asarray(delta_smooth, dtype=np.float32))

    response_force_arr = np.asarray([float(r["response_force"]) for r in diagnostic_rows], dtype=np.float64)
    activation_driver_arr = np.asarray([float(r["activation_driver"]) for r in diagnostic_rows], dtype=np.float64)
    margin_arr = np.asarray([float(r["margin_before"]) for r in diagnostic_rows], dtype=np.float64)
    clip_applied_arr = np.asarray([1.0 if bool(r["clip_applied"]) else 0.0 for r in diagnostic_rows], dtype=np.float64)
    if response_force_arr.size >= 2 and np.std(response_force_arr) > 1e-12 and np.std(margin_arr) > 1e-12:
        response_vs_margin_correlation = float(np.corrcoef(response_force_arr, margin_arr)[0, 1])
    else:
        response_vs_margin_correlation = 0.0
    activation_coverage_ratio = float(np.mean(np.abs(activation_driver_arr) >= 1.0)) if activation_driver_arr.size else 0.0
    gate_saturation_ratio = float(np.mean(np.abs(response_force_arr) >= 0.95)) if response_force_arr.size else 0.0
    preactivation_clip_rate = float(np.mean(clip_applied_arr)) if clip_applied_arr.size else 0.0

    summary = _summarize_operator_application(
        original_z_seq_list=z_seq_list,
        z_seq_list=z_aug_list,
        delta_seq_list=delta_list,
        direction_seq_list=direction_list,
        diagnostics_rows=diagnostic_rows,
        fit_window_count=int(operator.pooled_window_count),
        fit_trial_count=int(operator.fit_trial_count),
        meta={
            "operator_arm": str(variant),
            "epsilon_scale": float(cfg.epsilon_scale),
            "smooth_lambda": float(cfg.smooth_lambda),
            "budget_scale_factor": float(budget_scale),
            "activation": str(operator.activation),
            "fit_target_mode": str(operator.meta.get("fit_target_mode", "auto_associative")),
            "target_mode": str(operator.meta.get("target_mode", "")),
            "pool_mode": str(operator.meta.get("pool_mode", "same_only")),
            "opp_pair_rule": str(operator.meta.get("opp_pair_rule", "")),
            "template_count": int(operator.meta.get("template_count", 1)),
            "template_readout_mode": str(operator.meta.get("template_readout_mode", "first_row")),
            "response_stats_mode": str(operator.meta.get("response_stats_mode", "fit_pool")),
            "response_stats_window_count": int(operator.meta.get("response_stats_window_count", operator.pooled_window_count)),
            "response_scale": float(operator.response_scale),
            "response_scale_iqr": float(operator.response_scale_iqr),
            "response_vs_margin_correlation": float(response_vs_margin_correlation),
            "activation_coverage_ratio": float(activation_coverage_ratio),
            "preactivation_clip_rate": float(preactivation_clip_rate),
            "response_centering_std_after_fix": float(np.std(activation_driver_arr)) if activation_driver_arr.size else 0.0,
            "gate_saturation_ratio": float(gate_saturation_ratio),
            "fit_mode": str(operator.meta.get("fit_mode", "unweighted")),
            "template_mean_direction_cosine": float(operator.meta.get("template_mean_direction_cosine", 0.0)),
            "geometry_alignment_cosine_mean": float(operator.meta.get("geometry_alignment_cosine_mean", 0.0)),
            "weight_kernel_name": str(operator.meta.get("weight_kernel_name", operator.meta.get("weight_kernel", "identity"))),
            "effective_sample_size": float(operator.meta.get("effective_sample_size", 0.0)),
            "effective_sample_ratio": float(operator.meta.get("effective_sample_ratio", 0.0)),
            "min_proto_effective_sample_size": float(operator.meta.get("min_proto_effective_sample_size", 0.0)),
            "median_proto_effective_sample_size": float(operator.meta.get("median_proto_effective_sample_size", 0.0)),
            "fit_anchor_margin_mean": float(operator.meta.get("fit_anchor_margin_mean", 0.0)),
            "fit_anchor_same_dist_mean": float(operator.meta.get("fit_anchor_same_dist_mean", 0.0)),
            "proto_weight_scale_mean": float(operator.meta.get("proto_weight_scale_mean", 0.0)),
            "proto_weight_scale_min": float(operator.meta.get("proto_weight_scale_min", 0.0)),
            "same_pool_count": int(operator.meta.get("same_pool_count", 0)),
            "opp_pool_count": int(operator.meta.get("opp_pool_count", 0)),
            "same_weight_mass": float(operator.meta.get("same_weight_mass", 0.0)),
            "opp_weight_mass": float(operator.meta.get("opp_weight_mass", 0.0)),
            "same_opp_count_ratio": float(operator.meta.get("same_opp_count_ratio", 0.0)),
            "same_opp_weight_mass_ratio": float(operator.meta.get("same_opp_weight_mass_ratio", 0.0)),
            "same_proto_effective_sample_size": float(operator.meta.get("same_proto_effective_sample_size", 0.0)),
            "opp_proto_effective_sample_size": float(operator.meta.get("opp_proto_effective_sample_size", 0.0)),
            "discriminative_target_gap": float(operator.meta.get("discriminative_target_gap", 0.0)),
        },
    )
    return OperatorApplyResult(
        z_seq_list=z_aug_list,
        summary=summary,
        diagnostics_rows=diagnostic_rows,
        meta={
            "operator_arm": str(variant),
            "budget_scale_factor": float(budget_scale),
            "mean_direction_vector": _mean_direction_vector(direction_list).tolist(),
            **dict(operator.meta),
        },
    )


def apply_continuous_geometric_coupling(
    *,
    z_seq_list: Sequence[np.ndarray],
    operator: SingleTemplatePIAOperator,
    cfg: ContinuousGeometricCouplingConfig,
) -> OperatorApplyResult:
    response_variant = str(cfg.response_variant).strip().lower()
    if response_variant not in {
        "current_sigmoid_minimal",
        "sigmoid_clip_tanh",
        "sigmoid_clip_tanh_local_median",
        "sigmoid_clip_tanh_scaled",
        "sigmoid_clip_tanh_local_median_scaled",
        "sigmoid_clip_tanh_scaled_iqr",
        "sigmoid_clip_tanh_local_median_scaled_iqr",
    }:
        raise ValueError("unsupported response_variant for continuous geometric coupling")

    z_aug_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    direction_list: List[np.ndarray] = []
    diagnostic_rows: List[Dict[str, object]] = []

    template_direction = np.asarray(operator.direction, dtype=np.float64)
    raw_delta_list: List[np.ndarray] = []
    local_step_vals: List[float] = []

    for seq_idx, seq in enumerate(z_seq_list):
        arr = np.asarray(seq, dtype=np.float64)
        raw_delta = np.zeros_like(arr, dtype=np.float64)
        raw_dir = np.zeros_like(arr, dtype=np.float64)
        raw_preactivation, clipped_preactivation, activation_driver, response_force = _stage_a_response_bundle(
            arr,
            operator=operator,
            variant=str(response_variant),
        )
        clip_mask = (raw_preactivation < float(operator.preactivation_clip_lower)) | (
            raw_preactivation > float(operator.preactivation_clip_upper)
        )

        for window_index in range(int(arr.shape[0])):
            z = np.asarray(arr[int(window_index)], dtype=np.float64)
            local_step = float(_window_local_step_mean(arr, int(window_index)))
            same_cls, same_pid, p_same, p_opp, same_dist, opp_dist = _assign_same_opp_from_fixed_geometry(
                z,
                geometry=operator.geometry,
            )
            margin_before = float(opp_dist - same_dist)
            u_geom = _normalize_direction(np.asarray(p_same, dtype=np.float64) - np.asarray(p_opp, dtype=np.float64))
            g_geom = float(_safe_cosine(template_direction, u_geom))
            coupled_direction = np.asarray(g_geom * u_geom, dtype=np.float64)
            raw_dir[int(window_index)] = coupled_direction
            raw_delta[int(window_index)] = float(cfg.epsilon_scale) * float(local_step) * float(response_force[int(window_index)]) * coupled_direction
            local_step_vals.append(float(local_step))
            diagnostic_rows.append(
                {
                    "sequence_index": int(seq_idx),
                    "window_index": int(window_index),
                    "assigned_class_id": int(same_cls),
                    "assigned_prototype_id": int(same_pid),
                    "same_dist": float(same_dist),
                    "opp_dist": float(opp_dist),
                    "margin_before": float(margin_before),
                    "raw_preactivation": float(raw_preactivation[int(window_index)]),
                    "clipped_preactivation": float(clipped_preactivation[int(window_index)]),
                    "activation_driver": float(activation_driver[int(window_index)]),
                    "response_force": float(response_force[int(window_index)]),
                    "clip_applied": bool(clip_mask[int(window_index)]),
                    "response_gate": float(abs(response_force[int(window_index)])),
                    "g_geom": float(g_geom),
                    "delta_norm_raw": float(np.linalg.norm(raw_delta[int(window_index)])),
                }
            )

        raw_delta_list.append(np.asarray(raw_delta, dtype=np.float64))
        direction_list.append(np.asarray(raw_dir, dtype=np.float32))

    smooth_delta_list = [_smooth_delta(v, float(cfg.smooth_lambda)) for v in raw_delta_list]
    if cfg.budget_scale_factor is not None:
        budget_scale = float(cfg.budget_scale_factor)
    elif float(cfg.budget_target_operator_to_step_ratio or 0.0) > 0.0:
        delta_norm_vals = [float(np.linalg.norm(v[i])) for v, seq in zip(smooth_delta_list, z_seq_list) for i in range(int(np.asarray(seq).shape[0]))]
        current_ratio = float(np.mean(np.asarray(delta_norm_vals, dtype=np.float64) / np.maximum(1e-6, np.asarray(local_step_vals, dtype=np.float64)))) if local_step_vals else 0.0
        budget_scale = float(cfg.budget_target_operator_to_step_ratio) / max(1e-8, current_ratio) if current_ratio > 0.0 else 1.0
    else:
        budget_scale = 1.0
    smooth_delta_list = [np.asarray(v * float(budget_scale), dtype=np.float64) for v in smooth_delta_list]

    for seq, delta_smooth in zip(z_seq_list, smooth_delta_list):
        arr = np.asarray(seq, dtype=np.float64)
        z_aug = arr + delta_smooth
        z_aug_list.append(np.asarray(z_aug, dtype=np.float32))
        delta_list.append(np.asarray(delta_smooth, dtype=np.float32))

    response_force_arr = np.asarray([float(r["response_force"]) for r in diagnostic_rows], dtype=np.float64)
    activation_driver_arr = np.asarray([float(r["activation_driver"]) for r in diagnostic_rows], dtype=np.float64)
    margin_arr = np.asarray([float(r["margin_before"]) for r in diagnostic_rows], dtype=np.float64)
    clip_applied_arr = np.asarray([1.0 if bool(r["clip_applied"]) else 0.0 for r in diagnostic_rows], dtype=np.float64)
    g_geom_arr = np.asarray([float(r["g_geom"]) for r in diagnostic_rows], dtype=np.float64)
    if response_force_arr.size >= 2 and np.std(response_force_arr) > 1e-12 and np.std(margin_arr) > 1e-12:
        response_vs_margin_correlation = float(np.corrcoef(response_force_arr, margin_arr)[0, 1])
    else:
        response_vs_margin_correlation = 0.0
    activation_coverage_ratio = float(np.mean(np.abs(activation_driver_arr) >= 1.0)) if activation_driver_arr.size else 0.0
    gate_saturation_ratio = float(np.mean(np.abs(response_force_arr) >= 0.95)) if response_force_arr.size else 0.0
    preactivation_clip_rate = float(np.mean(clip_applied_arr)) if clip_applied_arr.size else 0.0

    summary = _summarize_operator_application(
        original_z_seq_list=z_seq_list,
        z_seq_list=z_aug_list,
        delta_seq_list=delta_list,
        direction_seq_list=direction_list,
        diagnostics_rows=diagnostic_rows,
        fit_window_count=int(operator.pooled_window_count),
        fit_trial_count=int(operator.fit_trial_count),
        meta={
            "operator_arm": "continuous_geometric_force_field",
            "epsilon_scale": float(cfg.epsilon_scale),
            "smooth_lambda": float(cfg.smooth_lambda),
            "budget_scale_factor": float(budget_scale),
            "activation": str(operator.activation),
            "response_variant": str(response_variant),
            "fit_target_mode": str(operator.meta.get("fit_target_mode", "auto_associative")),
            "target_mode": str(operator.meta.get("target_mode", "")),
            "pool_mode": str(operator.meta.get("pool_mode", "same_only")),
            "opp_pair_rule": str(operator.meta.get("opp_pair_rule", "")),
            "template_count": int(operator.meta.get("template_count", 1)),
            "template_readout_mode": str(operator.meta.get("template_readout_mode", "first_row")),
            "response_stats_mode": str(operator.meta.get("response_stats_mode", "fit_pool")),
            "response_stats_window_count": int(operator.meta.get("response_stats_window_count", operator.pooled_window_count)),
            "response_scale": float(operator.response_scale),
            "response_scale_iqr": float(operator.response_scale_iqr),
            "response_vs_margin_correlation": float(response_vs_margin_correlation),
            "activation_coverage_ratio": float(activation_coverage_ratio),
            "preactivation_clip_rate": float(preactivation_clip_rate),
            "response_centering_std_after_fix": float(np.std(activation_driver_arr)) if activation_driver_arr.size else 0.0,
            "gate_saturation_ratio": float(gate_saturation_ratio),
            "geometry_coupling_mean": float(np.mean(g_geom_arr)) if g_geom_arr.size else 0.0,
            "geometry_coupling_abs_mean": float(np.mean(np.abs(g_geom_arr))) if g_geom_arr.size else 0.0,
            "fit_mode": str(operator.meta.get("fit_mode", "unweighted")),
            "template_mean_direction_cosine": float(operator.meta.get("template_mean_direction_cosine", 0.0)),
            "geometry_alignment_cosine_mean": float(operator.meta.get("geometry_alignment_cosine_mean", 0.0)),
            "weight_kernel_name": str(operator.meta.get("weight_kernel_name", operator.meta.get("weight_kernel", "identity"))),
            "effective_sample_size": float(operator.meta.get("effective_sample_size", 0.0)),
            "effective_sample_ratio": float(operator.meta.get("effective_sample_ratio", 0.0)),
            "min_proto_effective_sample_size": float(operator.meta.get("min_proto_effective_sample_size", 0.0)),
            "median_proto_effective_sample_size": float(operator.meta.get("median_proto_effective_sample_size", 0.0)),
            "fit_anchor_margin_mean": float(operator.meta.get("fit_anchor_margin_mean", 0.0)),
            "fit_anchor_same_dist_mean": float(operator.meta.get("fit_anchor_same_dist_mean", 0.0)),
            "proto_weight_scale_mean": float(operator.meta.get("proto_weight_scale_mean", 0.0)),
            "proto_weight_scale_min": float(operator.meta.get("proto_weight_scale_min", 0.0)),
            "same_pool_count": int(operator.meta.get("same_pool_count", 0)),
            "opp_pool_count": int(operator.meta.get("opp_pool_count", 0)),
            "same_weight_mass": float(operator.meta.get("same_weight_mass", 0.0)),
            "opp_weight_mass": float(operator.meta.get("opp_weight_mass", 0.0)),
            "same_opp_count_ratio": float(operator.meta.get("same_opp_count_ratio", 0.0)),
            "same_opp_weight_mass_ratio": float(operator.meta.get("same_opp_weight_mass_ratio", 0.0)),
            "same_proto_effective_sample_size": float(operator.meta.get("same_proto_effective_sample_size", 0.0)),
            "opp_proto_effective_sample_size": float(operator.meta.get("opp_proto_effective_sample_size", 0.0)),
            "discriminative_target_gap": float(operator.meta.get("discriminative_target_gap", 0.0)),
        },
    )
    return OperatorApplyResult(
        z_seq_list=z_aug_list,
        summary=summary,
        diagnostics_rows=diagnostic_rows,
        meta={
            "operator_arm": "continuous_geometric_force_field",
            "budget_scale_factor": float(budget_scale),
            "mean_direction_vector": _mean_direction_vector(direction_list).tolist(),
            **dict(operator.meta),
        },
    )
