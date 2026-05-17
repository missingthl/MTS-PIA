from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from core.trajectory_feedback_pool_windows import TrajectoryWindowReferenceStats


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


def _mean_pairwise_distance(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    tri = dists[np.triu_indices(arr.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else 0.0


def _nearest_dist(q: np.ndarray, reps: np.ndarray) -> np.ndarray:
    arr = np.asarray(reps, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError("reps must be non-empty [N, D]")
    q_arr = np.asarray(q, dtype=np.float64)
    return np.linalg.norm(arr - q_arr[None, :], axis=1)


@dataclass(frozen=True)
class SCPPrototypeMemoryConfig:
    knn_k: int = 5
    purity_quantile: float = 50.0
    continuity_quantile: float = 75.0
    prototype_count: int = 4
    cluster_mode: str = "kmeans_centroid"
    seed: int | None = None


@dataclass
class SCPPrototypeMemoryResult:
    candidate_rows: List[Dict[str, object]]
    safe_rows: List[Dict[str, object]]
    prototype_rows: List[Dict[str, object]]
    random_control_rows: List[Dict[str, object]]
    class_summary_rows: List[Dict[str, object]]
    structure_rows: List[Dict[str, object]]
    summary: Dict[str, object] = field(default_factory=dict)


def _fit_classwise_kmeans_prototypes(
    rows: Sequence[Dict[str, object]],
    *,
    prototype_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise ImportError("SCP prototype memory requires scikit-learn in the active environment.") from e

    x = np.stack([np.asarray(r["z_window"], dtype=np.float64) for r in rows], axis=0).astype(np.float64)
    k = int(max(1, min(int(prototype_count), int(x.shape[0]))))
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
    labels = km.fit_predict(x)
    return np.asarray(km.cluster_centers_, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _fit_classwise_random_control(
    rows: Sequence[Dict[str, object]],
    *,
    representative_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([np.asarray(r["z_window"], dtype=np.float64) for r in rows], axis=0).astype(np.float64)
    k = int(max(1, min(int(representative_count), int(x.shape[0]))))
    chosen = np.asarray(rng.choice(int(x.shape[0]), size=int(k), replace=False), dtype=np.int64)
    reps = np.asarray(x[chosen], dtype=np.float64)
    assign = []
    for row in x:
        dists = np.linalg.norm(reps - row[None, :], axis=1)
        assign.append(int(np.argmin(dists)))
    return reps, np.asarray(assign, dtype=np.int64), chosen


def _build_structure_rows(
    *,
    mode: str,
    safe_rows_by_class: Dict[int, List[Dict[str, object]]],
    reps_by_class: Dict[int, np.ndarray],
    assignments_by_class: Dict[int, np.ndarray],
) -> Dict[str, float]:
    within_rows: List[float] = []
    between_rows: List[float] = []
    margin_rows: List[float] = []
    stability_rows: List[float] = []

    # Within-class compactness.
    for cls, rows in safe_rows_by_class.items():
        reps = np.asarray(reps_by_class[int(cls)], dtype=np.float64)
        assn = np.asarray(assignments_by_class[int(cls)], dtype=np.int64)
        if reps.ndim != 2 or reps.shape[0] <= 0:
            continue
        for row, proto_id in zip(rows, assn.tolist()):
            z = np.asarray(row["z_window"], dtype=np.float64)
            within_rows.append(float(np.linalg.norm(z - reps[int(proto_id)])))

    # Between-class separation and nearest-prototype margin.
    all_proto_rows: List[tuple[int, np.ndarray]] = []
    for cls, reps in reps_by_class.items():
        for rep in np.asarray(reps, dtype=np.float64):
            all_proto_rows.append((int(cls), np.asarray(rep, dtype=np.float64)))
    for i, (cls_i, rep_i) in enumerate(all_proto_rows):
        for cls_j, rep_j in all_proto_rows[i + 1 :]:
            if int(cls_i) == int(cls_j):
                continue
            between_rows.append(float(np.linalg.norm(rep_i - rep_j)))

    class_ids = sorted(int(k) for k in reps_by_class)
    for cls, rows in safe_rows_by_class.items():
        same_reps = np.asarray(reps_by_class[int(cls)], dtype=np.float64)
        diff_reps = np.concatenate(
            [np.asarray(reps_by_class[int(other)], dtype=np.float64) for other in class_ids if int(other) != int(cls)],
            axis=0,
        )
        for row in rows:
            z = np.asarray(row["z_window"], dtype=np.float64)
            same_dist = float(np.min(_nearest_dist(z, same_reps)))
            diff_dist = float(np.min(_nearest_dist(z, diff_reps))) if diff_reps.size > 0 else same_dist
            margin_rows.append(float(diff_dist - same_dist))

    # Temporal assignment stability: adjacent safe windows in the same trial keeping the same prototype id.
    by_trial: Dict[str, List[tuple[int, int]]] = {}
    for cls, rows in safe_rows_by_class.items():
        assn = np.asarray(assignments_by_class[int(cls)], dtype=np.int64)
        for row, proto_id in zip(rows, assn.tolist()):
            by_trial.setdefault(str(row["trial_id"]), []).append((int(row["window_index"]), int(proto_id)))
    for rows in by_trial.values():
        rows = sorted(rows, key=lambda x: x[0])
        if len(rows) <= 1:
            continue
        same = 0
        total = 0
        for (w0, p0), (w1, p1) in zip(rows[:-1], rows[1:]):
            if int(w1) != int(w0) + 1:
                continue
            total += 1
            same += int(int(p0) == int(p1))
        if total > 0:
            stability_rows.append(float(same / total))

    return {
        "mode": str(mode),
        "within_prototype_compactness": float(np.mean(within_rows)) if within_rows else 0.0,
        "between_prototype_separation": float(np.mean(between_rows)) if between_rows else 0.0,
        "nearest_prototype_margin": float(np.mean(margin_rows)) if margin_rows else 0.0,
        "temporal_assignment_stability": float(np.mean(stability_rows)) if stability_rows else 0.0,
        "prototype_family_dispersion": float(
            _mean_pairwise_distance(
                np.stack([rep for _cls, rep in all_proto_rows], axis=0).astype(np.float64)
            )
        )
        if len(all_proto_rows) >= 2
        else 0.0,
    }


def build_scp_prototype_memory(
    *,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    reference_stats: TrajectoryWindowReferenceStats,
    cfg: SCPPrototypeMemoryConfig,
) -> SCPPrototypeMemoryResult:
    tids = [str(v) for v in train_tids]
    labels = [int(v) for v in train_labels]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    if len(tids) != len(labels) or len(tids) != len(seqs):
        raise ValueError("train inputs must align")
    if not seqs:
        raise ValueError("train_z_seq_list cannot be empty")
    if str(cfg.cluster_mode).strip().lower() != "kmeans_centroid":
        raise ValueError("SCP-Branch v0 locks cluster_mode to kmeans_centroid")

    candidate_rows: List[Dict[str, object]] = []

    for trial_idx, (tid, cls, seq) in enumerate(zip(tids, labels, seqs)):
        start = int(reference_stats.trial_offsets[int(trial_idx)])
        for window_index in range(int(seq.shape[0])):
            ref_index = int(start + int(window_index))
            z = np.asarray(seq[int(window_index)], dtype=np.float64)
            local_step = float(_window_local_step_mean(seq, int(window_index)))
            candidate_rows.append(
                {
                    "trial_id": str(tid),
                    "label": int(cls),
                    "window_index": int(window_index),
                    "ref_index": int(ref_index),
                    "z_window": np.asarray(z, dtype=np.float32),
                    "local_step_orig": float(local_step),
                    "safe": True,
                    "reject_reason": "",
                }
            )
    # SCP-Branch v0 intentionally avoids a heavier safety gate and measures whether
    # prototype compression itself yields distribution-supported local anchors that are
    # better structured than a same-size random control.
    safe_rows = [dict(row) for row in candidate_rows]
    safe_rows_by_class: Dict[int, List[Dict[str, object]]] = {}
    for row in safe_rows:
        safe_rows_by_class.setdefault(int(row["label"]), []).append(dict(row))

    rng = np.random.default_rng(None if cfg.seed is None else int(cfg.seed))
    prototype_rows: List[Dict[str, object]] = []
    random_control_rows: List[Dict[str, object]] = []
    class_summary_rows: List[Dict[str, object]] = []
    prototype_reps_by_class: Dict[int, np.ndarray] = {}
    random_reps_by_class: Dict[int, np.ndarray] = {}
    proto_assign_by_class: Dict[int, np.ndarray] = {}
    rand_assign_by_class: Dict[int, np.ndarray] = {}

    class_ids = sorted(set(labels))
    for cls in class_ids:
        rows = safe_rows_by_class.get(int(cls), [])
        candidate_count = int(sum(int(r["label"]) == int(cls) for r in candidate_rows))
        safe_count = int(len(rows))
        coverage_threshold = int(max(8, int(np.ceil(0.05 * max(0, candidate_count)))))
        low_coverage_flag = int(safe_count < coverage_threshold)
        prototype_count = int(max(0, min(int(cfg.prototype_count), int(safe_count))))
        if prototype_count > 0:
            reps, assign = _fit_classwise_kmeans_prototypes(
                rows,
                prototype_count=int(prototype_count),
                seed=int(0 if cfg.seed is None else cfg.seed) + int(cls),
            )
            prototype_reps_by_class[int(cls)] = np.asarray(reps, dtype=np.float64)
            proto_assign_by_class[int(cls)] = np.asarray(assign, dtype=np.int64)
            for proto_id, rep in enumerate(reps.tolist()):
                member_count = int(np.sum(np.asarray(assign, dtype=np.int64) == int(proto_id)))
                prototype_rows.append(
                    {
                        "class_id": int(cls),
                        "prototype_id": int(proto_id),
                        "prototype_member_count": int(member_count),
                        "prototype_source": "kmeans_centroid",
                        "prototype_vector_norm": float(np.linalg.norm(np.asarray(rep, dtype=np.float64))),
                    }
                )

            rand_reps, rand_assign, chosen = _fit_classwise_random_control(
                rows,
                representative_count=int(prototype_count),
                rng=rng,
            )
            random_reps_by_class[int(cls)] = np.asarray(rand_reps, dtype=np.float64)
            rand_assign_by_class[int(cls)] = np.asarray(rand_assign, dtype=np.int64)
            for proto_id, row_index in enumerate(chosen.tolist()):
                picked = rows[int(row_index)]
                random_control_rows.append(
                    {
                        "class_id": int(cls),
                        "random_memory_id": int(proto_id),
                        "trial_id": str(picked["trial_id"]),
                        "window_index": int(picked["window_index"]),
                        "vector_norm": float(np.linalg.norm(np.asarray(picked["z_window"], dtype=np.float64))),
                    }
                )
        else:
            prototype_reps_by_class[int(cls)] = np.zeros((0, seqs[0].shape[1]), dtype=np.float64)
            random_reps_by_class[int(cls)] = np.zeros((0, seqs[0].shape[1]), dtype=np.float64)
            proto_assign_by_class[int(cls)] = np.zeros((0,), dtype=np.int64)
            rand_assign_by_class[int(cls)] = np.zeros((0,), dtype=np.int64)

        class_summary_rows.append(
            {
                "class_id": int(cls),
                "candidate_window_count_class": int(candidate_count),
                "safe_window_count_class": int(safe_count),
                "prototype_count_class": int(prototype_count),
                "coverage_ratio_class": float(safe_count / max(1, candidate_count)),
                "coverage_threshold_class": int(coverage_threshold),
                "low_coverage_flag": int(low_coverage_flag),
                "effective_trigger": int(0 if low_coverage_flag else 1),
                "purity_threshold_class": np.nan,
                "local_step_threshold_class": np.nan,
            }
        )

    prototype_diag = _build_structure_rows(
        mode="prototype_memory",
        safe_rows_by_class=safe_rows_by_class,
        reps_by_class=prototype_reps_by_class,
        assignments_by_class=proto_assign_by_class,
    )
    random_diag = _build_structure_rows(
        mode="random_memory_control",
        safe_rows_by_class=safe_rows_by_class,
        reps_by_class=random_reps_by_class,
        assignments_by_class=rand_assign_by_class,
    )

    structure_rows = [
        dict(prototype_diag),
        dict(random_diag),
    ]
    summary = {
        "candidate_window_count": int(len(candidate_rows)),
        "safe_window_count": int(len(safe_rows)),
        "safe_ratio": float(len(safe_rows) / max(1, len(candidate_rows))),
        "prototype_class_count": int(sum(1 for r in class_summary_rows if int(r["prototype_count_class"]) > 0)),
        "low_coverage_class_count": int(sum(int(r["low_coverage_flag"]) for r in class_summary_rows)),
        "low_coverage_class_rate": float(np.mean([float(r["low_coverage_flag"]) for r in class_summary_rows]))
        if class_summary_rows
        else 0.0,
        "cluster_mode": "kmeans_centroid",
        "comparison_with_random": {
            "within_compactness_delta": float(random_diag["within_prototype_compactness"] - prototype_diag["within_prototype_compactness"]),
            "between_separation_delta": float(prototype_diag["between_prototype_separation"] - random_diag["between_prototype_separation"]),
            "nearest_margin_delta": float(prototype_diag["nearest_prototype_margin"] - random_diag["nearest_prototype_margin"]),
            "temporal_stability_delta": float(
                prototype_diag["temporal_assignment_stability"] - random_diag["temporal_assignment_stability"]
            ),
        },
        "scope_note": (
            "SCP-Branch v0 measures whether a prototype-memory object can recover reproducible local representative "
            "states under the dense z_seq backbone, rather than searching for a presumed global physiological center. "
            "It compares those distribution-supported local anchors against a same-size random window set and does not "
            "yet include replay, curriculum, PIA-guided geometry, or a heavier safety gate."
        ),
    }
    return SCPPrototypeMemoryResult(
        candidate_rows=candidate_rows,
        safe_rows=safe_rows,
        prototype_rows=prototype_rows,
        random_control_rows=random_control_rows,
        class_summary_rows=class_summary_rows,
        structure_rows=structure_rows,
        summary=summary,
    )
