from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Sequence

import numpy as np

from core.scp_local_shaping import SCPLocalShapingConfig, apply_scp_local_shaping


def _fit_classwise_kmeans(rows: Sequence[Dict[str, object]], *, prototype_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    x = np.stack([np.asarray(r["z_window"], dtype=np.float64) for r in rows], axis=0).astype(np.float64)
    k = int(max(1, min(int(prototype_count), int(x.shape[0]))))
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
    labels = km.fit_predict(x)
    return np.asarray(km.cluster_centers_, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _nearest_dist(q: np.ndarray, reps: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(reps, dtype=np.float64) - np.asarray(q, dtype=np.float64)[None, :], axis=1)


def _mean_pairwise_distance(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    tri = dists[np.triu_indices(arr.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else 0.0


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom <= 1e-12:
        return 1.0
    return float(np.clip(np.dot(a_arr, b_arr) / denom, -1.0, 1.0))


def _structure_from_assignments(
    *,
    rows_by_class: Dict[int, List[Dict[str, object]]],
    reps_by_class: Dict[int, np.ndarray],
    assignments_by_class: Dict[int, np.ndarray],
) -> Dict[str, float]:
    within_rows: List[float] = []
    between_rows: List[float] = []
    margin_rows: List[float] = []
    stability_rows: List[float] = []

    for cls, rows in rows_by_class.items():
        reps = np.asarray(reps_by_class[int(cls)], dtype=np.float64)
        assn = np.asarray(assignments_by_class[int(cls)], dtype=np.int64)
        for row, proto_id in zip(rows, assn.tolist()):
            z = np.asarray(row["z_window"], dtype=np.float64)
            within_rows.append(float(np.linalg.norm(z - reps[int(proto_id)])))

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
    for cls, rows in rows_by_class.items():
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

    by_trial: Dict[str, List[tuple[int, int]]] = {}
    for cls, rows in rows_by_class.items():
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
        "within_prototype_compactness": float(np.mean(within_rows)) if within_rows else 0.0,
        "between_prototype_separation": float(np.mean(between_rows)) if between_rows else 0.0,
        "nearest_prototype_margin": float(np.mean(margin_rows)) if margin_rows else 0.0,
        "temporal_assignment_stability": float(np.mean(stability_rows)) if stability_rows else 0.0,
        "prototype_family_dispersion": float(_mean_pairwise_distance(np.stack([rep for _cls, rep in all_proto_rows], axis=0)))
        if len(all_proto_rows) >= 2
        else 0.0,
    }


def _prototype_local_metrics(
    *,
    candidate_proto: np.ndarray,
    member_rows: Sequence[Dict[str, object]],
    other_reps: np.ndarray,
) -> Dict[str, float]:
    proto = np.asarray(candidate_proto, dtype=np.float64)
    within_vals: List[float] = []
    margin_vals: List[float] = []
    for row in member_rows:
        z = np.asarray(row["z_window"], dtype=np.float64)
        same_dist = float(np.linalg.norm(z - proto))
        diff_dist = float(np.min(_nearest_dist(z, other_reps))) if other_reps.size > 0 else same_dist
        within_vals.append(same_dist)
        margin_vals.append(float(diff_dist - same_dist))
    between_val = float(np.min(_nearest_dist(proto, other_reps))) if other_reps.size > 0 else 0.0
    return {
        "within": float(np.mean(within_vals)) if within_vals else 0.0,
        "margin": float(np.mean(margin_vals)) if margin_vals else 0.0,
        "between": float(between_val),
    }


@dataclass(frozen=True)
class SCPClosedFormUpdateConfig:
    prototype_count: int = 4
    anchors_per_prototype: int = 8
    anchor_selection_mode: str = "tight_margin"
    same_dist_quantile: float = 50.0
    beta: float = 0.5
    epsilon_scale: float = 0.10
    proto_update_alpha: float = 0.2
    candidate_refresh_mode: str = "anchor_mean"
    trial_m_min: int = 4
    trial_proto_min: int = 2
    seed: int | None = None


@dataclass
class SCPClosedFormUpdateResult:
    updated_structure: Dict[str, float]
    prototype_rows: List[Dict[str, object]]
    direction_rows: List[Dict[str, object]]
    acceptance_rows: List[Dict[str, object]]
    trial_anchor_rows: List[Dict[str, object]]
    summary: Dict[str, object] = field(default_factory=dict)


def _trial_proto_candidates(
    *,
    base_proto: np.ndarray,
    member_rows: Sequence[Dict[str, object]],
    trial_m_min: int,
) -> List[Dict[str, object]]:
    by_trial: Dict[str, List[np.ndarray]] = {}
    for row in member_rows:
        by_trial.setdefault(str(row["trial_id"]), []).append(np.asarray(row["z_window"], dtype=np.float64))
    out: List[Dict[str, object]] = []
    for trial_id, vectors in sorted(by_trial.items()):
        support_count = int(len(vectors))
        if support_count < int(trial_m_min):
            continue
        mat = np.stack(vectors, axis=0).astype(np.float64)
        trial_mean = np.mean(mat, axis=0).astype(np.float64)
        within_dispersion = float(np.mean(np.linalg.norm(mat - trial_mean[None, :], axis=1)))
        out.append(
            {
                "trial_id": str(trial_id),
                "trial_proto_support_count": int(support_count),
                "trial_proto_mean": np.asarray(trial_mean, dtype=np.float64),
                "trial_proto_mean_shift_norm": float(np.linalg.norm(trial_mean - np.asarray(base_proto, dtype=np.float64))),
                "trial_proto_within_dispersion": float(within_dispersion),
            }
        )
    return out


def run_scp_closed_form_update(
    *,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    cfg: SCPClosedFormUpdateConfig,
) -> SCPClosedFormUpdateResult:
    t0 = perf_counter()
    tids = [str(v) for v in train_tids]
    labels = [int(v) for v in train_labels]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]

    rows_by_class: Dict[int, List[Dict[str, object]]] = {}
    for tid, cls, seq in zip(tids, labels, seqs):
        for window_index in range(int(seq.shape[0])):
            rows_by_class.setdefault(int(cls), []).append(
                {
                    "trial_id": str(tid),
                    "label": int(cls),
                    "window_index": int(window_index),
                    "z_window": np.asarray(seq[int(window_index)], dtype=np.float32),
                }
            )

    base_reps_by_class: Dict[int, np.ndarray] = {}
    assignments_by_class: Dict[int, np.ndarray] = {}
    for cls, rows in rows_by_class.items():
        reps, assign = _fit_classwise_kmeans(rows, prototype_count=int(cfg.prototype_count), seed=int(0 if cfg.seed is None else cfg.seed) + int(cls))
        base_reps_by_class[int(cls)] = np.asarray(reps, dtype=np.float64)
        assignments_by_class[int(cls)] = np.asarray(assign, dtype=np.int64)

    shaping = apply_scp_local_shaping(
        train_tids=tids,
        train_labels=labels,
        train_z_seq_list=seqs,
        cfg=SCPLocalShapingConfig(
            prototype_count=int(cfg.prototype_count),
            anchors_per_prototype=int(cfg.anchors_per_prototype),
            anchor_selection_mode=str(cfg.anchor_selection_mode),
            same_dist_quantile=float(cfg.same_dist_quantile),
            beta=float(cfg.beta),
            epsilon_scale=float(cfg.epsilon_scale),
            seed=int(0 if cfg.seed is None else cfg.seed),
        ),
    )
    anchor_key = {(str(r["trial_id"]), int(r["window_index"])) for r in shaping.anchor_rows}

    updated_reps_by_class: Dict[int, np.ndarray] = {}
    prototype_rows: List[Dict[str, object]] = []
    direction_rows: List[Dict[str, object]] = []
    acceptance_rows: List[Dict[str, object]] = []
    trial_anchor_rows: List[Dict[str, object]] = []

    for cls, rows in rows_by_class.items():
        reps = np.asarray(base_reps_by_class[int(cls)], dtype=np.float64)
        assn = np.asarray(assignments_by_class[int(cls)], dtype=np.int64)
        updated_reps = np.asarray(reps, dtype=np.float64).copy()
        other_base_parts = [np.asarray(base_reps_by_class[int(other)], dtype=np.float64) for other in sorted(base_reps_by_class) if int(other) != int(cls)]
        other_base_reps = np.concatenate(other_base_parts, axis=0) if other_base_parts else np.zeros((0, reps.shape[1]), dtype=np.float64)
        for proto_id in range(int(reps.shape[0])):
            member_ids = np.where(assn == int(proto_id))[0]
            if member_ids.size <= 0:
                continue
            member_rows = [rows[int(i)] for i in member_ids.tolist()]
            base_proto = np.asarray(reps[int(proto_id)], dtype=np.float64)
            anchor_vectors = [
                np.asarray(r["z_window"], dtype=np.float64)
                for r in member_rows
                if (str(r["trial_id"]), int(r["window_index"])) in anchor_key
            ]
            refresh_mode = str(cfg.candidate_refresh_mode).strip().lower()
            skip_refresh = 0
            effective_trial_count = 0
            trial_anchor_target = None
            if refresh_mode == "trial_proto_mean":
                trial_rows = _trial_proto_candidates(
                    base_proto=base_proto,
                    member_rows=member_rows,
                    trial_m_min=int(cfg.trial_m_min),
                )
                for row in trial_rows:
                    trial_anchor_rows.append(
                        {
                            "class_id": int(cls),
                            "prototype_id": int(proto_id),
                            **{k: (v.tolist() if k == "trial_proto_mean" else v) for k, v in row.items()},
                        }
                    )
                effective_trial_count = int(len(trial_rows))
                if effective_trial_count >= int(cfg.trial_proto_min):
                    trial_anchor_target = np.mean(
                        np.stack([np.asarray(r["trial_proto_mean"], dtype=np.float64) for r in trial_rows], axis=0).astype(np.float64),
                        axis=0,
                    ).astype(np.float64)
                    candidate = (1.0 - float(cfg.proto_update_alpha)) * base_proto + float(cfg.proto_update_alpha) * trial_anchor_target
                else:
                    skip_refresh = 1
                    candidate = np.asarray(base_proto, dtype=np.float64)
            else:
                if anchor_vectors:
                    anchor_mean = np.mean(np.stack(anchor_vectors, axis=0).astype(np.float64), axis=0).astype(np.float64)
                    candidate = (1.0 - float(cfg.proto_update_alpha)) * base_proto + float(cfg.proto_update_alpha) * anchor_mean
                else:
                    skip_refresh = 1
                    candidate = np.asarray(base_proto, dtype=np.float64)

            base_metrics = _prototype_local_metrics(
                candidate_proto=base_proto,
                member_rows=member_rows,
                other_reps=other_base_reps,
            )
            candidate_metrics = _prototype_local_metrics(
                candidate_proto=candidate,
                member_rows=member_rows,
                other_reps=other_base_reps,
            )

            delta_between = float(candidate_metrics["between"] - base_metrics["between"])
            delta_margin = float(candidate_metrics["margin"] - base_metrics["margin"])
            delta_within = float(candidate_metrics["within"] - base_metrics["within"])
            accept_between = int(delta_between > 0.0)
            accept_margin = int(delta_margin >= 0.0)
            accept_within = int(delta_within <= 0.0)
            final_accept = int(bool(accept_between and accept_margin and accept_within))
            updated = np.asarray(candidate if final_accept else base_proto, dtype=np.float64)

            member_vectors = np.stack([np.asarray(r["z_window"], dtype=np.float64) for r in member_rows], axis=0).astype(np.float64)
            member_dists = np.linalg.norm(member_vectors - updated[None, :], axis=1)
            medoid_idx = int(np.argmin(member_dists))
            medoid = np.asarray(member_vectors[medoid_idx], dtype=np.float64)
            updated_reps[int(proto_id)] = np.asarray(updated, dtype=np.float64)
            acceptance_rows.append(
                {
                    "class_id": int(cls),
                    "prototype_id": int(proto_id),
                    "prototype_member_count": int(member_ids.size),
                    "anchor_count": int(len(anchor_vectors)),
                    "refresh_mode": str(refresh_mode),
                    "trial_proto_effective_count": int(effective_trial_count),
                    "skip_refresh": int(skip_refresh),
                    "base_between": float(base_metrics["between"]),
                    "candidate_between": float(candidate_metrics["between"]),
                    "delta_between": float(delta_between),
                    "base_margin": float(base_metrics["margin"]),
                    "candidate_margin": float(candidate_metrics["margin"]),
                    "delta_margin": float(delta_margin),
                    "base_within": float(base_metrics["within"]),
                    "candidate_within": float(candidate_metrics["within"]),
                    "delta_within": float(delta_within),
                    "accept_between": int(accept_between),
                    "accept_margin": int(accept_margin),
                    "accept_within": int(accept_within),
                    "final_accept": int(final_accept),
                }
            )
            prototype_rows.append(
                {
                    "class_id": int(cls),
                    "prototype_id": int(proto_id),
                    "refresh_mode": str(refresh_mode),
                    "original_norm": float(np.linalg.norm(base_proto)),
                    "updated_norm": float(np.linalg.norm(updated)),
                    "prototype_shift_norm": float(np.linalg.norm(updated - base_proto)),
                    "medoid_refresh_dist": float(np.linalg.norm(medoid - updated)),
                    "trial_proto_effective_count": int(effective_trial_count),
                    "final_accept": int(final_accept),
                }
            )
        updated_reps_by_class[int(cls)] = np.asarray(updated_reps, dtype=np.float64)

    for cls, reps in updated_reps_by_class.items():
        for proto_id, rep in enumerate(np.asarray(reps, dtype=np.float64)):
            old_rep = np.asarray(base_reps_by_class[int(cls)][int(proto_id)], dtype=np.float64)
            old_opp_rows = [
                (other_cls, other_proto_id, other_rep)
                for other_cls, other_reps in base_reps_by_class.items()
                for other_proto_id, other_rep in enumerate(np.asarray(other_reps, dtype=np.float64))
                if int(other_cls) != int(cls)
            ]
            opp_rows = [
                (other_cls, other_proto_id, other_rep)
                for other_cls, other_reps in updated_reps_by_class.items()
                for other_proto_id, other_rep in enumerate(np.asarray(other_reps, dtype=np.float64))
                if int(other_cls) != int(cls)
            ]
            if not opp_rows:
                continue
            old_opp_dists = np.asarray(
                [float(np.linalg.norm(old_rep - np.asarray(other_rep, dtype=np.float64))) for _c, _pid, other_rep in old_opp_rows],
                dtype=np.float64,
            )
            opp_dists = np.asarray([float(np.linalg.norm(rep - np.asarray(other_rep, dtype=np.float64))) for _c, _pid, other_rep in opp_rows], dtype=np.float64)
            best_old_opp = int(np.argmin(old_opp_dists)) if old_opp_dists.size else 0
            best_opp = int(np.argmin(opp_dists))
            old_opp_cls, old_opp_proto_id, old_opp_rep = old_opp_rows[best_old_opp]
            opp_cls, opp_proto_id, opp_rep = opp_rows[best_opp]
            old_direction = np.asarray(old_rep - np.asarray(old_opp_rep, dtype=np.float64), dtype=np.float64)
            direction = np.asarray(rep - np.asarray(opp_rep, dtype=np.float64), dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            direction_cosine_to_old = float(_safe_cosine(direction, old_direction))
            direction_rows.append(
                {
                    "class_id": int(cls),
                    "prototype_id": int(proto_id),
                    "opp_class_id": int(opp_cls),
                    "opp_prototype_id": int(opp_proto_id),
                    "old_opp_class_id": int(old_opp_cls),
                    "old_opp_prototype_id": int(old_opp_proto_id),
                    "direction_norm": float(norm),
                    "pair_distance": float(opp_dists[best_opp]),
                    "direction_cosine_to_old": float(direction_cosine_to_old),
                    "direction_change_angle_proxy": float(np.sqrt(max(0.0, 1.0 - direction_cosine_to_old**2))),
                }
            )

    updated_structure = _structure_from_assignments(
        rows_by_class=rows_by_class,
        reps_by_class=updated_reps_by_class,
        assignments_by_class=assignments_by_class,
    )

    summary = {
        "update_time_seconds": float(perf_counter() - t0),
        "prototype_shift_norm_mean": float(np.mean([float(r["prototype_shift_norm"]) for r in prototype_rows])) if prototype_rows else 0.0,
        "medoid_refresh_dist_mean": float(np.mean([float(r["medoid_refresh_dist"]) for r in prototype_rows])) if prototype_rows else 0.0,
        "direction_norm_mean": float(np.mean([float(r["direction_norm"]) for r in direction_rows])) if direction_rows else 0.0,
        "direction_cosine_to_old_mean": float(np.mean([float(r["direction_cosine_to_old"]) for r in direction_rows])) if direction_rows else 1.0,
        "direction_change_angle_proxy_mean": float(np.mean([float(r["direction_change_angle_proxy"]) for r in direction_rows])) if direction_rows else 0.0,
        "accept_between_rate": float(np.mean([float(r["accept_between"]) for r in acceptance_rows])) if acceptance_rows else 0.0,
        "accept_margin_rate": float(np.mean([float(r["accept_margin"]) for r in acceptance_rows])) if acceptance_rows else 0.0,
        "accept_within_rate": float(np.mean([float(r["accept_within"]) for r in acceptance_rows])) if acceptance_rows else 0.0,
        "final_accept_rate": float(np.mean([float(r["final_accept"]) for r in acceptance_rows])) if acceptance_rows else 0.0,
        "skip_refresh_rate": float(np.mean([float(r["skip_refresh"]) for r in acceptance_rows])) if acceptance_rows else 0.0,
        "trial_proto_effective_count_mean": float(np.mean([float(r["trial_proto_effective_count"]) for r in acceptance_rows])) if acceptance_rows else 0.0,
        "trial_proto_within_dispersion_mean": float(np.mean([float(r["trial_proto_within_dispersion"]) for r in trial_anchor_rows])) if trial_anchor_rows else 0.0,
        "scope_note": (
            "SCP-Branch v2 performs a train-only closed-form local geometry refresh over prototype-memory objects. "
            "It does not update the terminal or backbone and is evaluated only by structure/cost diagnostics with balanced acceptance."
        ),
    }
    return SCPClosedFormUpdateResult(
        updated_structure=updated_structure,
        prototype_rows=prototype_rows,
        direction_rows=direction_rows,
        acceptance_rows=acceptance_rows,
        trial_anchor_rows=trial_anchor_rows,
        summary=summary,
    )
