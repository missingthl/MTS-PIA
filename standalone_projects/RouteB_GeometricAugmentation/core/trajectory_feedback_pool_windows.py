from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from core.trajectory_pia_operator import TrajectoryPIAOperator


def _class_balance_proxy(labels: Sequence[int]) -> float:
    arr = np.asarray(list(labels), dtype=np.int64)
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    if counts.size == 0 or int(np.max(counts)) <= 0:
        return 0.0
    return float(np.min(counts) / np.max(counts))


def _knn_majority_and_purity(
    query: np.ndarray,
    *,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    k: int,
    exclude_index: int | None,
    fallback_label: int,
) -> tuple[int, float]:
    emb = np.asarray(ref_embeddings, dtype=np.float64)
    labels = np.asarray(ref_labels, dtype=np.int64)
    if emb.ndim != 2 or emb.shape[0] != labels.shape[0]:
        raise ValueError("reference embeddings/labels shape mismatch")
    if emb.shape[0] <= 0:
        return int(fallback_label), 1.0

    dists = np.linalg.norm(emb - np.asarray(query, dtype=np.float64)[None, :], axis=1)
    order = np.argsort(dists, kind="mergesort")
    if exclude_index is not None:
        order = order[order != int(exclude_index)]
    if order.size <= 0:
        return int(fallback_label), 1.0

    k_eff = int(max(1, min(int(k), int(order.size))))
    nn_labels = labels[order[:k_eff]]
    values, counts = np.unique(nn_labels, return_counts=True)
    if values.size <= 0:
        return int(fallback_label), 1.0
    best_idx = int(np.argmax(counts))
    majority = int(values[best_idx])
    purity = float(counts[best_idx] / max(1, int(nn_labels.size)))
    return int(majority), float(purity)


def _window_local_step_mean(z_seq: np.ndarray, idx: int) -> float:
    arr = np.asarray(z_seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError("z_seq must be [K, D] with K>0")
    k = int(arr.shape[0])
    if k <= 1:
        return 0.0
    # Edge padding keeps the continuity definition compatible with trajectory-level T3/T4a,
    # but localized to the window-centered neighborhood.
    prev_idx = max(0, int(idx) - 1)
    next_idx = min(k - 1, int(idx) + 1)
    prev_step = float(np.linalg.norm(arr[int(idx)] - arr[int(prev_idx)]))
    next_step = float(np.linalg.norm(arr[int(next_idx)] - arr[int(idx)]))
    return float(0.5 * (prev_step + next_step))


def _compute_margin(query: np.ndarray, *, true_class: int, class_centers: Dict[int, np.ndarray]) -> float:
    if not class_centers:
        return 0.0
    q = np.asarray(query, dtype=np.float64)
    cls = int(true_class)
    if cls not in class_centers:
        raise KeyError(f"missing class center for class {cls}")
    d_pos = float(np.linalg.norm(q - np.asarray(class_centers[int(cls)], dtype=np.float64)))
    other_dists = [
        float(np.linalg.norm(q - np.asarray(center, dtype=np.float64)))
        for c, center in class_centers.items()
        if int(c) != int(cls)
    ]
    if not other_dists:
        return 0.0
    d_neg = float(np.min(np.asarray(other_dists, dtype=np.float64)))
    return float(d_neg - d_pos)


def _compute_local_knn_margin(
    query: np.ndarray,
    *,
    true_class: int,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    k: int,
    exclude_index: int | None,
) -> float:
    emb = np.asarray(ref_embeddings, dtype=np.float64)
    labels = np.asarray(ref_labels, dtype=np.int64)
    if emb.ndim != 2 or emb.shape[0] != labels.shape[0]:
        raise ValueError("reference embeddings/labels shape mismatch")
    if emb.shape[0] <= 1:
        return 0.0

    q = np.asarray(query, dtype=np.float64).reshape(1, -1)
    dists = np.linalg.norm(emb - q, axis=1)
    cls = int(true_class)

    same_mask = labels == int(cls)
    # Same-class local margin must exclude the current window itself.
    if exclude_index is not None and 0 <= int(exclude_index) < int(labels.shape[0]):
        same_mask[int(exclude_index)] = False
    diff_mask = labels != int(cls)

    same_dists = np.sort(dists[same_mask], kind="mergesort")
    diff_dists = np.sort(dists[diff_mask], kind="mergesort")
    if same_dists.size <= 0 or diff_dists.size <= 0:
        return 0.0

    k_same = int(max(1, min(int(k), int(same_dists.size))))
    k_diff = int(max(1, min(int(k), int(diff_dists.size))))
    mean_same = float(np.mean(same_dists[:k_same]))
    mean_diff = float(np.mean(diff_dists[:k_diff]))
    return float(mean_diff - mean_same)


@dataclass(frozen=True)
class TrajectoryWindowReferenceStats:
    ref_embeddings: np.ndarray
    ref_labels: np.ndarray
    class_centers: Dict[int, np.ndarray]
    trial_offsets: List[int]
    total_windows: int


@dataclass(frozen=True)
class TrajectoryWindowFeedbackPoolConfig:
    gamma_main: float = 0.05
    smooth_lambda: float = 0.50
    knn_k: int = 5
    max_purity_drop: float = 0.10
    continuity_quantile: float = 75.0
    informative_gate: str = "safety_only"  # safety_only | radial | margin | local_knn_margin


@dataclass
class TrajectoryWindowFeedbackPoolResult:
    accepted_window_seq_list: List[np.ndarray]
    accepted_labels: List[int]
    accepted_window_rows: List[Dict[str, object]]
    candidate_rows: List[Dict[str, object]]
    class_coverage_rows: List[Dict[str, object]] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)


def build_window_feedback_reference_stats(
    *,
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
) -> TrajectoryWindowReferenceStats:
    if len(train_labels) != len(train_z_seq_list):
        raise ValueError("train_labels and train_z_seq_list must align")
    ref_rows: List[np.ndarray] = []
    ref_labels: List[int] = []
    trial_offsets: List[int] = []
    offset = 0
    per_class_rows: Dict[int, List[np.ndarray]] = {}
    for cls, seq in zip(train_labels, train_z_seq_list):
        arr = np.asarray(seq, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            raise ValueError("each train z_seq must be [K, D] with K>0")
        trial_offsets.append(int(offset))
        ref_rows.append(arr)
        ref_labels.extend([int(cls)] * int(arr.shape[0]))
        per_class_rows.setdefault(int(cls), []).append(arr)
        offset += int(arr.shape[0])
    x = np.concatenate(ref_rows, axis=0).astype(np.float64)
    y = np.asarray(ref_labels, dtype=np.int64)
    class_centers = {
        int(cls): np.mean(np.concatenate(rows, axis=0).astype(np.float64), axis=0).astype(np.float64)
        for cls, rows in per_class_rows.items()
    }
    return TrajectoryWindowReferenceStats(
        ref_embeddings=np.asarray(x, dtype=np.float64),
        ref_labels=np.asarray(y, dtype=np.int64),
        class_centers=class_centers,
        trial_offsets=trial_offsets,
        total_windows=int(x.shape[0]),
    )


def build_window_feedback_pool(
    *,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    operator: TrajectoryPIAOperator,
    reference_stats: TrajectoryWindowReferenceStats,
    cfg: TrajectoryWindowFeedbackPoolConfig,
) -> TrajectoryWindowFeedbackPoolResult:
    tids = [str(v) for v in train_tids]
    labels = [int(v) for v in train_labels]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    if len(tids) != len(labels) or len(tids) != len(seqs):
        raise ValueError("train inputs must align")
    if not seqs:
        raise ValueError("train_z_seq_list cannot be empty")

    gate = str(cfg.informative_gate).strip().lower()
    if gate not in {"safety_only", "radial", "margin", "local_knn_margin"}:
        raise ValueError("informative_gate must be one of: safety_only, radial, margin, local_knn_margin")

    ref_embeddings = np.asarray(reference_stats.ref_embeddings, dtype=np.float64)
    ref_labels = np.asarray(reference_stats.ref_labels, dtype=np.int64)
    center_old = np.asarray(operator.get_artifacts().mu, dtype=np.float64)

    candidate_rows: List[Dict[str, object]] = []
    continuity_by_class: Dict[int, List[float]] = {}
    k = int(max(1, cfg.knn_k))

    for trial_idx, (tid, cls, seq) in enumerate(zip(tids, labels, seqs)):
        z_aug, _delta, _op_meta = operator.transform(
            seq,
            gamma_main=float(cfg.gamma_main),
            smooth_lambda=float(cfg.smooth_lambda),
        )
        start = int(reference_stats.trial_offsets[int(trial_idx)])
        for window_index in range(int(seq.shape[0])):
            ref_index = int(start + int(window_index))
            z_orig = np.asarray(seq[int(window_index)], dtype=np.float64)
            z_aug_t = np.asarray(z_aug[int(window_index)], dtype=np.float64)

            maj_orig, purity_orig = _knn_majority_and_purity(
                z_orig,
                ref_embeddings=ref_embeddings,
                ref_labels=ref_labels,
                k=int(k),
                exclude_index=int(ref_index),
                fallback_label=int(cls),
            )
            maj_aug, purity_aug = _knn_majority_and_purity(
                z_aug_t,
                ref_embeddings=ref_embeddings,
                ref_labels=ref_labels,
                k=int(k),
                exclude_index=None,
                fallback_label=int(cls),
            )

            local_step_orig = _window_local_step_mean(seq, int(window_index))
            local_step_aug = _window_local_step_mean(z_aug, int(window_index))
            local_cont_ratio = float(local_step_aug / (local_step_orig + 1e-12)) if seq.shape[0] > 1 else 1.0
            radial_gain = float(np.linalg.norm(z_aug_t - center_old) - np.linalg.norm(z_orig - center_old))
            margin_orig = float(_compute_margin(z_orig, true_class=int(cls), class_centers=reference_stats.class_centers))
            margin_aug = float(_compute_margin(z_aug_t, true_class=int(cls), class_centers=reference_stats.class_centers))
            margin_gain = float(margin_aug - margin_orig)
            local_knn_margin_orig = float(
                _compute_local_knn_margin(
                    z_orig,
                    true_class=int(cls),
                    ref_embeddings=ref_embeddings,
                    ref_labels=ref_labels,
                    k=int(k),
                    exclude_index=int(ref_index),
                )
            )
            local_knn_margin_aug = float(
                _compute_local_knn_margin(
                    z_aug_t,
                    true_class=int(cls),
                    ref_embeddings=ref_embeddings,
                    ref_labels=ref_labels,
                    k=int(k),
                    exclude_index=int(ref_index),
                )
            )
            local_knn_margin_gain = float(local_knn_margin_aug - local_knn_margin_orig)

            continuity_by_class.setdefault(int(cls), []).append(float(local_cont_ratio))
            candidate_rows.append(
                {
                    "trial_id": str(tid),
                    "label": int(cls),
                    "window_index": int(window_index),
                    "ref_index": int(ref_index),
                    "z_window_aug": np.asarray(z_aug_t, dtype=np.float32),
                    "majority_orig_window": int(maj_orig),
                    "majority_aug_window": int(maj_aug),
                    "purity_orig_window": float(purity_orig),
                    "purity_aug_window": float(purity_aug),
                    "purity_drop_window": float(purity_orig - purity_aug),
                    "local_step_orig": float(local_step_orig),
                    "local_step_aug": float(local_step_aug),
                    "local_continuity_ratio": float(local_cont_ratio),
                    "radial_gain_window": float(radial_gain),
                    "margin_gain_window": float(margin_gain),
                    "local_knn_margin_orig_window": float(local_knn_margin_orig),
                    "local_knn_margin_aug_window": float(local_knn_margin_aug),
                    "local_knn_margin_gain_window": float(local_knn_margin_gain),
                    "safe": False,
                    "accepted": False,
                    "reject_reason": "",
                }
            )

    continuity_q_by_class: Dict[int, float] = {}
    for cls, vals in continuity_by_class.items():
        continuity_q_by_class[int(cls)] = float(
            np.percentile(np.asarray(vals, dtype=np.float64), float(cfg.continuity_quantile))
        )

    safe_rows_by_class: Dict[int, List[Dict[str, object]]] = {}
    for row in candidate_rows:
        cls = int(row["label"])
        reasons: List[str] = []
        if int(row["majority_aug_window"]) != int(cls):
            reasons.append("majority_changed")
        if float(row["purity_drop_window"]) > float(cfg.max_purity_drop):
            reasons.append("purity_drop")
        if float(row["local_continuity_ratio"]) > float(continuity_q_by_class.get(int(cls), np.inf)):
            reasons.append("continuity_q75")
        if not reasons:
            row["safe"] = True
            safe_rows_by_class.setdefault(int(cls), []).append(row)
        else:
            row["safe"] = False
            row["reject_reason"] = "|".join(reasons)

    gate_thresholds: Dict[int, float] = {}
    if gate == "radial":
        for cls, rows in safe_rows_by_class.items():
            gate_thresholds[int(cls)] = float(np.median([float(r["radial_gain_window"]) for r in rows])) if rows else np.inf
    elif gate == "margin":
        for cls, rows in safe_rows_by_class.items():
            gate_thresholds[int(cls)] = float(np.median([float(r["margin_gain_window"]) for r in rows])) if rows else np.inf
    elif gate == "local_knn_margin":
        for cls, rows in safe_rows_by_class.items():
            gate_thresholds[int(cls)] = (
                float(np.median([float(r["local_knn_margin_gain_window"]) for r in rows])) if rows else np.inf
            )

    accepted_labels: List[int] = []
    accepted_window_seq_list: List[np.ndarray] = []
    accepted_window_rows: List[Dict[str, object]] = []
    for row in candidate_rows:
        if not bool(row["safe"]):
            row["accepted"] = False
            continue
        cls = int(row["label"])
        if gate == "safety_only":
            keep = True
        elif gate == "radial":
            keep = float(row["radial_gain_window"]) >= float(gate_thresholds.get(int(cls), np.inf))
        elif gate == "local_knn_margin":
            keep = float(row["local_knn_margin_gain_window"]) >= float(gate_thresholds.get(int(cls), np.inf))
        else:
            keep = float(row["margin_gain_window"]) >= float(gate_thresholds.get(int(cls), np.inf))
        if keep:
            row["accepted"] = True
            accepted_labels.append(int(cls))
            accepted_window_seq_list.append(np.asarray(row["z_window_aug"], dtype=np.float32)[None, :])
            accepted_window_rows.append(dict(row))
        else:
            row["accepted"] = False
            row["reject_reason"] = "informative_gate"
        row["z_window_aug"] = np.asarray(row["z_window_aug"], dtype=np.float32)

    class_coverage_rows: List[Dict[str, object]] = []
    accepted_by_class: Dict[int, int] = {}
    for row in accepted_window_rows:
        cls = int(row["label"])
        accepted_by_class[int(cls)] = int(accepted_by_class.get(int(cls), 0) + 1)
    for cls in sorted(set(int(v) for v in labels)):
        safe_count = int(len(safe_rows_by_class.get(int(cls), [])))
        admitted_count = int(accepted_by_class.get(int(cls), 0))
        coverage_threshold = int(max(8, int(np.ceil(0.05 * max(0, safe_count)))))
        low_coverage = int(admitted_count < coverage_threshold)
        class_coverage_rows.append(
            {
                "class_id": int(cls),
                "safe_window_count_class": int(safe_count),
                "admitted_window_count_class": int(admitted_count),
                "coverage_ratio_class": float(admitted_count / max(1, safe_count)),
                "coverage_threshold_class": int(coverage_threshold),
                "low_coverage_flag": int(low_coverage),
                "effective_trigger": int(0 if low_coverage else 1),
            }
        )

    safe_mask = np.asarray([bool(r["safe"]) for r in candidate_rows], dtype=bool)
    accepted_mask = np.asarray([bool(r["accepted"]) for r in candidate_rows], dtype=bool)
    accepted_trial_ids = sorted(set(str(r["trial_id"]) for r in accepted_window_rows))
    summary = {
        "candidate_window_count": int(len(candidate_rows)),
        "safe_window_count": int(np.sum(safe_mask)),
        "accepted_window_count": int(np.sum(accepted_mask)),
        "accept_rate": float(np.mean(accepted_mask.astype(np.float64))) if accepted_mask.size else 0.0,
        "source_trial_coverage": float(len(accepted_trial_ids) / max(1, len(tids))),
        "class_balance_proxy": float(_class_balance_proxy(accepted_labels)),
        "mean_purity_drop_accepted": float(
            np.mean([float(r["purity_drop_window"]) for r in accepted_window_rows])
        )
        if accepted_window_rows
        else 0.0,
        "mean_local_continuity_ratio_accepted": float(
            np.mean([float(r["local_continuity_ratio"]) for r in accepted_window_rows])
        )
        if accepted_window_rows
        else 0.0,
        "mean_radial_gain_accepted": float(np.mean([float(r["radial_gain_window"]) for r in accepted_window_rows]))
        if accepted_window_rows
        else 0.0,
        "mean_margin_gain_accepted": float(np.mean([float(r["margin_gain_window"]) for r in accepted_window_rows]))
        if accepted_window_rows
        else 0.0,
        "mean_local_knn_margin_gain_accepted": float(
            np.mean([float(r["local_knn_margin_gain_window"]) for r in accepted_window_rows])
        )
        if accepted_window_rows
        else 0.0,
        "low_coverage_class_count": int(sum(int(r["low_coverage_flag"]) for r in class_coverage_rows)),
        "low_coverage_class_rate": float(
            np.mean([float(r["low_coverage_flag"]) for r in class_coverage_rows])
        )
        if class_coverage_rows
        else 0.0,
        "pool_type": "window_conditioned_feedback_pool",
        "object_mode": "window_level",
        "informative_gate": str(gate),
        "knn_reference_set": "orig_train_only_windows",
        "margin_center_reference": "orig_train_only_window_class_centers",
        "local_knn_margin_reference": "orig_train_only_windows_excluding_current_window_for_same_class",
        "coverage_guard_mode": "max(8, ceil(0.05 * safe_window_count_class))",
        "window_to_rebasis_mode": "length1_pseudo_sequence",
        "scope_note": (
            "This window-conditioned feedback-pool evaluation measures rebasis signal via length-1 pseudo sequences; "
            "it does not yet claim full segment-aware or trajectory-aware rebasis geometry."
        ),
    }
    return TrajectoryWindowFeedbackPoolResult(
        accepted_window_seq_list=accepted_window_seq_list,
        accepted_labels=accepted_labels,
        accepted_window_rows=accepted_window_rows,
        candidate_rows=candidate_rows,
        class_coverage_rows=class_coverage_rows,
        summary=summary,
    )
