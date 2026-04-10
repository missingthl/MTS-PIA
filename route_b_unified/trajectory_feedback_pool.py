from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator


def _trajectory_embedding(z_seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(z_seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] <= 0:
        raise ValueError("trajectory sequence must be [K, D] with K>0")
    return np.mean(seq, axis=0).astype(np.float64)


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
    best_idx = int(np.argmax(counts))
    majority = int(values[best_idx]) if values.size else int(fallback_label)
    purity = float(counts[best_idx] / max(1, int(nn_labels.size))) if counts.size else 1.0
    return int(majority), float(purity)


@dataclass(frozen=True)
class TrajectoryFeedbackPoolConfig:
    gamma_main: float = 0.05
    smooth_lambda: float = 0.50
    knn_k: int = 5
    max_purity_drop: float = 0.10
    continuity_quantile: float = 75.0


@dataclass
class TrajectoryFeedbackPoolResult:
    accepted_tids: List[str]
    accepted_labels: List[int]
    accepted_z_seq_list: List[np.ndarray]
    candidate_rows: List[Dict[str, object]]
    summary: Dict[str, object] = field(default_factory=dict)


def build_trajectory_feedback_pool(
    *,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    operator: TrajectoryPIAOperator,
    cfg: TrajectoryFeedbackPoolConfig,
) -> TrajectoryFeedbackPoolResult:
    tids = [str(v) for v in train_tids]
    labels = [int(v) for v in train_labels]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    if len(tids) != len(labels) or len(tids) != len(seqs):
        raise ValueError("train_tids/train_labels/train_z_seq_list must align")
    if not seqs:
        raise ValueError("train_z_seq_list cannot be empty")

    # Hard constraint: all purity stats are computed against orig-train-only reference set.
    ref_embeddings = np.stack([_trajectory_embedding(v) for v in seqs], axis=0).astype(np.float64)
    ref_labels = np.asarray(labels, dtype=np.int64)

    candidate_rows: List[Dict[str, object]] = []
    continuity_by_class: Dict[int, List[float]] = {}
    accepted_tids: List[str] = []
    accepted_labels: List[int] = []
    accepted_z_seq_list: List[np.ndarray] = []

    k = int(max(1, cfg.knn_k))
    gamma_main = float(cfg.gamma_main)
    smooth_lambda = float(cfg.smooth_lambda)

    for idx, (tid, cls, seq) in enumerate(zip(tids, labels, seqs)):
        z_aug, _delta, op_meta = operator.transform(
            seq,
            gamma_main=float(gamma_main),
            smooth_lambda=float(smooth_lambda),
        )
        orig_emb = ref_embeddings[int(idx)]
        aug_emb = _trajectory_embedding(z_aug)

        maj_orig, purity_orig = _knn_majority_and_purity(
            orig_emb,
            ref_embeddings=ref_embeddings,
            ref_labels=ref_labels,
            k=int(k),
            exclude_index=int(idx),
            fallback_label=int(cls),
        )
        maj_aug, purity_aug = _knn_majority_and_purity(
            aug_emb,
            ref_embeddings=ref_embeddings,
            ref_labels=ref_labels,
            k=int(k),
            exclude_index=None,
            fallback_label=int(cls),
        )
        continuity = float(op_meta["continuity_distortion_ratio"])
        continuity_by_class.setdefault(int(cls), []).append(float(continuity))

        candidate_rows.append(
            {
                "trial_id": str(tid),
                "label": int(cls),
                "z_seq_aug": np.asarray(z_aug, dtype=np.float32),
                "continuity_distortion_ratio": float(continuity),
                "purity_orig": float(purity_orig),
                "purity_aug": float(purity_aug),
                "purity_drop": float(purity_orig - purity_aug),
                "majority_orig": int(maj_orig),
                "majority_aug": int(maj_aug),
                "accepted": False,
                "reject_reason": "",
            }
        )

    q_by_class: Dict[int, float] = {}
    for cls, rows in continuity_by_class.items():
        q_by_class[int(cls)] = float(np.percentile(np.asarray(rows, dtype=np.float64), float(cfg.continuity_quantile)))

    for row in candidate_rows:
        cls = int(row["label"])
        reasons: List[str] = []
        if int(row["majority_aug"]) != int(cls):
            reasons.append("majority_changed")
        if float(row["purity_drop"]) > float(cfg.max_purity_drop):
            reasons.append("purity_drop")
        if float(row["continuity_distortion_ratio"]) > float(q_by_class.get(int(cls), np.inf)):
            reasons.append("continuity_q75")

        if not reasons:
            row["accepted"] = True
            row["reject_reason"] = ""
            accepted_tids.append(str(row["trial_id"]))
            accepted_labels.append(int(cls))
            accepted_z_seq_list.append(np.asarray(row["z_seq_aug"], dtype=np.float32))
        else:
            row["accepted"] = False
            row["reject_reason"] = "|".join(reasons)

        # keep tabular rows serializable
        row["z_seq_aug"] = np.asarray(row["z_seq_aug"], dtype=np.float32)

    accepted_mask = np.asarray([bool(r["accepted"]) for r in candidate_rows], dtype=bool)
    accepted_purity_drop = [float(r["purity_drop"]) for r in candidate_rows if bool(r["accepted"])]
    accepted_cont = [float(r["continuity_distortion_ratio"]) for r in candidate_rows if bool(r["accepted"])]
    summary = {
        "candidate_count": int(len(candidate_rows)),
        "accepted_count": int(np.sum(accepted_mask)),
        "accept_rate": float(np.mean(accepted_mask.astype(np.float64))) if accepted_mask.size else 0.0,
        "class_balance_proxy": float(_class_balance_proxy(accepted_labels)),
        "mean_purity_drop_accepted": float(np.mean(accepted_purity_drop)) if accepted_purity_drop else 0.0,
        "mean_continuity_ratio_accepted": float(np.mean(accepted_cont)) if accepted_cont else 0.0,
        "pool_type": "safety_filtered_pool",
        "knn_reference_set": "orig_train_only",
        "continuity_quantile": float(cfg.continuity_quantile),
        "max_purity_drop": float(cfg.max_purity_drop),
    }
    return TrajectoryFeedbackPoolResult(
        accepted_tids=accepted_tids,
        accepted_labels=accepted_labels,
        accepted_z_seq_list=accepted_z_seq_list,
        candidate_rows=candidate_rows,
        summary=summary,
    )
