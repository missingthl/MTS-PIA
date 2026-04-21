from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from core.trajectory_feedback_rebasis import _basis_cosine
from core.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig


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
    return np.asarray((1.0 - lam) * cur_delta + 0.5 * lam * (prev_delta + next_delta), dtype=np.float64)


def _seq_step_change_mean(z_seq: np.ndarray) -> float:
    arr = np.asarray(z_seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    delta = np.diff(arr, axis=0)
    return float(np.mean(np.linalg.norm(delta, axis=1)))


def _flatten_windows(z_seq_list: Sequence[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(seq, dtype=np.float64) for seq in z_seq_list if np.asarray(seq).size > 0]
    if not rows:
        raise ValueError("z_seq_list cannot be empty")
    x = np.concatenate(rows, axis=0).astype(np.float64)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError("pooled windows must be a non-empty 2D array")
    return x


def _pairwise_cos_stats(directions_by_class: Dict[int, np.ndarray]) -> tuple[float, float, float]:
    classes = sorted(int(k) for k in directions_by_class)
    vals: List[float] = []
    for i, cls_i in enumerate(classes):
        for cls_j in classes[i + 1 :]:
            vals.append(float(_basis_cosine(directions_by_class[int(cls_i)], directions_by_class[int(cls_j)])))
    if not vals:
        return 1.0, 1.0, 1.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))


def _fit_ovr_direction_from_centered_windows(
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    *,
    operator_cfg: TrajectoryPIAOperatorConfig,
) -> np.ndarray:
    pos = np.asarray(x_pos, dtype=np.float64)
    neg = np.asarray(x_neg, dtype=np.float64)
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        raise ValueError("x_pos and x_neg must be 2D arrays with matching feature dimension")
    if pos.shape[0] <= 0 or neg.shape[0] <= 0:
        raise ValueError("OvR fitting requires both positive and negative windows")

    mu_pos = np.mean(pos, axis=0)
    mu_neg = np.mean(neg, axis=0)
    diff = np.asarray(mu_pos - mu_neg, dtype=np.float64)

    pos_res = pos - mu_pos[None, :]
    neg_res = neg - mu_neg[None, :]
    sw = pos_res.T @ pos_res + neg_res.T @ neg_res
    dim = int(sw.shape[0])
    scale = float(np.trace(sw) / max(1, dim))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    ridge = float(max(1e-6, 1e-3 * scale / max(1e-6, float(operator_cfg.C_repr))))
    sw_reg = sw + ridge * np.eye(dim, dtype=np.float64)
    direction = np.linalg.pinv(sw_reg) @ diff
    return np.asarray(_normalize_direction(direction), dtype=np.float64)


@dataclass
class TrajectoryClassConditionedOVRBasisFamily:
    center_old: np.ndarray
    center_new: np.ndarray
    direction_old_shared: np.ndarray
    directions_by_class: Dict[int, np.ndarray]
    meta: Dict[str, object] = field(default_factory=dict)

    def transform(
        self,
        z_seq: np.ndarray,
        *,
        label: int,
        gamma_main: float,
        smooth_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        cls = int(label)
        if int(cls) not in self.directions_by_class:
            raise KeyError(f"class-conditioned basis missing class {cls}")
        direction = np.asarray(self.directions_by_class[int(cls)], dtype=np.float64)
        seq = np.asarray(z_seq, dtype=np.float64)
        centered = seq - np.asarray(self.center_new, dtype=np.float64)[None, :]
        coeff = centered @ direction[:, None]
        comp = coeff * direction[None, :]
        delta_raw = float(gamma_main) * comp
        delta_smooth = _smooth_delta(delta_raw, float(smooth_lambda))
        z_aug = seq + delta_smooth
        base_step = _seq_step_change_mean(seq)
        aug_step = _seq_step_change_mean(z_aug)
        continuity_ratio = float(aug_step / (base_step + 1e-12)) if seq.shape[0] > 1 else 1.0
        meta = {
            "label": int(cls),
            "gamma_main": float(gamma_main),
            "smooth_lambda": float(smooth_lambda),
            "base_step_change_mean": float(base_step),
            "aug_step_change_mean": float(aug_step),
            "continuity_distortion_ratio": float(continuity_ratio),
        }
        return np.asarray(z_aug, dtype=np.float32), np.asarray(delta_smooth, dtype=np.float32), meta

    def transform_many(
        self,
        z_seq_list: Sequence[np.ndarray],
        *,
        labels: Sequence[int],
        gamma_main: float,
        smooth_lambda: float,
    ) -> tuple[List[np.ndarray], List[np.ndarray], Dict[str, object]]:
        if len(z_seq_list) != len(labels):
            raise ValueError("z_seq_list and labels must align")
        z_aug_list: List[np.ndarray] = []
        delta_list: List[np.ndarray] = []
        continuity_rows: List[float] = []
        class_counts: Dict[int, int] = {}
        for seq, label in zip(z_seq_list, labels):
            z_aug, delta_seq, meta = self.transform(
                seq,
                label=int(label),
                gamma_main=float(gamma_main),
                smooth_lambda=float(smooth_lambda),
            )
            z_aug_list.append(z_aug)
            delta_list.append(delta_seq)
            continuity_rows.append(float(meta["continuity_distortion_ratio"]))
            class_counts[int(label)] = int(class_counts.get(int(label), 0) + 1)
        return z_aug_list, delta_list, {
            "generator_mode": "class_conditioned_ovr_basis_family",
            "gamma_main": float(gamma_main),
            "smooth_lambda": float(smooth_lambda),
            "mean_continuity_distortion_ratio": float(np.mean(continuity_rows)) if continuity_rows else 1.0,
            "n_aug_sequences": int(len(z_aug_list)),
            "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
            "routing_scope": "train_augmentation_only_true_label_container_selection",
            "test_time_routing": False,
            "causal_scope_note": (
                "T7a measures the overall effect of an OvR class-conditioned rebasis container during training-time "
                "augmentation and does not isolate container contribution as the only causal source."
            ),
        }


@dataclass
class TrajectoryClassConditionedOVRRebasisResult:
    family: TrajectoryClassConditionedOVRBasisFamily
    class_rows: List[Dict[str, object]]
    summary: Dict[str, object] = field(default_factory=dict)


def fit_trajectory_class_conditioned_rebasis_ovr(
    *,
    orig_train_labels: Sequence[int],
    orig_train_z_seq_list: Sequence[np.ndarray],
    feedback_labels: Sequence[int],
    feedback_z_seq_list: Sequence[np.ndarray],
    old_operator: TrajectoryPIAOperator,
    operator_cfg: TrajectoryPIAOperatorConfig,
) -> TrajectoryClassConditionedOVRRebasisResult:
    if len(orig_train_labels) != len(orig_train_z_seq_list):
        raise ValueError("orig_train_labels and orig_train_z_seq_list must align")
    if len(feedback_labels) != len(feedback_z_seq_list):
        raise ValueError("feedback_labels and feedback_z_seq_list must align")

    old_arts = old_operator.get_artifacts()
    all_orig = [np.asarray(v, dtype=np.float32) for v in orig_train_z_seq_list]
    all_feedback = [np.asarray(v, dtype=np.float32) for v in feedback_z_seq_list]
    densified = list(all_orig) + list(all_feedback)
    if not densified:
        raise ValueError("densified pool cannot be empty")

    center_new = np.mean(_flatten_windows(densified), axis=0).astype(np.float64)
    center_old = np.asarray(old_arts.mu, dtype=np.float64)
    direction_old = np.asarray(old_arts.direction, dtype=np.float64)

    class_ids = sorted(set(int(v) for v in orig_train_labels))
    directions_by_class: Dict[int, np.ndarray] = {}
    class_rows: List[Dict[str, object]] = []

    orig_by_class: Dict[int, List[np.ndarray]] = {int(cls): [] for cls in class_ids}
    for cls, seq in zip(orig_train_labels, all_orig):
        orig_by_class[int(cls)].append(np.asarray(seq, dtype=np.float32))
    feedback_by_class: Dict[int, List[np.ndarray]] = {int(cls): [] for cls in class_ids}
    for cls, seq in zip(feedback_labels, all_feedback):
        feedback_by_class.setdefault(int(cls), []).append(np.asarray(seq, dtype=np.float32))

    for cls in class_ids:
        pos_seqs = list(orig_by_class.get(int(cls), [])) + list(feedback_by_class.get(int(cls), []))
        neg_seqs: List[np.ndarray] = []
        for other in class_ids:
            if int(other) == int(cls):
                continue
            neg_seqs.extend(list(orig_by_class.get(int(other), [])))
            neg_seqs.extend(list(feedback_by_class.get(int(other), [])))
        if not pos_seqs or not neg_seqs:
            raise ValueError(f"class {cls} cannot form OvR split for rebasis")
        x_pos = _flatten_windows(pos_seqs) - center_new[None, :]
        x_neg = _flatten_windows(neg_seqs) - center_new[None, :]
        direction_cls = _fit_ovr_direction_from_centered_windows(
            x_pos,
            x_neg,
            operator_cfg=operator_cfg,
        )
        directions_by_class[int(cls)] = np.asarray(direction_cls, dtype=np.float64)

    inter_mean, inter_min, inter_max = _pairwise_cos_stats(directions_by_class)
    center_shift_norm = float(np.linalg.norm(center_new - center_old))

    for cls in class_ids:
        pos_orig = list(orig_by_class.get(int(cls), []))
        pos_feedback = list(feedback_by_class.get(int(cls), []))
        neg_total = 0
        for other in class_ids:
            if int(other) == int(cls):
                continue
            neg_total += int(sum(int(np.asarray(v).shape[0]) for v in orig_by_class.get(int(other), [])))
            neg_total += int(sum(int(np.asarray(v).shape[0]) for v in feedback_by_class.get(int(other), [])))
        direction_cls = np.asarray(directions_by_class[int(cls)], dtype=np.float64)
        cos_old = float(_basis_cosine(direction_old, direction_cls))
        angle_old = float(np.arccos(np.clip(cos_old, 0.0, 1.0)))
        class_rows.append(
            {
                "class_id": int(cls),
                "pooled_window_count_orig": int(sum(int(np.asarray(v).shape[0]) for v in pos_orig)),
                "pooled_window_count_feedback": int(sum(int(np.asarray(v).shape[0]) for v in pos_feedback)),
                "pooled_window_count_total": int(sum(int(np.asarray(v).shape[0]) for v in pos_orig + pos_feedback)),
                "ovr_negative_window_count": int(neg_total),
                "basis_cosine_to_old_shared": float(cos_old),
                "basis_angle_to_old_shared": float(angle_old),
                "inter_basis_cosine_mean": float(inter_mean),
                "inter_basis_cosine_min": float(inter_min),
                "inter_basis_cosine_max": float(inter_max),
                "basis_mode": "class_conditioned_single_axis_ovr",
                "basis_solver_mode": "ovr_fisher_single_axis",
                "routing_scope": "train_augmentation_only",
            }
        )

    summary = {
        "center_shift_norm": float(center_shift_norm),
        "n_classes": int(len(class_ids)),
        "rebasis_mode": "class_conditioned_single_axis_family_ovr",
        "shared_basis_reference_mode": "global_single_axis_old_basis",
        "center_mode": "global_pooled_center",
        "inter_basis_cosine_mean": float(inter_mean),
        "inter_basis_cosine_min": float(inter_min),
        "inter_basis_cosine_max": float(inter_max),
        "routing_scope": "train_augmentation_only_true_label_container_selection",
        "test_time_routing": False,
        "solver_scope_note": (
            "Each W_c_new is solved with train-only one-vs-rest logic: class-c orig+admitted windows act as "
            "positives and all other class windows act as negatives."
        ),
    }

    family = TrajectoryClassConditionedOVRBasisFamily(
        center_old=np.asarray(center_old, dtype=np.float32),
        center_new=np.asarray(center_new, dtype=np.float32),
        direction_old_shared=np.asarray(direction_old, dtype=np.float32),
        directions_by_class={int(k): np.asarray(v, dtype=np.float32) for k, v in directions_by_class.items()},
        meta=dict(summary),
    )
    return TrajectoryClassConditionedOVRRebasisResult(
        family=family,
        class_rows=class_rows,
        summary=summary,
    )
