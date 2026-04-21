from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from PIA.telm2 import TELM2Config, TELM2Transformer
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


def _make_telm_cfg(cfg: TrajectoryPIAOperatorConfig) -> TELM2Config:
    return TELM2Config(
        r_dimension=1,
        n_iters=int(cfg.n_iters),
        C_repr=float(cfg.C_repr),
        activation=str(cfg.activation),
        bias_lr=float(cfg.bias_lr),
        orthogonalize=bool(cfg.orthogonalize),
        enable_repr_learning=bool(cfg.enable_repr_learning),
        bias_update_mode=str(cfg.bias_update_mode),
        seed=None if cfg.seed is None else int(cfg.seed),
    )


def _flatten_windows(z_seq_list: Sequence[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(seq, dtype=np.float64) for seq in z_seq_list if np.asarray(seq).size > 0]
    if not rows:
        raise ValueError("z_seq_list cannot be empty")
    x = np.concatenate(rows, axis=0).astype(np.float64)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError("pooled windows must be a non-empty 2D array")
    return x


def _fit_direction_from_centered_windows(
    centered_windows: np.ndarray,
    *,
    operator_cfg: TrajectoryPIAOperatorConfig,
) -> np.ndarray:
    telm = TELM2Transformer(_make_telm_cfg(operator_cfg)).fit(np.asarray(centered_windows, dtype=np.float64))
    raw = telm.get_artifacts()
    w = np.asarray(raw.W, dtype=np.float64)
    if w.ndim != 2:
        w = np.asarray(w, dtype=np.float64).reshape(1, -1)
    return np.asarray(_normalize_direction(w[0]), dtype=np.float64)


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


@dataclass
class TrajectoryClassConditionedBasisFamily:
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
            "generator_mode": "class_conditioned_basis_family",
            "gamma_main": float(gamma_main),
            "smooth_lambda": float(smooth_lambda),
            "mean_continuity_distortion_ratio": float(np.mean(continuity_rows)) if continuity_rows else 1.0,
            "n_aug_sequences": int(len(z_aug_list)),
            "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
            "causal_scope_note": (
                "T4a measures the overall effect of a class-conditioned basis-family generator "
                "and does not isolate basis-organization contribution as the only causal source."
            ),
        }


@dataclass
class TrajectoryClassConditionedRebasisResult:
    family: TrajectoryClassConditionedBasisFamily
    class_rows: List[Dict[str, object]]
    summary: Dict[str, object] = field(default_factory=dict)


def fit_trajectory_class_conditioned_rebasis(
    *,
    orig_train_labels: Sequence[int],
    orig_train_z_seq_list: Sequence[np.ndarray],
    feedback_labels: Sequence[int],
    feedback_z_seq_list: Sequence[np.ndarray],
    old_operator: TrajectoryPIAOperator,
    operator_cfg: TrajectoryPIAOperatorConfig,
) -> TrajectoryClassConditionedRebasisResult:
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
        orig_cls = list(orig_by_class.get(int(cls), []))
        fb_cls = list(feedback_by_class.get(int(cls), []))
        densified_cls = orig_cls + fb_cls
        if not densified_cls:
            raise ValueError(f"class {cls} has no densified trajectories")
        x_cls = _flatten_windows(densified_cls) - center_new[None, :]
        direction_cls = _fit_direction_from_centered_windows(x_cls, operator_cfg=operator_cfg)
        directions_by_class[int(cls)] = np.asarray(direction_cls, dtype=np.float64)

    inter_mean, inter_min, inter_max = _pairwise_cos_stats(directions_by_class)
    center_shift_norm = float(np.linalg.norm(center_new - center_old))

    for cls in class_ids:
        orig_cls = list(orig_by_class.get(int(cls), []))
        fb_cls = list(feedback_by_class.get(int(cls), []))
        direction_cls = np.asarray(directions_by_class[int(cls)], dtype=np.float64)
        cos_old = float(_basis_cosine(direction_old, direction_cls))
        angle_old = float(np.arccos(np.clip(cos_old, 0.0, 1.0)))
        class_rows.append(
            {
                "class_id": int(cls),
                "pooled_window_count_orig": int(sum(int(np.asarray(v).shape[0]) for v in orig_cls)),
                "pooled_window_count_feedback": int(sum(int(np.asarray(v).shape[0]) for v in fb_cls)),
                "pooled_window_count_total": int(sum(int(np.asarray(v).shape[0]) for v in orig_cls + fb_cls)),
                "basis_cosine_to_old_shared": float(cos_old),
                "basis_angle_to_old_shared": float(angle_old),
                "inter_basis_cosine_mean": float(inter_mean),
                "inter_basis_cosine_min": float(inter_min),
                "inter_basis_cosine_max": float(inter_max),
                "basis_mode": "class_conditioned_single_axis",
            }
        )

    summary = {
        "center_shift_norm": float(center_shift_norm),
        "n_classes": int(len(class_ids)),
        "rebasis_mode": "class_conditioned_single_axis_family",
        "shared_basis_reference_mode": "global_single_axis_old_basis",
        "inter_basis_cosine_mean": float(inter_mean),
        "inter_basis_cosine_min": float(inter_min),
        "inter_basis_cosine_max": float(inter_max),
        "causal_scope_note": (
            "T4a measures the overall effect of a class-conditioned basis-family generator; "
            "it does not isolate basis-organization contribution as the only causal source."
        ),
    }

    family = TrajectoryClassConditionedBasisFamily(
        center_old=np.asarray(center_old, dtype=np.float32),
        center_new=np.asarray(center_new, dtype=np.float32),
        direction_old_shared=np.asarray(direction_old, dtype=np.float32),
        directions_by_class={int(k): np.asarray(v, dtype=np.float32) for k, v in directions_by_class.items()},
        meta=dict(summary),
    )
    return TrajectoryClassConditionedRebasisResult(
        family=family,
        class_rows=class_rows,
        summary=summary,
    )
