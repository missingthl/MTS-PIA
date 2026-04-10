from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

from route_b_unified.trajectory_pia_operator import TrajectoryPIAOperator, TrajectoryPIAOperatorConfig


def _basis_cosine(old_dir: np.ndarray, new_dir: np.ndarray) -> float:
    a = np.asarray(old_dir, dtype=np.float64).ravel()
    b = np.asarray(new_dir, dtype=np.float64).ravel()
    if a.size != b.size or a.size <= 0:
        raise ValueError("direction vectors must have same non-zero size")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0
    return float(np.clip(abs(float(np.dot(a, b))) / denom, 0.0, 1.0))


@dataclass
class TrajectoryFeedbackRebasisResult:
    operator_new: TrajectoryPIAOperator
    center_old: np.ndarray
    center_new: np.ndarray
    direction_old: np.ndarray
    direction_new: np.ndarray
    summary: Dict[str, object] = field(default_factory=dict)


def fit_trajectory_feedback_rebasis(
    *,
    orig_train_z_seq_list: Sequence[np.ndarray],
    feedback_z_seq_list: Sequence[np.ndarray],
    old_operator: TrajectoryPIAOperator,
    operator_cfg: TrajectoryPIAOperatorConfig,
) -> TrajectoryFeedbackRebasisResult:
    old_arts = old_operator.get_artifacts()
    densified = [np.asarray(v, dtype=np.float32) for v in orig_train_z_seq_list] + [
        np.asarray(v, dtype=np.float32) for v in feedback_z_seq_list
    ]
    operator_new = TrajectoryPIAOperator(operator_cfg).fit(densified)
    new_arts = operator_new.get_artifacts()

    cosine = float(_basis_cosine(old_arts.direction, new_arts.direction))
    angle = float(np.arccos(np.clip(cosine, 0.0, 1.0)))
    center_shift_norm = float(np.linalg.norm(np.asarray(new_arts.mu, dtype=np.float64) - np.asarray(old_arts.mu, dtype=np.float64)))
    pooled_feedback_windows = int(sum(int(np.asarray(v).shape[0]) for v in feedback_z_seq_list))

    summary = {
        "center_shift_norm": float(center_shift_norm),
        "basis_cosine_to_old": float(cosine),
        "basis_angle_proxy": float(angle),
        "pooled_window_count_orig": int(old_arts.pooled_window_count),
        "pooled_window_count_feedback": int(pooled_feedback_windows),
        "pooled_window_count_total": int(new_arts.pooled_window_count),
        "rebasis_mode": "constrained_shared_basis_refit",
        "shared_basis_mode": "global_single_axis",
    }
    return TrajectoryFeedbackRebasisResult(
        operator_new=operator_new,
        center_old=np.asarray(old_arts.mu, dtype=np.float32),
        center_new=np.asarray(new_arts.mu, dtype=np.float32),
        direction_old=np.asarray(old_arts.direction, dtype=np.float32),
        direction_new=np.asarray(new_arts.direction, dtype=np.float32),
        summary=summary,
    )
