from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


def _seq_step_change_mean(z_seq: np.ndarray) -> float:
    arr = np.asarray(z_seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    delta = np.diff(arr, axis=0)
    return float(np.mean(np.linalg.norm(delta, axis=1)))


@dataclass
class TrajectoryDualRolePolicyResult:
    aug_tids: List[str]
    aug_labels: List[int]
    aug_z_seq_list: List[np.ndarray]
    constructive_summary: Dict[str, object] = field(default_factory=dict)
    discriminative_summary: Dict[str, object] = field(default_factory=dict)
    role_overlap_summary: Dict[str, object] = field(default_factory=dict)
    stitching_summary: Dict[str, object] = field(default_factory=dict)


def _group_rows_by_trial(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[int, Dict[str, object]]]:
    out: Dict[str, Dict[int, Dict[str, object]]] = {}
    for row in rows:
        tid = str(row["trial_id"])
        win = int(row["window_index"])
        out.setdefault(tid, {})[int(win)] = dict(row)
    return out


def compute_role_overlap_summary(
    *,
    constructive_rows: Sequence[Dict[str, object]],
    discriminative_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    constructive_set = {(str(r["trial_id"]), int(r["window_index"])) for r in constructive_rows}
    discriminative_set = {(str(r["trial_id"]), int(r["window_index"])) for r in discriminative_rows}
    overlap = constructive_set & discriminative_set
    union = constructive_set | discriminative_set
    constructive_count = int(len(constructive_set))
    discriminative_count = int(len(discriminative_set))
    overlap_count = int(len(overlap))
    return {
        "constructive_count": int(constructive_count),
        "discriminative_count": int(discriminative_count),
        "overlap_count": int(overlap_count),
        "role_overlap_jaccard": float(overlap_count / max(1, len(union))),
        "constructive_only_ratio": float((constructive_count - overlap_count) / max(1, constructive_count)),
        "discriminative_only_ratio": float((discriminative_count - overlap_count) / max(1, discriminative_count)),
    }


def build_dual_role_augmented_trajectories(
    *,
    train_tids: Sequence[str],
    train_labels: Sequence[int],
    train_z_seq_list: Sequence[np.ndarray],
    constructive_pool_summary: Dict[str, object],
    discriminative_pool_summary: Dict[str, object],
    constructive_rows: Sequence[Dict[str, object]],
    discriminative_rows: Sequence[Dict[str, object]],
) -> TrajectoryDualRolePolicyResult:
    tids = [str(v) for v in train_tids]
    labels = [int(v) for v in train_labels]
    seqs = [np.asarray(v, dtype=np.float32) for v in train_z_seq_list]
    if len(tids) != len(labels) or len(tids) != len(seqs):
        raise ValueError("train inputs must align")

    # T5 first version only rewrites z_seq-level windows. No raw-level stitching is allowed.
    disc_by_trial = _group_rows_by_trial(discriminative_rows)
    aug_tids: List[str] = []
    aug_labels: List[int] = []
    aug_z_seq_list: List[np.ndarray] = []

    masked_window_ratios: List[float] = []
    stitch_boundary_counts: List[int] = []
    stitch_jump_ratios: List[float] = []
    stitched_cont_ratios: List[float] = []

    for tid, cls, seq in zip(tids, labels, seqs):
        accepted = disc_by_trial.get(str(tid), {})
        if not accepted:
            continue
        mask = np.zeros((int(seq.shape[0]),), dtype=bool)
        mixed = np.asarray(seq, dtype=np.float32).copy()
        for win, row in accepted.items():
            if int(win) < 0 or int(win) >= int(seq.shape[0]):
                continue
            mask[int(win)] = True
            mixed[int(win)] = np.asarray(row["z_window_aug"], dtype=np.float32)
        if not np.any(mask):
            continue

        aug_tids.append(f"{tid}__t5_dual_role_aug")
        aug_labels.append(int(cls))
        aug_z_seq_list.append(np.asarray(mixed, dtype=np.float32))

        masked_window_ratios.append(float(np.mean(mask.astype(np.float64))))
        if int(seq.shape[0]) >= 2:
            orig_step = np.linalg.norm(np.diff(np.asarray(seq, dtype=np.float64), axis=0), axis=1)
            mixed_step = np.linalg.norm(np.diff(np.asarray(mixed, dtype=np.float64), axis=0), axis=1)
            boundaries = np.flatnonzero(mask[1:] != mask[:-1])
            stitch_boundary_counts.append(int(boundaries.size))
            for b in boundaries.tolist():
                stitch_jump_ratios.append(float(mixed_step[int(b)] / (orig_step[int(b)] + 1e-12)))
            stitched_cont_ratios.append(float(_seq_step_change_mean(mixed) / (_seq_step_change_mean(seq) + 1e-12)))
        else:
            stitch_boundary_counts.append(0)
            stitched_cont_ratios.append(1.0)

    stitching_summary = {
        "augmented_trial_count": int(len(aug_tids)),
        "masked_window_ratio": float(np.mean(masked_window_ratios)) if masked_window_ratios else 0.0,
        "stitch_boundary_count": float(np.mean(stitch_boundary_counts)) if stitch_boundary_counts else 0.0,
        "stitch_boundary_jump_ratio_mean": float(np.mean(stitch_jump_ratios)) if stitch_jump_ratios else 1.0,
        "stitched_continuity_distortion_ratio": float(np.mean(stitched_cont_ratios)) if stitched_cont_ratios else 1.0,
        "stitch_scope": "z_seq_only",
        "raw_level_stitching": False,
    }
    return TrajectoryDualRolePolicyResult(
        aug_tids=aug_tids,
        aug_labels=aug_labels,
        aug_z_seq_list=aug_z_seq_list,
        constructive_summary=dict(constructive_pool_summary),
        discriminative_summary=dict(discriminative_pool_summary),
        role_overlap_summary=compute_role_overlap_summary(
            constructive_rows=constructive_rows,
            discriminative_rows=discriminative_rows,
        ),
        stitching_summary=stitching_summary,
    )
