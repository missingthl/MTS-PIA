from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence

import numpy as np

from route_b_unified.trajectory_representation import TrajectoryRepresentationState, TrajectorySplit


@dataclass(frozen=True)
class SCPSingleReplayConfig:
    replay_suffix: str = "__replay1"


@dataclass
class SCPSingleReplayResult:
    replay_state: TrajectoryRepresentationState
    replay_rows: List[Dict[str, object]]
    summary: Dict[str, object] = field(default_factory=dict)


def _changed_mask_from_shaping(
    *,
    tids: Sequence[str],
    z_seq_list: Sequence[np.ndarray],
    shaping_rows: Sequence[Dict[str, object]],
) -> Dict[str, np.ndarray]:
    masks = {
        str(tid): np.zeros(int(np.asarray(seq, dtype=np.float32).shape[0]), dtype=bool)
        for tid, seq in zip(tids, z_seq_list)
    }
    for row in shaping_rows:
        tid = str(row["trial_id"])
        if tid not in masks:
            continue
        idx = int(row["window_index"])
        if 0 <= idx < int(masks[tid].shape[0]):
            masks[tid][idx] = True
    return masks


def _duplicate_trial_dict(trial: Dict[str, object], *, suffix: str) -> Dict[str, object]:
    out = dict(trial)
    if "trial_id_str" in out:
        out["trial_id_str"] = f"{out['trial_id_str']}{suffix}"
    if "trial_id" in out:
        out["trial_id"] = f"{out['trial_id']}{suffix}"
    return out


def build_single_replay_state(
    *,
    state: TrajectoryRepresentationState,
    shaped_train_z_seq_list: Sequence[np.ndarray],
    shaping_rows: Sequence[Dict[str, object]],
    cfg: SCPSingleReplayConfig,
) -> SCPSingleReplayResult:
    orig_seqs = [np.asarray(v, dtype=np.float32) for v in state.train.z_seq_list]
    replay_seqs = [np.asarray(v, dtype=np.float32) for v in shaped_train_z_seq_list]
    if len(orig_seqs) != len(replay_seqs):
        raise ValueError("shaped_train_z_seq_list must align with state.train.z_seq_list")

    masks = _changed_mask_from_shaping(
        tids=state.train.tids.tolist(),
        z_seq_list=state.train.z_seq_list,
        shaping_rows=shaping_rows,
    )

    replay_rows: List[Dict[str, object]] = []
    all_boundary_ratios: List[float] = []
    all_cont_ratios: List[float] = []
    total_changed = 0
    total_windows = 0

    for tid, orig_seq, replay_seq in zip(state.train.tids.tolist(), orig_seqs, replay_seqs):
        mask = np.asarray(masks[str(tid)], dtype=bool)
        total_changed += int(np.sum(mask))
        total_windows += int(mask.size)

        boundary_ratios: List[float] = []
        for idx in range(1, int(mask.size)):
            if bool(mask[idx]) == bool(mask[idx - 1]):
                continue
            orig_step = float(np.linalg.norm(np.asarray(orig_seq[idx], dtype=np.float64) - np.asarray(orig_seq[idx - 1], dtype=np.float64)))
            replay_step = float(
                np.linalg.norm(np.asarray(replay_seq[idx], dtype=np.float64) - np.asarray(replay_seq[idx - 1], dtype=np.float64))
            )
            boundary_ratios.append(float(replay_step / max(1e-6, orig_step)))

        orig_local = np.linalg.norm(np.diff(np.asarray(orig_seq, dtype=np.float64), axis=0), axis=1) if orig_seq.shape[0] > 1 else np.zeros((0,), dtype=np.float64)
        replay_local = (
            np.linalg.norm(np.diff(np.asarray(replay_seq, dtype=np.float64), axis=0), axis=1)
            if replay_seq.shape[0] > 1
            else np.zeros((0,), dtype=np.float64)
        )
        continuity_ratio = float(np.mean(replay_local) / max(1e-6, float(np.mean(orig_local)))) if orig_local.size > 0 else 1.0

        all_boundary_ratios.extend(boundary_ratios)
        all_cont_ratios.append(float(continuity_ratio))
        replay_rows.append(
            {
                "trial_id": str(tid),
                "changed_window_count": int(np.sum(mask)),
                "replay_window_ratio": float(np.mean(mask)) if mask.size > 0 else 0.0,
                "stitch_boundary_count": int(len(boundary_ratios)),
                "stitch_boundary_jump_ratio_mean": float(np.mean(boundary_ratios)) if boundary_ratios else 1.0,
                "replay_continuity_distortion_ratio": float(continuity_ratio),
            }
        )

    suffix = str(cfg.replay_suffix)
    replay_tids = np.asarray([f"{tid}{suffix}" for tid in state.train.tids.tolist()], dtype=object)
    replay_trial_dicts = [_duplicate_trial_dict(trial, suffix=suffix) for trial in state.train.trial_dicts]
    replay_x_static = np.asarray(state.train.X_static, dtype=np.float32).copy()
    replay_logs = [np.asarray(v, dtype=np.float32) for v in state.train.log_matrix_seq_list]
    replay_meta = [list(v) for v in state.train.window_meta_list]

    combined_train = TrajectorySplit(
        split_name=str(state.train.split_name),
        trial_dicts=list(state.train.trial_dicts) + replay_trial_dicts,
        y=np.concatenate([np.asarray(state.train.y, dtype=np.int64), np.asarray(state.train.y, dtype=np.int64)], axis=0),
        tids=np.concatenate([np.asarray(state.train.tids, dtype=object), replay_tids], axis=0),
        X_static=np.concatenate([np.asarray(state.train.X_static, dtype=np.float32), replay_x_static], axis=0),
        z_seq_list=list(orig_seqs) + list(replay_seqs),
        log_matrix_seq_list=list(state.train.log_matrix_seq_list) + replay_logs,
        window_meta_list=list(state.train.window_meta_list) + replay_meta,
        meta={
            "n_trials": int(len(state.train.trial_dicts) * 2),
            "trajectory_len_mean": float(np.mean([int(v.shape[0]) for v in (list(orig_seqs) + list(replay_seqs))])),
            "trajectory_len_min": int(min(int(v.shape[0]) for v in (list(orig_seqs) + list(replay_seqs)))),
            "trajectory_len_max": int(max(int(v.shape[0]) for v in (list(orig_seqs) + list(replay_seqs)))),
            "replay_suffix": str(suffix),
        },
    )
    replay_state = replace(state, train=combined_train)

    summary = {
        "replay_trial_count": int(len(replay_seqs)),
        "replay_window_ratio": float(total_changed / max(1, total_windows)),
        "stitch_boundary_count": int(sum(int(r["stitch_boundary_count"]) for r in replay_rows)),
        "stitch_boundary_jump_ratio_mean": float(np.mean(all_boundary_ratios)) if all_boundary_ratios else 1.0,
        "stitch_boundary_jump_ratio_p95": float(np.percentile(np.asarray(all_boundary_ratios, dtype=np.float64), 95.0))
        if all_boundary_ratios
        else 1.0,
        "replay_continuity_distortion_ratio": float(np.mean(all_cont_ratios)) if all_cont_ratios else 1.0,
    }
    return SCPSingleReplayResult(
        replay_state=replay_state,
        replay_rows=replay_rows,
        summary=summary,
    )
