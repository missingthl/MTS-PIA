from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from route_b_unified.representation import RepresentationConfig, build_representation
from route_b_unified.spd_features import logm_spd, vec_utri
from route_b_unified.trial_records import _covariance_from_trial
from route_b_unified.types import RepresentationState


@dataclass(frozen=True)
class TrajectoryRepresentationConfig:
    dataset: str
    seed: int
    val_fraction: float = 0.25
    spd_eps: float = 1e-4
    processed_root: str = "data/SEED/SEED_EEG/Preprocessed_EEG"
    stim_xlsx: str = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    seediv_root: str = "data/SEED_IV"
    seedv_root: str = "data/SEED_V"
    prop_win_ratio: float = 0.20
    prop_hop_ratio: float = 0.10
    min_window_extra_channels: int = 4
    min_hop_len: int = 4
    force_hop_len: int | None = None


@dataclass
class TrajectorySplit:
    split_name: str
    trial_dicts: List[Dict[str, object]]
    y: np.ndarray
    tids: np.ndarray
    X_static: np.ndarray
    z_seq_list: List[np.ndarray]
    log_matrix_seq_list: List[np.ndarray]
    window_meta_list: List[List[Dict[str, int | str]]]
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrajectoryRepresentationState:
    dataset: str
    seed: int
    split_meta: Dict[str, object]
    static_representation: RepresentationState
    train: TrajectorySplit
    val: TrajectorySplit
    test: TrajectorySplit
    num_classes: int
    channels: int
    z_dim: int
    window_len: int
    hop_len: int
    dynamic_mean_log_train: np.ndarray
    static_feature_mean: np.ndarray
    static_feature_std: np.ndarray
    dynamic_feature_mean: np.ndarray
    dynamic_feature_std: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


def _compute_feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return (
            np.zeros((arr.shape[1] if arr.ndim == 2 else 0,), dtype=np.float32),
            np.ones((arr.shape[1] if arr.ndim == 2 else 0,), dtype=np.float32),
        )
    mean = np.mean(arr, axis=0).astype(np.float32)
    std = np.std(arr, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _window_starts(n_samples: int, win_len: int, hop_len: int) -> np.ndarray:
    if int(win_len) <= 0 or int(hop_len) <= 0:
        raise ValueError("win_len and hop_len must be > 0")
    if int(n_samples) < int(win_len):
        raise ValueError(f"window_len={win_len} exceeds n_samples={n_samples}")
    n_windows = 1 + (int(n_samples) - int(win_len)) // int(hop_len)
    return np.arange(n_windows, dtype=np.int64) * int(hop_len)


def _resolve_window_policy(
    train_trials: Sequence[Dict[str, object]],
    *,
    prop_win_ratio: float,
    prop_hop_ratio: float,
    min_window_extra_channels: int,
    min_hop_len: int,
    force_hop_len: int | None = None,
) -> Dict[str, int]:
    if not train_trials:
        raise ValueError("train_trials cannot be empty")
    first = np.asarray(train_trials[0]["x_trial"], dtype=np.float32)
    channels = int(first.shape[0])
    t_med = int(np.median([int(np.asarray(t["x_trial"], dtype=np.float32).shape[1]) for t in train_trials]))
    win_len = int(max(int(channels + min_window_extra_channels), int(round(float(prop_win_ratio) * float(t_med)))))
    win_len = min(int(win_len), int(t_med))
    hop_len = int(max(int(min_hop_len), int(round(float(prop_hop_ratio) * float(t_med)))))
    if force_hop_len is not None:
        hop_len = int(force_hop_len)
    if int(win_len) <= int(channels):
        raise ValueError(
            f"Invalid dynamic window policy: window_len={win_len} must be > channels={channels}"
        )
    if int(hop_len) <= 0:
        raise ValueError(f"Invalid dynamic hop_len={hop_len}; force_hop_len must be > 0")
    if int(hop_len) >= int(win_len):
        hop_len = int(max(1, int(win_len) - 1))
    assert int(win_len) > int(channels)
    return {
        "channels": int(channels),
        "t_med_train": int(t_med),
        "window_len": int(win_len),
        "hop_len": int(hop_len),
    }


def _build_trial_window_logs(
    trial: Dict[str, object],
    *,
    spd_eps: float,
    win_len: int,
    hop_len: int,
) -> tuple[np.ndarray, List[Dict[str, int | str]]]:
    x = np.asarray(trial["x_trial"], dtype=np.float32)
    starts = _window_starts(int(x.shape[1]), int(win_len), int(hop_len))
    logs: List[np.ndarray] = []
    meta: List[Dict[str, int | str]] = []
    for idx, start in enumerate(starts.tolist()):
        x_win = np.asarray(x[:, int(start) : int(start) + int(win_len)], dtype=np.float32)
        cov = _covariance_from_trial(x_win, float(spd_eps))
        log_cov = logm_spd(np.asarray(cov, dtype=np.float64), float(spd_eps)).astype(np.float32)
        logs.append(np.asarray(log_cov, dtype=np.float32))
        meta.append(
            {
                "trial_id": str(trial["trial_id_str"]),
                "window_index": int(idx),
                "start": int(start),
                "end": int(start) + int(win_len),
            }
        )
    return np.stack(logs, axis=0).astype(np.float32), meta


def _seqs_from_trials(
    trial_dicts: Sequence[Dict[str, object]],
    *,
    spd_eps: float,
    win_len: int,
    hop_len: int,
) -> tuple[List[np.ndarray], List[List[Dict[str, int | str]]]]:
    log_seqs: List[np.ndarray] = []
    meta_list: List[List[Dict[str, int | str]]] = []
    for tr in trial_dicts:
        logs, meta = _build_trial_window_logs(
            tr,
            spd_eps=float(spd_eps),
            win_len=int(win_len),
            hop_len=int(hop_len),
        )
        log_seqs.append(logs)
        meta_list.append(meta)
    return log_seqs, meta_list


def _center_and_vectorize_log_seqs(
    log_seqs: Sequence[np.ndarray],
    *,
    mean_log_train: np.ndarray,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    ref = np.asarray(mean_log_train, dtype=np.float64)
    for seq in log_seqs:
        z_seq = [
            vec_utri(np.asarray(log_cov, dtype=np.float64) - ref).astype(np.float32)
            for log_cov in np.asarray(seq, dtype=np.float32)
        ]
        out.append(np.stack(z_seq, axis=0).astype(np.float32))
    return out


def _make_split(
    *,
    split_name: str,
    trial_dicts: Sequence[Dict[str, object]],
    y: np.ndarray,
    tids: np.ndarray,
    X_static: np.ndarray,
    z_seq_list: Sequence[np.ndarray],
    log_matrix_seq_list: Sequence[np.ndarray],
    window_meta_list: Sequence[List[Dict[str, int | str]]],
) -> TrajectorySplit:
    return TrajectorySplit(
        split_name=str(split_name),
        trial_dicts=list(trial_dicts),
        y=np.asarray(y, dtype=np.int64),
        tids=np.asarray(tids, dtype=object),
        X_static=np.asarray(X_static, dtype=np.float32),
        z_seq_list=[np.asarray(v, dtype=np.float32) for v in z_seq_list],
        log_matrix_seq_list=[np.asarray(v, dtype=np.float32) for v in log_matrix_seq_list],
        window_meta_list=[list(v) for v in window_meta_list],
        meta={
            "n_trials": int(len(trial_dicts)),
            "trajectory_len_mean": float(np.mean([int(v.shape[0]) for v in z_seq_list])) if z_seq_list else 0.0,
            "trajectory_len_min": int(min(int(v.shape[0]) for v in z_seq_list)) if z_seq_list else 0,
            "trajectory_len_max": int(max(int(v.shape[0]) for v in z_seq_list)) if z_seq_list else 0,
        },
    )


def build_trajectory_representation(cfg: TrajectoryRepresentationConfig) -> TrajectoryRepresentationState:
    static_state = build_representation(
        RepresentationConfig(
            dataset=str(cfg.dataset),
            seed=int(cfg.seed),
            val_fraction=float(cfg.val_fraction),
            spd_eps=float(cfg.spd_eps),
            processed_root=str(cfg.processed_root),
            stim_xlsx=str(cfg.stim_xlsx),
            seediv_root=str(cfg.seediv_root),
            seedv_root=str(cfg.seedv_root),
        )
    )
    policy = _resolve_window_policy(
        static_state.train_trial_dicts,
        prop_win_ratio=float(cfg.prop_win_ratio),
        prop_hop_ratio=float(cfg.prop_hop_ratio),
        min_window_extra_channels=int(cfg.min_window_extra_channels),
        min_hop_len=int(cfg.min_hop_len),
        force_hop_len=None if cfg.force_hop_len is None else int(cfg.force_hop_len),
    )
    win_len = int(policy["window_len"])
    hop_len = int(policy["hop_len"])
    channels = int(policy["channels"])
    assert int(win_len) > int(channels)

    train_log_seqs, train_meta = _seqs_from_trials(
        static_state.train_trial_dicts,
        spd_eps=float(cfg.spd_eps),
        win_len=int(win_len),
        hop_len=int(hop_len),
    )
    val_log_seqs, val_meta = _seqs_from_trials(
        static_state.val_trial_dicts,
        spd_eps=float(cfg.spd_eps),
        win_len=int(win_len),
        hop_len=int(hop_len),
    )
    test_log_seqs, test_meta = _seqs_from_trials(
        static_state.test_trial_dicts,
        spd_eps=float(cfg.spd_eps),
        win_len=int(win_len),
        hop_len=int(hop_len),
    )

    all_train_logs = np.concatenate(train_log_seqs, axis=0).astype(np.float32)
    mean_log_train = np.mean(all_train_logs, axis=0).astype(np.float32)

    train_z_seqs = _center_and_vectorize_log_seqs(train_log_seqs, mean_log_train=mean_log_train)
    val_z_seqs = _center_and_vectorize_log_seqs(val_log_seqs, mean_log_train=mean_log_train)
    test_z_seqs = _center_and_vectorize_log_seqs(test_log_seqs, mean_log_train=mean_log_train)

    train_dynamic_cat = np.concatenate(train_z_seqs, axis=0).astype(np.float32)
    static_mean, static_std = _compute_feature_stats(np.asarray(static_state.X_train, dtype=np.float32))
    dynamic_mean, dynamic_std = _compute_feature_stats(train_dynamic_cat)

    train_split = _make_split(
        split_name="train",
        trial_dicts=static_state.train_trial_dicts,
        y=static_state.y_train,
        tids=static_state.tid_train,
        X_static=static_state.X_train,
        z_seq_list=train_z_seqs,
        log_matrix_seq_list=train_log_seqs,
        window_meta_list=train_meta,
    )
    val_split = _make_split(
        split_name="val",
        trial_dicts=static_state.val_trial_dicts,
        y=static_state.y_val,
        tids=static_state.tid_val,
        X_static=static_state.X_val,
        z_seq_list=val_z_seqs,
        log_matrix_seq_list=val_log_seqs,
        window_meta_list=val_meta,
    )
    test_split = _make_split(
        split_name="test",
        trial_dicts=static_state.test_trial_dicts,
        y=static_state.y_test,
        tids=static_state.tid_test,
        X_static=static_state.X_test,
        z_seq_list=test_z_seqs,
        log_matrix_seq_list=test_log_seqs,
        window_meta_list=test_meta,
    )

    return TrajectoryRepresentationState(
        dataset=str(static_state.dataset),
        seed=int(static_state.seed),
        split_meta=dict(static_state.split_meta),
        static_representation=static_state,
        train=train_split,
        val=val_split,
        test=test_split,
        num_classes=int(len(np.unique(np.asarray(static_state.y_train, dtype=np.int64)))),
        channels=int(channels),
        z_dim=int(static_state.X_train.shape[1]) if static_state.X_train.ndim == 2 else 0,
        window_len=int(win_len),
        hop_len=int(hop_len),
        dynamic_mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
        static_feature_mean=np.asarray(static_mean, dtype=np.float32),
        static_feature_std=np.asarray(static_std, dtype=np.float32),
        dynamic_feature_mean=np.asarray(dynamic_mean, dtype=np.float32),
        dynamic_feature_std=np.asarray(dynamic_std, dtype=np.float32),
        meta={
            **dict(static_state.meta),
            "trajectory_window_policy": {
                "window_len": int(win_len),
                "hop_len": int(hop_len),
                "channels": int(channels),
                "t_med_train": int(policy["t_med_train"]),
                "prop_win_ratio": float(cfg.prop_win_ratio),
                "prop_hop_ratio": float(cfg.prop_hop_ratio),
                "min_window_extra_channels": int(cfg.min_window_extra_channels),
                "min_hop_len": int(cfg.min_hop_len),
                "force_hop_len": None if cfg.force_hop_len is None else int(cfg.force_hop_len),
            },
        },
    )
