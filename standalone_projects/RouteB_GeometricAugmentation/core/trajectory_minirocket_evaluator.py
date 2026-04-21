from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from core.trajectory_representation import TrajectoryRepresentationState


@dataclass(frozen=True)
class TrajectoryMiniRocketEvalConfig:
    n_kernels: int = 10000
    n_jobs: int = 1
    padding_mode: str = "edge"
    target_len_mode: str = "train_max_len"


@dataclass
class TrajectoryMiniRocketEvalResult:
    dataset: str
    seed: int
    variant: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    history_rows: List[Dict[str, float]] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _normalize_sequence(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return ((arr - np.asarray(mean, dtype=np.float32)[None, :]) / (np.asarray(std, dtype=np.float32)[None, :] + 1e-6)).astype(
        np.float32
    )


def _pad_or_truncate_edge(seq: np.ndarray, target_len: int) -> Tuple[np.ndarray, Dict[str, float]]:
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D] trajectory sequence, got shape={arr.shape}")
    cur_len = int(arr.shape[0])
    feat_dim = int(arr.shape[1])
    if int(target_len) <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")
    if cur_len == int(target_len):
        return arr, {
            "orig_len": float(cur_len),
            "target_len": float(target_len),
            "pad_count": 0.0,
            "truncate_count": 0.0,
            "padding_ratio": 0.0,
            "truncate_ratio": 0.0,
        }
    if cur_len > int(target_len):
        out = np.asarray(arr[: int(target_len), :], dtype=np.float32)
        trunc = int(cur_len - int(target_len))
        return out, {
            "orig_len": float(cur_len),
            "target_len": float(target_len),
            "pad_count": 0.0,
            "truncate_count": float(trunc),
            "padding_ratio": 0.0,
            "truncate_ratio": float(trunc) / float(cur_len),
        }
    pad = int(target_len) - cur_len
    if cur_len <= 0:
        out = np.zeros((int(target_len), feat_dim), dtype=np.float32)
    else:
        tail = np.repeat(arr[-1:, :], repeats=int(pad), axis=0).astype(np.float32)
        out = np.concatenate([arr, tail], axis=0).astype(np.float32)
    return out, {
        "orig_len": float(cur_len),
        "target_len": float(target_len),
        "pad_count": float(pad),
        "truncate_count": 0.0,
        "padding_ratio": float(pad) / float(target_len),
        "truncate_ratio": 0.0,
    }


def _seqs_to_collection(
    seqs: Sequence[np.ndarray],
    *,
    target_len: int,
    padding_mode: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if str(padding_mode).lower() != "edge":
        raise ValueError(f"Unsupported padding_mode={padding_mode}; T8-pre locks this to edge")
    rows: List[np.ndarray] = []
    meta_rows: List[Dict[str, float]] = []
    for seq in seqs:
        fixed, info = _pad_or_truncate_edge(np.asarray(seq, dtype=np.float32), int(target_len))
        rows.append(np.asarray(fixed.T, dtype=np.float32))
        meta_rows.append(info)
    if not rows:
        raise ValueError("seqs cannot be empty")
    arr = np.stack(rows, axis=0).astype(np.float32)
    return arr, {
        "count": int(len(meta_rows)),
        "target_len": int(target_len),
        "orig_len_mean": float(np.mean([row["orig_len"] for row in meta_rows])),
        "orig_len_min": float(np.min([row["orig_len"] for row in meta_rows])),
        "orig_len_max": float(np.max([row["orig_len"] for row in meta_rows])),
        "pad_count_mean": float(np.mean([row["pad_count"] for row in meta_rows])),
        "truncate_count_mean": float(np.mean([row["truncate_count"] for row in meta_rows])),
        "padding_ratio_mean": float(np.mean([row["padding_ratio"] for row in meta_rows])),
        "truncate_ratio_mean": float(np.mean([row["truncate_ratio"] for row in meta_rows])),
        "n_padded": int(sum(1 for row in meta_rows if row["pad_count"] > 0.0)),
        "n_truncated": int(sum(1 for row in meta_rows if row["truncate_count"] > 0.0)),
    }


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "loss": 0.0,
    }


def evaluate_dynamic_minirocket_classifier(
    state: TrajectoryRepresentationState,
    *,
    seed: int,
    eval_cfg: TrajectoryMiniRocketEvalConfig,
) -> TrajectoryMiniRocketEvalResult:
    try:
        from aeon.classification.convolution_based import MiniRocketClassifier
    except Exception as e:
        raise ImportError(
            "dynamic_minirocket requires aeon in the active environment. Install in pia env: `pip install aeon`."
        ) from e

    _set_seed(int(seed))
    if str(eval_cfg.target_len_mode).lower() != "train_max_len":
        raise ValueError(f"Unsupported target_len_mode={eval_cfg.target_len_mode}; T8-pre locks this to train_max_len")

    train_seqs = [
        _normalize_sequence(np.asarray(v, dtype=np.float32), state.dynamic_feature_mean, state.dynamic_feature_std)
        for v in state.train.z_seq_list
    ]
    val_seqs = [
        _normalize_sequence(np.asarray(v, dtype=np.float32), state.dynamic_feature_mean, state.dynamic_feature_std)
        for v in state.val.z_seq_list
    ]
    test_seqs = [
        _normalize_sequence(np.asarray(v, dtype=np.float32), state.dynamic_feature_mean, state.dynamic_feature_std)
        for v in state.test.z_seq_list
    ]
    y_train = np.asarray(state.train.y, dtype=np.int64)
    y_val = np.asarray(state.val.y, dtype=np.int64)
    y_test = np.asarray(state.test.y, dtype=np.int64)

    train_max_len = int(max(int(np.asarray(seq).shape[0]) for seq in train_seqs))
    # MiniROCKET requires at least 9 timepoints. We keep the pad policy fixed and
    # only raise the target length to the minimal admissible floor when needed.
    model_input_len = int(max(int(train_max_len), 9))
    X_train, train_pad_meta = _seqs_to_collection(train_seqs, target_len=int(model_input_len), padding_mode=str(eval_cfg.padding_mode))
    X_val, val_pad_meta = _seqs_to_collection(val_seqs, target_len=int(model_input_len), padding_mode=str(eval_cfg.padding_mode))
    X_test, test_pad_meta = _seqs_to_collection(test_seqs, target_len=int(model_input_len), padding_mode=str(eval_cfg.padding_mode))

    clf = MiniRocketClassifier(
        n_kernels=int(eval_cfg.n_kernels),
        random_state=int(seed),
        n_jobs=int(eval_cfg.n_jobs),
    )
    clf.fit(X_train, y_train)
    y_pred_train = np.asarray(clf.predict(X_train), dtype=np.int64)
    y_pred_val = np.asarray(clf.predict(X_val), dtype=np.int64)
    y_pred_test = np.asarray(clf.predict(X_test), dtype=np.int64)

    return TrajectoryMiniRocketEvalResult(
        dataset=str(state.dataset),
        seed=int(seed),
        variant="dynamic_minirocket",
        train_metrics=_evaluate_predictions(y_train, y_pred_train),
        val_metrics=_evaluate_predictions(y_val, y_pred_val),
        test_metrics=_evaluate_predictions(y_test, y_pred_test),
        meta={
            "n_kernels": int(eval_cfg.n_kernels),
            "n_jobs": int(eval_cfg.n_jobs),
            "padding_mode": str(eval_cfg.padding_mode),
            "target_len_mode": str(eval_cfg.target_len_mode),
            "train_max_len": int(train_max_len),
            "model_input_len": int(model_input_len),
            "train_padding": dict(train_pad_meta),
            "val_padding": dict(val_pad_meta),
            "test_padding": dict(test_pad_meta),
            "feature_dim": int(state.z_dim),
            "num_classes": int(state.num_classes),
        },
    )
