from __future__ import annotations

from typing import List

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, repeat_anchor_indices, rng


def dba_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    k: int = 5,
    max_iter: int = 5,
) -> ExternalAugResult:
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("dba_sameclass requires optional dependency `tslearn`.") from exc

    gen = rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = {int(c): np.flatnonzero(y_train == c) for c in np.unique(y_train)}
    anchor_idx = repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    warning_count = 0

    for anchor in anchor_idx:
        cls = int(y_train[int(anchor)])
        pool = class_to_idx[cls]
        replace = len(pool) < int(k)
        if replace:
            warning_count += 1
        chosen = gen.choice(pool, size=int(k), replace=replace)
        group_tc = np.transpose(np.asarray(X_train[chosen], dtype=np.float64), (0, 2, 1))
        bary_tc = dtw_barycenter_averaging(group_tc, max_iter=int(max_iter))
        X_out.append(np.transpose(np.asarray(bary_tc, dtype=np.float32), (1, 0)))
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=np.stack(X_out, axis=0).astype(np.float32),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_barycenter",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="same_class_dba",
        warning_count=int(warning_count),
        meta={"dba_k": float(k), "dba_max_iter": float(max_iter)},
    )
