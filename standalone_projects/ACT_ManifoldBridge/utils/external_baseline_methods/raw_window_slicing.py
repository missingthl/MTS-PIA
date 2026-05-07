from __future__ import annotations

from typing import List

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, finite_stack, repeat_anchor_indices, resample_ct, rng


def raw_aug_window_slicing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    slice_ratio: float = 0.90,
    min_window_len: int = 4,
) -> ExternalAugResult:
    gen = rng(seed)
    idx = repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    t = int(X_src.shape[-1])
    X_out: List[np.ndarray] = []
    fallback_count = 0
    for x in X_src:
        slice_len = int(round(float(slice_ratio) * t))
        slice_len = max(int(min_window_len), slice_len)
        slice_len = min(max(1, slice_len), t)
        if t <= 1 or slice_len >= t:
            if t <= 1:
                fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        start = int(gen.integers(0, t - slice_len + 1))
        X_out.append(resample_ct(x[:, start:start + slice_len], t))

    return ExternalAugResult(
        X_aug=finite_stack(X_out),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_window_slicing",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "window_slice_ratio": float(slice_ratio),
            "window_slice_min_window_len": float(min_window_len),
        },
    )
