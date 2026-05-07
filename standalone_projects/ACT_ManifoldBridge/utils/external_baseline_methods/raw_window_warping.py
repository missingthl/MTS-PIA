from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, finite_stack, repeat_anchor_indices, resample_ct, rng


def raw_aug_window_warping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    window_ratio: float = 0.10,
    speed_factors: Tuple[float, ...] = (0.5, 2.0),
    min_window_len: int = 4,
) -> ExternalAugResult:
    gen = rng(seed)
    idx = repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    t = int(X_src.shape[-1])
    X_out: List[np.ndarray] = []
    fallback_count = 0
    for x in X_src:
        if t < 3:
            fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        win_len = int(round(float(window_ratio) * t))
        win_len = max(int(min_window_len), win_len)
        win_len = min(max(1, win_len), max(1, t - 1))
        if win_len >= t:
            fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        start = int(gen.integers(0, t - win_len + 1))
        speed = float(gen.choice(np.asarray(speed_factors, dtype=np.float64)))
        warped_len = max(1, int(round(win_len * speed)))
        before = x[:, :start]
        segment = x[:, start:start + win_len]
        after = x[:, start + win_len:]
        warped = resample_ct(segment, warped_len)
        stitched = np.concatenate([before, warped, after], axis=1)
        X_out.append(resample_ct(stitched, t))

    return ExternalAugResult(
        X_aug=finite_stack(X_out),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_window_warping",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "window_warp_ratio": float(window_ratio),
            "window_warp_min_window_len": float(min_window_len),
            "window_warp_speed_min": float(np.min(speed_factors)),
            "window_warp_speed_max": float(np.max(speed_factors)),
        },
    )
