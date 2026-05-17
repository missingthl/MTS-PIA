from __future__ import annotations

from typing import List

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, class_to_indices, finite_stack, repeat_anchor_indices, rng
from utils.external_baseline_methods.dtw_helpers import guided_warp_ct


def rgw_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    slope_constraint: str = "symmetric",
    use_window: bool = True,
) -> ExternalAugResult:
    """Random Guided Warping clean-room adapter."""
    gen = rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = class_to_indices(y_train)
    anchor_idx = repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0
    dtw_values: List[float] = []
    warp_amounts: List[float] = []

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        pool = class_to_idx[cls]
        candidates = pool[pool != anchor_i]
        if len(candidates) == 0:
            X_out.append(np.asarray(X_train[anchor_i], dtype=np.float32).copy())
            y_out.append(cls)
            fallback_count += 1
            continue
        prototype_i = int(gen.choice(candidates))
        try:
            x_aug, dtw_value, warp_amount = guided_warp_ct(
                np.asarray(X_train[anchor_i], dtype=np.float32),
                np.asarray(X_train[prototype_i], dtype=np.float32),
                slope_constraint=slope_constraint,
                use_window=bool(use_window),
            )
        except Exception:
            x_aug = np.asarray(X_train[anchor_i], dtype=np.float32).copy()
            dtw_value = float("nan")
            warp_amount = 0.0
            fallback_count += 1
        X_out.append(x_aug)
        y_out.append(cls)
        dtw_values.append(float(dtw_value))
        warp_amounts.append(float(warp_amount))

    return ExternalAugResult(
        X_aug=finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_guided_warp",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="random_guided_warp_same_class_dtw_cleanroom",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "guided_warp_mode": "rgw",
            "guided_warp_slope_constraint": str(slope_constraint),
            "guided_warp_use_window": float(bool(use_window)),
            "guided_warp_dtw_value_mean": float(np.nanmean(dtw_values)) if dtw_values else float("nan"),
            "guided_warp_amount_mean": float(np.nanmean(warp_amounts)) if warp_amounts else float("nan"),
            "guided_warp_cleanroom_adapter": 1.0,
        },
    )
