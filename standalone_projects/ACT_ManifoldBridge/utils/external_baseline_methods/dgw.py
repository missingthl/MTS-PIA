from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, class_to_indices, finite_stack, repeat_anchor_indices, rng
from utils.external_baseline_methods.dtw_helpers import dtw_path_tc, guided_warp_ct, window_slice_ct


def dgw_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    batch_size: int = 6,
    slope_constraint: str = "symmetric",
    use_window: bool = True,
    use_variable_slice: bool = True,
    min_window_len: int = 4,
) -> ExternalAugResult:
    """Discriminative Guided Warping clean-room adapter."""
    gen = rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = class_to_indices(y_train)
    all_idx = np.arange(len(X_train), dtype=np.int64)
    anchor_idx = repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0
    dtw_values: List[float] = []
    warp_amounts: List[float] = []
    score_values: List[float] = []
    positive_batch = max(1, int(np.ceil(float(batch_size) / 2.0)))
    negative_batch = max(1, int(np.floor(float(batch_size) / 2.0)))

    def distance_ct(a_ct: np.ndarray, b_ct: np.ndarray) -> float:
        _, _, dist = dtw_path_tc(
            np.asarray(a_ct, dtype=np.float32).T,
            np.asarray(b_ct, dtype=np.float32).T,
            slope_constraint=slope_constraint,
            window=int(np.ceil(a_ct.shape[1] / 10.0)) if use_window else None,
        )
        return float(dist)

    warped_items: List[Tuple[np.ndarray, float]] = []
    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        positive = class_to_idx[cls]
        positive = positive[positive != anchor_i]
        negative = all_idx[y_train != cls]
        if len(positive) == 0 or len(negative) == 0:
            warped_items.append((np.asarray(X_train[anchor_i], dtype=np.float32).copy(), 0.0))
            y_out.append(cls)
            fallback_count += 1
            continue

        pos_chosen = gen.choice(positive, size=min(len(positive), positive_batch), replace=False)
        neg_chosen = gen.choice(negative, size=min(len(negative), negative_batch), replace=False)
        best_score = -np.inf
        best_proto_i = int(pos_chosen[0])
        for proto_i in pos_chosen:
            proto = np.asarray(X_train[int(proto_i)], dtype=np.float32)
            other_pos = [int(x) for x in pos_chosen if int(x) != int(proto_i)]
            try:
                pos_dist = float(np.mean([distance_ct(proto, np.asarray(X_train[j], dtype=np.float32)) for j in other_pos])) if other_pos else 0.0
                neg_dist = float(np.mean([distance_ct(proto, np.asarray(X_train[int(j)], dtype=np.float32)) for j in neg_chosen]))
                score = neg_dist - pos_dist
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score = float(score)
                best_proto_i = int(proto_i)

        try:
            x_aug, dtw_value, warp_amount = guided_warp_ct(
                np.asarray(X_train[anchor_i], dtype=np.float32),
                np.asarray(X_train[best_proto_i], dtype=np.float32),
                slope_constraint=slope_constraint,
                use_window=bool(use_window),
            )
        except Exception:
            x_aug = np.asarray(X_train[anchor_i], dtype=np.float32).copy()
            dtw_value = float("nan")
            warp_amount = 0.0
            fallback_count += 1
        warped_items.append((x_aug, float(warp_amount)))
        y_out.append(cls)
        dtw_values.append(float(dtw_value))
        warp_amounts.append(float(warp_amount))
        score_values.append(float(best_score))

    max_warp = max([amount for _, amount in warped_items], default=0.0)
    for x_aug, warp_amount in warped_items:
        if use_variable_slice:
            reduce_ratio = 0.9 + 0.1 * float(warp_amount) / float(max_warp) if max_warp > 1e-12 else 0.9
            x_aug = window_slice_ct(x_aug, reduce_ratio=reduce_ratio, rng=gen, min_window_len=min_window_len)
        X_out.append(x_aug)

    return ExternalAugResult(
        X_aug=finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_guided_warp",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="discriminative_guided_warp_same_class_dtw_cleanroom",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "guided_warp_mode": "dgw",
            "guided_warp_batch_size": float(batch_size),
            "guided_warp_positive_batch": float(positive_batch),
            "guided_warp_negative_batch": float(negative_batch),
            "guided_warp_slope_constraint": str(slope_constraint),
            "guided_warp_use_window": float(bool(use_window)),
            "guided_warp_use_variable_slice": float(bool(use_variable_slice)),
            "guided_warp_dtw_value_mean": float(np.nanmean(dtw_values)) if dtw_values else float("nan"),
            "guided_warp_amount_mean": float(np.nanmean(warp_amounts)) if warp_amounts else float("nan"),
            "guided_warp_score_mean": float(np.nanmean(score_values)) if score_values else float("nan"),
            "guided_warp_cleanroom_adapter": 1.0,
        },
    )
