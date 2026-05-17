from __future__ import annotations

from typing import List

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, repeat_anchor_indices, rng


def raw_aug_magnitude_warping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    sigma: float = 0.2,
    knots: int = 4,
    per_channel_curve: bool = True,
) -> ExternalAugResult:
    try:
        from scipy.interpolate import CubicSpline
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_aug_magnitude_warping requires optional dependency `scipy`.") from exc

    gen = rng(seed)
    idx = repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    n_aug, c, t = X_src.shape
    n_knots = max(1, int(knots))
    x_knots = np.linspace(0.0, float(max(t - 1, 1)), n_knots + 2)
    x_full = np.arange(t, dtype=np.float64)
    X_out = np.empty_like(X_src, dtype=np.float32)

    for i in range(n_aug):
        n_curves = c if per_channel_curve else 1
        knot_vals = gen.normal(1.0, float(sigma), size=(n_curves, n_knots + 2))
        knot_vals = np.clip(knot_vals, 0.05, None)
        curves: List[np.ndarray] = []
        for curve_idx in range(n_curves):
            curve = CubicSpline(x_knots, knot_vals[curve_idx], bc_type="natural")(x_full)
            curves.append(np.clip(curve, 0.05, None).astype(np.float32))
        curve_arr = np.stack(curves, axis=0)
        if not per_channel_curve:
            curve_arr = np.repeat(curve_arr, c, axis=0)
        X_out[i] = X_src[i] * curve_arr

    return ExternalAugResult(
        X_aug=np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_magnitude_warping",
        meta={
            "warp_sigma": float(sigma),
            "warp_knots": float(knots),
            "per_channel_curve": float(bool(per_channel_curve)),
        },
    )
