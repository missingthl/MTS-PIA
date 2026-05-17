from __future__ import annotations

from typing import List

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, class_to_indices, finite_stack, repeat_anchor_indices, rng


def spawner_sameclass_style(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    noise_scale: float = 0.05,
) -> ExternalAugResult:
    try:
        from tslearn.metrics import dtw_path
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("spawner_sameclass_style requires optional dependency `tslearn`.") from exc

    gen = rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = class_to_indices(y_train)
    anchor_idx = repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        x = np.asarray(X_train[anchor_i], dtype=np.float32)
        pool = class_to_idx[cls]
        candidates = pool[pool != anchor_i]
        if len(candidates) == 0:
            mate_i = anchor_i
            fallback_count += 1
        else:
            mate_i = int(gen.choice(candidates))
        mate = np.asarray(X_train[mate_i], dtype=np.float32)
        try:
            path, _ = dtw_path(x.T.astype(np.float64), mate.T.astype(np.float64))
            aligned = np.empty_like(x, dtype=np.float32)
            buckets: List[List[int]] = [[] for _ in range(x.shape[1])]
            for i_t, j_t in path:
                if 0 <= int(i_t) < x.shape[1] and 0 <= int(j_t) < mate.shape[1]:
                    buckets[int(i_t)].append(int(j_t))
            for i_t, js in enumerate(buckets):
                if js:
                    aligned[:, i_t] = np.mean(mate[:, js], axis=1)
                else:
                    aligned[:, i_t] = mate[:, min(i_t, mate.shape[1] - 1)]
        except Exception:
            aligned = mate
            fallback_count += 1
        mixed = 0.5 * x + 0.5 * aligned
        ch_std = np.std(x, axis=1, keepdims=True).astype(np.float32)
        noise = gen.normal(0.0, float(noise_scale), size=x.shape).astype(np.float32) * (ch_std + 1e-6)
        X_out.append(mixed + noise)
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_pattern_mix",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="spawner_style_same_class_dtw_aligned_average",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={"spawner_noise_scale": float(noise_scale)},
    )
