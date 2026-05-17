from __future__ import annotations

from typing import List, Optional

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, class_to_indices, finite_stack, repeat_anchor_indices, rng


def wdba_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    k: int = 5,
    max_iter: int = 5,
    tau: Optional[float] = None,
) -> ExternalAugResult:
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
        from tslearn.metrics import dtw
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("wdba_sameclass requires optional dependency `tslearn`.") from exc

    gen = rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = class_to_indices(y_train)
    anchor_idx = repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0
    tau_values: List[float] = []
    k_eff = int(k)

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        pool = class_to_idx[cls]
        replace = len(pool) < k_eff
        if replace:
            fallback_count += 1
        chosen = gen.choice(pool, size=k_eff, replace=replace)
        if anchor_i not in chosen:
            chosen[0] = anchor_i
        group_tc = np.transpose(np.asarray(X_train[chosen], dtype=np.float64), (0, 2, 1))
        anchor_tc = np.asarray(X_train[anchor_i], dtype=np.float64).T
        dists = np.asarray([float(dtw(anchor_tc, group_tc[j])) for j in range(k_eff)], dtype=np.float64)
        if tau is None:
            positive = dists[dists > 1e-12]
            tau_i = float(np.median(positive)) if positive.size else 1.0
            if not np.isfinite(tau_i) or tau_i <= 1e-12:
                tau_i = 1.0
                fallback_count += 1
        else:
            tau_i = float(tau)
        tau_values.append(tau_i)
        logits = -dists / max(tau_i, 1e-12)
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        weights /= float(np.sum(weights) + 1e-12)
        try:
            bary_tc = dtw_barycenter_averaging(group_tc, weights=weights, max_iter=int(max_iter))
        except Exception:
            fallback_count += 1
            bary_tc = np.average(group_tc, axis=0, weights=weights)
        X_out.append(np.transpose(np.asarray(bary_tc, dtype=np.float32), (1, 0)))
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_barycenter",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="same_class_weighted_dba_anchor_dtw_softmax",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "wdba_k": float(k),
            "wdba_max_iter": float(max_iter),
            "wdba_tau": float(np.mean(tau_values)) if tau_values else float("nan"),
        },
    )
