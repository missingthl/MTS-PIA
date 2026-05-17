from __future__ import annotations

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult


def raw_smote_flatten_balanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> ExternalAugResult:
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_smote_flatten_balanced requires optional dependency `imbalanced-learn`.") from exc

    y_train = np.asarray(y_train, dtype=np.int64)
    _, counts = np.unique(y_train, return_counts=True)
    if counts.size == 0 or int(counts.min()) < 2:
        empty = np.empty((0, X_train.shape[1], X_train.shape[2]), dtype=np.float32)
        return ExternalAugResult(
            X_aug=empty,
            y_aug=np.empty((0,), dtype=np.int64),
            source_space="flattened_raw",
            label_mode="hard",
            uses_external_library=True,
            library_name="imbalanced-learn",
            budget_matched=False,
            selection_rule="class_balancing_smote_auto",
            warning_count=1,
        )

    k_neighbors = max(1, min(5, int(counts.min()) - 1))
    flat = np.asarray(X_train, dtype=np.float32).reshape(len(X_train), -1)
    smote = SMOTE(sampling_strategy="auto", random_state=int(seed), k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(flat, y_train)
    X_new = np.asarray(X_res[len(X_train):], dtype=np.float32).reshape(-1, X_train.shape[1], X_train.shape[2])
    y_new = np.asarray(y_res[len(y_train):], dtype=np.int64)
    return ExternalAugResult(
        X_aug=X_new,
        y_aug=y_new,
        source_space="flattened_raw",
        label_mode="hard",
        uses_external_library=True,
        library_name="imbalanced-learn",
        budget_matched=False,
        selection_rule="class_balancing_smote_auto",
        meta={"smote_k_neighbors": float(k_neighbors)},
    )
