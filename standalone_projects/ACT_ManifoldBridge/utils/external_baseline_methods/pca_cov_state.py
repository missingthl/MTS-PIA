from __future__ import annotations

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, repeat_anchor_indices
from utils.external_baseline_methods.cov_state_common import build_covariance_records, materialize_cov_state_aug


def pca_cov_state(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    gamma: float,
    k_dir: int,
) -> ExternalAugResult:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("pca_cov_state requires optional dependency `scikit-learn`.") from exc

    records, mean_log = build_covariance_records(X_train)
    Z = np.stack([rec["z"] for rec in records], axis=0)
    n_components = max(1, min(int(k_dir), int(Z.shape[0]), int(Z.shape[1])))
    pca = PCA(n_components=n_components, random_state=int(seed))
    pca.fit(Z)
    components = np.asarray(pca.components_, dtype=np.float32)
    components /= np.linalg.norm(components, axis=1, keepdims=True) + 1e-12
    anchor_idx = repeat_anchor_indices(len(X_train), multiplier)
    slots = np.arange(len(anchor_idx), dtype=np.int64)
    dirs = components[slots % n_components]
    signs = np.where((slots // n_components) % 2 == 0, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    z0 = np.stack([records[int(i)]["z"] for i in anchor_idx], axis=0)
    z_cands = z0 + signs * float(gamma) * dirs
    X_aug, transport_err = materialize_cov_state_aug(records, mean_log, z_cands, anchor_idx)
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[anchor_idx], dtype=np.int64),
        source_space="covariance_state",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="pca_top_z_direction",
        meta={
            "pca_n_components": float(n_components),
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "transport_error_logeuc_mean": transport_err,
        },
    )
