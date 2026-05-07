from __future__ import annotations

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, repeat_anchor_indices, rng
from utils.external_baseline_methods.cov_state_common import build_covariance_records, materialize_cov_state_aug


def random_cov_state(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    gamma: float,
) -> ExternalAugResult:
    gen = rng(seed)
    records, mean_log = build_covariance_records(X_train)
    anchor_idx = repeat_anchor_indices(len(X_train), multiplier)
    z_dim = int(records[0]["z"].shape[0])
    dirs = gen.normal(size=(len(anchor_idx), z_dim)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    signs = np.where(np.arange(len(anchor_idx)) % 2 == 0, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
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
        selection_rule="random_unit_z_direction",
        meta={"transport_error_logeuc_mean": transport_err},
    )
