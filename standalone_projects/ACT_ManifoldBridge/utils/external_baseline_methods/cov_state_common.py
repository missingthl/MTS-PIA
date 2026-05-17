from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from core.bridge import bridge_single, logvec_to_spd


def build_covariance_records(X_train: np.ndarray, spd_eps: float = 1e-4) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    records: List[Dict[str, np.ndarray]] = []
    log_covs = []
    for x_np in np.asarray(X_train, dtype=np.float32):
        x = torch.from_numpy(x_np).double()
        x = x - x.mean(dim=-1, keepdim=True)
        cov = (x @ x.transpose(-1, -2)) / float(max(1, x.shape[-1] - 1))
        cov = cov + float(spd_eps) * torch.eye(cov.shape[0], dtype=cov.dtype)
        vals, vecs = torch.linalg.eigh(cov)
        log_cov = vecs @ torch.diag_embed(torch.log(torch.clamp(vals, min=spd_eps))) @ vecs.transpose(-1, -2)
        log_covs.append(log_cov.numpy())
        records.append({"x_raw": x_np, "sigma_orig": cov.numpy(), "log_cov": log_cov.numpy()})

    mean_log = np.mean(log_covs, axis=0)
    idx = np.triu_indices(mean_log.shape[0])
    for record in records:
        record["z"] = (record["log_cov"] - mean_log)[idx].astype(np.float32)
    return records, mean_log.astype(np.float64)


def materialize_cov_state_aug(
    records: List[Dict[str, np.ndarray]],
    mean_log: np.ndarray,
    z_cands: np.ndarray,
    anchor_idx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    X_aug = []
    transport_errors = []
    for z, idx in zip(z_cands, anchor_idx):
        rec = records[int(idx)]
        sigma_aug = logvec_to_spd(np.asarray(z, dtype=np.float32), mean_log)
        x_aug, meta = bridge_single(
            torch.from_numpy(rec["x_raw"]),
            torch.from_numpy(rec["sigma_orig"]),
            torch.from_numpy(sigma_aug),
        )
        X_aug.append(x_aug.cpu().numpy().astype(np.float32))
        transport_errors.append(float(meta.get("transport_error_logeuc", np.nan)))
    mean_err = float(np.nanmean(transport_errors)) if transport_errors else float("nan")
    return np.stack(X_aug, axis=0).astype(np.float32), mean_err
