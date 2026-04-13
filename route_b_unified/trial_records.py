from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from route_b_unified.spd_features import logm_spd, vec_utri


def _covariance_from_trial(x: np.ndarray, eps: float) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    xx = xx - xx.mean(axis=1, keepdims=True)
    denom = max(1, int(xx.shape[1]) - 1)
    cov = (xx @ xx.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(eps) * np.eye(cov.shape[0], dtype=np.float64)
    return cov.astype(np.float32)


@dataclass
class TrialRecord:
    tid: str
    y: int
    x_raw: np.ndarray
    sigma_orig: np.ndarray
    log_cov: np.ndarray
    z: np.ndarray


def _build_trial_records(trials: Sequence[Dict], spd_eps: float) -> Tuple[List[TrialRecord], np.ndarray]:
    covs = [_covariance_from_trial(np.asarray(t["x_trial"], dtype=np.float32), float(spd_eps)) for t in trials]
    log_covs = [logm_spd(np.asarray(c, dtype=np.float64), float(spd_eps)).astype(np.float32) for c in covs]
    mean_log = np.mean(np.stack(log_covs, axis=0), axis=0).astype(np.float32) if log_covs else np.zeros((0, 0), dtype=np.float32)
    out: List[TrialRecord] = []
    for t, cov, log_cov in zip(trials, covs, log_covs):
        z = vec_utri(np.asarray(log_cov, dtype=np.float64) - np.asarray(mean_log, dtype=np.float64)).astype(np.float32)
        out.append(
            TrialRecord(
                tid=str(t["trial_id_str"]),
                y=int(t["label"]),
                x_raw=np.asarray(t["x_trial"], dtype=np.float32),
                sigma_orig=np.asarray(cov, dtype=np.float32),
                log_cov=np.asarray(log_cov, dtype=np.float32),
                z=np.asarray(z, dtype=np.float32),
            )
        )
    return out, mean_log


def _apply_mean_log(records: Sequence[TrialRecord], mean_log: np.ndarray) -> List[TrialRecord]:
    out: List[TrialRecord] = []
    ref = np.asarray(mean_log, dtype=np.float64)
    for r in records:
        z = vec_utri(np.asarray(r.log_cov, dtype=np.float64) - ref).astype(np.float32)
        out.append(
            TrialRecord(
                tid=str(r.tid),
                y=int(r.y),
                x_raw=np.asarray(r.x_raw, dtype=np.float32),
                sigma_orig=np.asarray(r.sigma_orig, dtype=np.float32),
                log_cov=np.asarray(r.log_cov, dtype=np.float32),
                z=z,
            )
        )
    return out
