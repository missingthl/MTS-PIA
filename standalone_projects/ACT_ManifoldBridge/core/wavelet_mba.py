from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from core.bridge import bridge_single, logvec_to_spd
from core.curriculum import estimate_local_manifold_margins


def _require_pywt():
    try:
        import pywt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyWavelets is required for --pipeline wavelet_mba. "
            "Install it with `pip install PyWavelets` or update the ACT conda env."
        ) from exc
    return pywt


@dataclass
class WaveletTrialRecord:
    tid: str
    y: int
    x_raw: np.ndarray
    coeffs: List[np.ndarray]
    cA: np.ndarray
    sigma_a: np.ndarray
    z_a: np.ndarray
    wavelet_level_eff: int
    wavelet_recon_error: float


def parse_step_tier_ratios(text: str) -> List[float]:
    out: List[float] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        value = float(t)
        if value <= 0.0 or value > 1.0:
            raise ValueError("wavelet step tier ratios must be in (0, 1].")
        out.append(value)
    if not out:
        raise ValueError("wavelet step tier ratios cannot be empty.")
    return sorted(out)


def resolve_wavelet_level(length: int, wavelet_name: str, wavelet_level: str | int) -> int:
    pywt = _require_pywt()
    wavelet = pywt.Wavelet(str(wavelet_name))
    max_level = int(pywt.dwt_max_level(int(length), wavelet.dec_len))
    if str(wavelet_level).lower() == "auto":
        level = min(2, max_level)
    else:
        level = int(wavelet_level)
    if level < 1:
        raise ValueError(
            f"Wavelet level must be >= 1 for wavelet_mba; got {wavelet_level!r} "
            f"with max_level={max_level}."
        )
    if level > max_level:
        raise ValueError(
            f"Requested wavelet level {level} exceeds max_level={max_level} "
            f"for length={length} and wavelet={wavelet_name}."
        )
    return int(level)


def decompose_signal(
    x: np.ndarray,
    *,
    wavelet_name: str,
    wavelet_level: str | int,
    wavelet_mode: str,
) -> Tuple[List[np.ndarray], int, float]:
    pywt = _require_pywt()
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2:
        raise ValueError("wavelet_mba expects trial arrays shaped [channels, length].")
    level_eff = resolve_wavelet_level(xx.shape[-1], wavelet_name, wavelet_level)
    coeffs = pywt.wavedec(xx, wavelet=wavelet_name, mode=wavelet_mode, level=level_eff, axis=-1)
    coeffs = [np.asarray(c, dtype=np.float64) for c in coeffs]
    recon = pywt.waverec(coeffs, wavelet=wavelet_name, mode=wavelet_mode, axis=-1)
    recon = np.asarray(recon, dtype=np.float64)[..., : xx.shape[-1]]
    recon_error = float(np.mean(np.abs(recon - xx)))
    return coeffs, level_eff, recon_error


def reconstruct_signal(
    coeffs: Sequence[np.ndarray],
    *,
    original_length: int,
    wavelet_name: str,
    wavelet_mode: str,
) -> np.ndarray:
    pywt = _require_pywt()
    recon = pywt.waverec(
        [np.asarray(c, dtype=np.float64) for c in coeffs],
        wavelet=wavelet_name,
        mode=wavelet_mode,
        axis=-1,
    )
    recon = np.asarray(recon, dtype=np.float64)
    if recon.shape[-1] < int(original_length):
        pad = int(original_length) - int(recon.shape[-1])
        recon = np.pad(recon, [(0, 0), (0, pad)], mode="edge")
    return recon[..., : int(original_length)].astype(np.float32)


def covariance_and_logvec(cA: np.ndarray, *, mean_log: np.ndarray | None, spd_eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.asarray(cA, dtype=np.float64)).double()
    x = x - x.mean(dim=-1, keepdim=True)
    denom = max(1, int(x.shape[-1]) - 1)
    cov = (x @ x.transpose(-1, -2)) / float(denom)
    cov = cov + float(spd_eps) * torch.eye(cov.shape[0], dtype=cov.dtype)
    vals, vecs = torch.linalg.eigh(cov)
    log_cov = vecs @ torch.diag_embed(torch.log(torch.clamp(vals, min=spd_eps))) @ vecs.transpose(-1, -2)
    log_cov_np = log_cov.numpy()
    if mean_log is None:
        z = np.zeros((np.triu_indices(log_cov_np.shape[0])[0].size,), dtype=np.float64)
    else:
        idx = np.triu_indices(mean_log.shape[0])
        z = (log_cov_np - mean_log)[idx]
    return cov.numpy(), log_cov_np, z.astype(np.float64)


def build_wavelet_trial_records(
    trials,
    *,
    wavelet_name: str,
    wavelet_level: str | int,
    wavelet_mode: str,
    spd_eps: float = 1e-4,
) -> Tuple[List[WaveletTrialRecord], np.ndarray | None, Dict[str, object]]:
    if not trials:
        return [], None, {}

    raw_rows: List[Dict[str, object]] = []
    log_covs: List[np.ndarray] = []
    cA_shape = None
    level_eff = None
    recon_errors: List[float] = []
    for t in trials:
        coeffs, level_i, recon_error = decompose_signal(
            t.x,
            wavelet_name=wavelet_name,
            wavelet_level=wavelet_level,
            wavelet_mode=wavelet_mode,
        )
        cA = coeffs[0]
        if cA_shape is None:
            cA_shape = tuple(cA.shape)
            level_eff = int(level_i)
        elif tuple(cA.shape) != cA_shape:
            raise ValueError(f"Inconsistent cA shape: got {tuple(cA.shape)}, expected {cA_shape}.")
        if int(level_i) != int(level_eff):
            raise ValueError(f"Inconsistent effective wavelet level: got {level_i}, expected {level_eff}.")
        sigma_a, log_cov, _ = covariance_and_logvec(cA, mean_log=None, spd_eps=spd_eps)
        log_covs.append(log_cov)
        recon_errors.append(recon_error)
        raw_rows.append(
            {
                "tid": t.tid,
                "y": int(t.y),
                "x_raw": np.asarray(t.x, dtype=np.float32),
                "coeffs": coeffs,
                "cA": cA,
                "sigma_a": sigma_a,
                "recon_error": recon_error,
            }
        )

    mean_log = np.mean(log_covs, axis=0)
    records: List[WaveletTrialRecord] = []
    for row, log_cov in zip(raw_rows, log_covs):
        idx = np.triu_indices(mean_log.shape[0])
        z_a = (log_cov - mean_log)[idx].astype(np.float64)
        records.append(
            WaveletTrialRecord(
                tid=str(row["tid"]),
                y=int(row["y"]),
                x_raw=np.asarray(row["x_raw"], dtype=np.float32),
                coeffs=[np.asarray(c, dtype=np.float64) for c in row["coeffs"]],
                cA=np.asarray(row["cA"], dtype=np.float64),
                sigma_a=np.asarray(row["sigma_a"], dtype=np.float64),
                z_a=z_a,
                wavelet_level_eff=int(level_eff),
                wavelet_recon_error=float(row["recon_error"]),
            )
        )

    meta = {
        "wavelet_level_eff": int(level_eff),
        "cA_shape": tuple(cA_shape or ()),
        "cA_length": int(cA_shape[-1]) if cA_shape else 0,
        "cD_frozen_count": int(level_eff),
        "wavelet_recon_error_mean": float(np.mean(recon_errors)),
    }
    return records, mean_log, meta


def assert_frozen_details(original_coeffs: Sequence[np.ndarray], candidate_coeffs: Sequence[np.ndarray]) -> None:
    if len(original_coeffs) != len(candidate_coeffs):
        raise AssertionError("DWT coefficient list length changed during wavelet_mba realization.")
    for level_idx, (orig, cand) in enumerate(zip(original_coeffs[1:], candidate_coeffs[1:]), start=1):
        if not np.array_equal(np.asarray(orig), np.asarray(cand)):
            raise AssertionError(f"Frozen cD mismatch at coefficient index {level_idx}.")


def compute_identity_checks(
    records: Sequence[WaveletTrialRecord],
    *,
    wavelet_name: str,
    wavelet_mode: str,
    max_items: int = 32,
) -> Dict[str, float]:
    if not records:
        return {"cA_identity_bridge_error_mean": 0.0, "idwt_identity_recon_error_mean": 0.0}
    cA_errs: List[float] = []
    idwt_errs: List[float] = []
    for rec in list(records)[: int(max_items)]:
        cA_same, _ = bridge_single(
            torch.from_numpy(rec.cA),
            torch.from_numpy(rec.sigma_a),
            torch.from_numpy(rec.sigma_a),
        )
        cA_np = cA_same.numpy().astype(np.float64)
        cA_errs.append(float(np.mean(np.abs(cA_np - rec.cA))))
        cand_coeffs = [cA_np] + [np.asarray(c, dtype=np.float64).copy() for c in rec.coeffs[1:]]
        assert_frozen_details(rec.coeffs, cand_coeffs)
        x_recon = reconstruct_signal(
            cand_coeffs,
            original_length=rec.x_raw.shape[-1],
            wavelet_name=wavelet_name,
            wavelet_mode=wavelet_mode,
        )
        idwt_errs.append(float(np.mean(np.abs(x_recon.astype(np.float64) - rec.x_raw.astype(np.float64)))))
    return {
        "cA_identity_bridge_error_mean": float(np.mean(cA_errs)),
        "idwt_identity_recon_error_mean": float(np.mean(idwt_errs)),
    }


def build_step_tier_candidates(
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    tid_train: np.ndarray,
    *,
    direction_bank: np.ndarray,
    gamma_by_dir: np.ndarray,
    tier_ratios: Sequence[float],
    seed: int,
    eta_safe: float | None = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    y_arr = np.asarray(y_train).astype(int).ravel()
    tid_arr = np.asarray(tid_train)
    X = np.asarray(X_train_z, dtype=np.float64)
    W = np.asarray(direction_bank, dtype=np.float64)
    actual_k = int(W.shape[0])
    if actual_k <= 0:
        return (
            np.empty((0, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int64),
            {"aug_total_count": 0, "step_tier_count": 0},
        )

    gammas = np.asarray(gamma_by_dir, dtype=np.float64).ravel()
    if gammas.size != actual_k:
        gammas = np.resize(gammas, actual_k)
    ratios = list(parse_step_tier_ratios(",".join(str(v) for v in tier_ratios)))
    margins = estimate_local_manifold_margins(X, y_arr)

    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    dir_aug: List[int] = []
    rows: List[Dict[str, object]] = []
    safe_ratios: List[float] = []
    label_by_ratio = {ratios[0]: "small", ratios[len(ratios) // 2]: "mid", ratios[-1]: "edge"}

    for idx in range(X.shape[0]):
        rs = np.random.RandomState(int(seed + idx * 1009))
        dir_id = int(rs.choice(actual_k))
        sign = float(rs.choice([-1.0, 1.0]))
        u = W[dir_id]
        u_norm = float(np.linalg.norm(u))
        g0 = float(gammas[dir_id])
        if eta_safe is None:
            safe_upper = g0
        else:
            safe_upper = float(eta_safe) * float(margins[idx]) / (u_norm + 1e-12)
            safe_upper = min(g0, safe_upper)
        for ratio in ratios:
            gamma_used = float(ratio) * float(safe_upper)
            if eta_safe is not None and gamma_used * u_norm > float(eta_safe) * float(margins[idx]) + 1e-10:
                gamma_used = float(eta_safe) * float(margins[idx]) / (u_norm + 1e-12)
            safe_ratio = gamma_used / (g0 + 1e-12)
            z_new = X[idx] + gamma_used * sign * u
            z_aug.append(z_new.astype(np.float32))
            y_aug.append(int(y_arr[idx]))
            tid_aug.append(tid_arr[idx])
            dir_aug.append(dir_id)
            safe_ratios.append(float(safe_ratio))
            rows.append(
                {
                    "anchor_index": int(idx),
                    "tid": tid_arr[idx],
                    "tier_ratio": float(ratio),
                    "tier_label": label_by_ratio.get(ratio, f"tier_{ratio:g}"),
                    "direction_id": int(dir_id),
                    "sign": float(sign),
                    "safe_upper_bound": float(safe_upper),
                    "gamma_used": float(gamma_used),
                    "safe_radius_ratio": float(safe_ratio),
                }
            )

    meta = {
        "aug_total_count": int(len(y_aug)),
        "step_tier_count": int(len(ratios)),
        "safe_radius_ratio_mean": float(np.mean(safe_ratios)) if safe_ratios else 1.0,
        "safe_radius_ratio_min": float(np.min(safe_ratios)) if safe_ratios else 1.0,
        "manifold_margin_mean": float(np.mean(margins)) if margins.size else 0.0,
        "candidate_rows": rows,
    }
    return (
        np.vstack(z_aug).astype(np.float32) if z_aug else np.empty((0, X.shape[1]), dtype=np.float32),
        np.asarray(y_aug, dtype=np.int64),
        np.asarray(tid_aug, dtype=object),
        np.asarray(dir_aug, dtype=np.int64),
        meta,
    )


def realize_wavelet_candidates(
    *,
    z_aug: np.ndarray,
    y_aug: np.ndarray,
    tid_aug: np.ndarray,
    candidate_rows: Sequence[Dict[str, object]],
    tid_to_rec: Dict[str, WaveletTrialRecord],
    mean_log: np.ndarray,
    wavelet_name: str,
    wavelet_mode: str,
) -> Dict[str, object]:
    aug_trials: List[Dict[str, object]] = []
    bridge_metrics: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    for i in range(len(z_aug)):
        src = tid_to_rec[str(tid_aug[i])]
        sigma_aug = logvec_to_spd(z_aug[i], mean_log)
        cA_aug, bridge_meta = bridge_single(
            torch.from_numpy(src.cA),
            torch.from_numpy(src.sigma_a),
            torch.from_numpy(sigma_aug),
        )
        cA_np = cA_aug.numpy().astype(np.float64)
        cand_coeffs = [cA_np] + [np.asarray(c, dtype=np.float64).copy() for c in src.coeffs[1:]]
        assert_frozen_details(src.coeffs, cand_coeffs)
        x_aug = reconstruct_signal(
            cand_coeffs,
            original_length=src.x_raw.shape[-1],
            wavelet_name=wavelet_name,
            wavelet_mode=wavelet_mode,
        )
        aug_trials.append({"x": x_aug, "y": int(y_aug[i]), "tid": str(tid_aug[i])})
        bridge_metrics.append(bridge_meta)
        row = dict(candidate_rows[i]) if i < len(candidate_rows) else {}
        row.update(
            {
                "y": int(y_aug[i]),
                "cA_transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "cA_transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
            }
        )
        audit_rows.append(row)
    if bridge_metrics:
        import pandas as pd

        avg_bridge = {f"cA_{k}": float(v) for k, v in pd.DataFrame(bridge_metrics).mean(numeric_only=True).to_dict().items()}
    else:
        avg_bridge = {}
    return {
        "aug_trials": aug_trials,
        "X_aug_raw": np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None,
        "y_aug_np": np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None,
        "avg_bridge": avg_bridge,
        "audit_rows": audit_rows,
    }
