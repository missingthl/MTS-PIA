from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    cDm: Optional[np.ndarray]
    sigma_dm: Optional[np.ndarray]
    z_dm: Optional[np.ndarray]
    secondary_detail_level: int
    secondary_coeff_index: int
    wavelet_level_eff: int
    wavelet_recon_error: float


def detail_level_to_coeff_index(level_eff: int, detail_level: int) -> int:
    """Map pywt detail level m to wavedec coefficient index for [cA_L, cD_L, ..., cD_1]."""
    level_eff = int(level_eff)
    detail_level = int(detail_level)
    if detail_level < 1 or detail_level > level_eff:
        raise ValueError(f"detail_level must be in [1, {level_eff}], got {detail_level}.")
    return int(level_eff - detail_level + 1)


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
    secondary_detail_level: int = 2,
    spd_eps: float = 1e-4,
) -> Tuple[List[WaveletTrialRecord], np.ndarray | None, Dict[str, object]]:
    if not trials:
        return [], None, {}

    raw_rows: List[Dict[str, object]] = []
    log_covs: List[np.ndarray] = []
    log_covs_dm: List[np.ndarray] = []
    cA_shape = None
    cDm_shape = None
    level_eff = None
    secondary_coeff_index = -1
    recon_errors: List[float] = []
    cDm_energy_values: List[float] = []
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
        if int(secondary_detail_level) <= int(level_eff):
            secondary_coeff_index = detail_level_to_coeff_index(int(level_eff), int(secondary_detail_level))
            cDm = coeffs[secondary_coeff_index]
            if cDm_shape is None:
                cDm_shape = tuple(cDm.shape)
            elif tuple(cDm.shape) != cDm_shape:
                raise ValueError(f"Inconsistent cD_{secondary_detail_level} shape: got {tuple(cDm.shape)}, expected {cDm_shape}.")
            sigma_dm, log_cov_dm, _ = covariance_and_logvec(cDm, mean_log=None, spd_eps=spd_eps)
            log_covs_dm.append(log_cov_dm)
            cDm_energy_values.append(float(np.mean(np.square(cDm))))
        else:
            cDm = None
            sigma_dm = None
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
                "cDm": cDm,
                "sigma_dm": sigma_dm,
                "recon_error": recon_error,
            }
        )

    mean_log = np.mean(log_covs, axis=0)
    mean_log_dm = np.mean(log_covs_dm, axis=0) if log_covs_dm else None
    records: List[WaveletTrialRecord] = []
    for row, log_cov in zip(raw_rows, log_covs):
        idx = np.triu_indices(mean_log.shape[0])
        z_a = (log_cov - mean_log)[idx].astype(np.float64)
        z_dm = None
        if mean_log_dm is not None and row["cDm"] is not None:
            _, log_cov_dm, _ = covariance_and_logvec(np.asarray(row["cDm"]), mean_log=None, spd_eps=spd_eps)
            idx_dm = np.triu_indices(mean_log_dm.shape[0])
            z_dm = (log_cov_dm - mean_log_dm)[idx_dm].astype(np.float64)
        records.append(
            WaveletTrialRecord(
                tid=str(row["tid"]),
                y=int(row["y"]),
                x_raw=np.asarray(row["x_raw"], dtype=np.float32),
                coeffs=[np.asarray(c, dtype=np.float64) for c in row["coeffs"]],
                cA=np.asarray(row["cA"], dtype=np.float64),
                sigma_a=np.asarray(row["sigma_a"], dtype=np.float64),
                z_a=z_a,
                cDm=None if row["cDm"] is None else np.asarray(row["cDm"], dtype=np.float64),
                sigma_dm=None if row["sigma_dm"] is None else np.asarray(row["sigma_dm"], dtype=np.float64),
                z_dm=z_dm,
                secondary_detail_level=int(secondary_detail_level),
                secondary_coeff_index=int(secondary_coeff_index),
                wavelet_level_eff=int(level_eff),
                wavelet_recon_error=float(row["recon_error"]),
            )
        )

    meta = {
        "wavelet_level_eff": int(level_eff),
        "cA_shape": tuple(cA_shape or ()),
        "cA_length": int(cA_shape[-1]) if cA_shape else 0,
        "cDm_shape": tuple(cDm_shape or ()),
        "cDm_length": int(cDm_shape[-1]) if cDm_shape else 0,
        "cDm_energy_mean": float(np.mean(cDm_energy_values)) if cDm_energy_values else 0.0,
        "secondary_detail_level": int(secondary_detail_level),
        "secondary_coeff_index": int(secondary_coeff_index),
        "mean_log_dm": mean_log_dm,
        "cD_frozen_count": int(level_eff),
        "wavelet_recon_error_mean": float(np.mean(recon_errors)),
    }
    return records, mean_log, meta


def assert_frozen_details(
    original_coeffs: Sequence[np.ndarray],
    candidate_coeffs: Sequence[np.ndarray],
    *,
    mutable_detail_indices: Optional[Sequence[int]] = None,
) -> None:
    if len(original_coeffs) != len(candidate_coeffs):
        raise AssertionError("DWT coefficient list length changed during wavelet_mba realization.")
    mutable = set(int(i) for i in (mutable_detail_indices or []))
    for level_idx, (orig, cand) in enumerate(zip(original_coeffs[1:], candidate_coeffs[1:]), start=1):
        if level_idx in mutable:
            continue
        if not np.array_equal(np.asarray(orig), np.asarray(cand)):
            raise AssertionError(f"Frozen cD mismatch at coefficient index {level_idx}.")


def compute_identity_checks(
    records: Sequence[WaveletTrialRecord],
    *,
    wavelet_name: str,
    wavelet_mode: str,
    object_mode: str = "ca_only",
    max_items: int = 32,
) -> Dict[str, float]:
    if not records:
        return {
            "cA_identity_bridge_error_mean": 0.0,
            "cDm_identity_bridge_error_mean": 0.0,
            "idwt_identity_recon_error_mean": 0.0,
        }
    cA_errs: List[float] = []
    cDm_errs: List[float] = []
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
        mutable_detail_indices: List[int] = []
        if str(object_mode) == "dual_a_dm":
            if rec.cDm is None or rec.sigma_dm is None:
                raise ValueError("dual_a_dm identity check requires cD_m object fields.")
            cDm_same, _ = bridge_single(
                torch.from_numpy(rec.cDm),
                torch.from_numpy(rec.sigma_dm),
                torch.from_numpy(rec.sigma_dm),
            )
            cDm_np = cDm_same.numpy().astype(np.float64)
            cDm_errs.append(float(np.mean(np.abs(cDm_np - rec.cDm))))
            cand_coeffs[rec.secondary_coeff_index] = cDm_np
            mutable_detail_indices.append(int(rec.secondary_coeff_index))
        assert_frozen_details(rec.coeffs, cand_coeffs, mutable_detail_indices=mutable_detail_indices)
        x_recon = reconstruct_signal(
            cand_coeffs,
            original_length=rec.x_raw.shape[-1],
            wavelet_name=wavelet_name,
            wavelet_mode=wavelet_mode,
        )
        idwt_errs.append(float(np.mean(np.abs(x_recon.astype(np.float64) - rec.x_raw.astype(np.float64)))))
    return {
        "cA_identity_bridge_error_mean": float(np.mean(cA_errs)),
        "cDm_identity_bridge_error_mean": float(np.mean(cDm_errs)) if cDm_errs else 0.0,
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
    object_mode: str = "ca_only",
    z_dm_aug: Optional[np.ndarray] = None,
    mean_log_dm: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    aug_trials: List[Dict[str, object]] = []
    bridge_metrics_a: List[Dict[str, object]] = []
    bridge_metrics_dm: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    dual_mode = str(object_mode) == "dual_a_dm"
    if dual_mode and (z_dm_aug is None or mean_log_dm is None):
        raise ValueError("dual_a_dm realization requires z_dm_aug and mean_log_dm.")
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
        mutable_detail_indices: List[int] = []
        if dual_mode:
            if src.cDm is None or src.sigma_dm is None:
                raise ValueError(f"Missing cD_m object for tid={src.tid}.")
            sigma_dm_aug = logvec_to_spd(np.asarray(z_dm_aug)[i], mean_log_dm)
            cDm_aug, bridge_meta_dm = bridge_single(
                torch.from_numpy(src.cDm),
                torch.from_numpy(src.sigma_dm),
                torch.from_numpy(sigma_dm_aug),
            )
            cDm_np = cDm_aug.numpy().astype(np.float64)
            cand_coeffs[src.secondary_coeff_index] = cDm_np
            mutable_detail_indices.append(int(src.secondary_coeff_index))
            bridge_metrics_dm.append(bridge_meta_dm)
        assert_frozen_details(src.coeffs, cand_coeffs, mutable_detail_indices=mutable_detail_indices)
        x_aug = reconstruct_signal(
            cand_coeffs,
            original_length=src.x_raw.shape[-1],
            wavelet_name=wavelet_name,
            wavelet_mode=wavelet_mode,
        )
        aug_trials.append({"x": x_aug, "y": int(y_aug[i]), "tid": str(tid_aug[i])})
        bridge_metrics_a.append(bridge_meta)
        row = dict(candidate_rows[i]) if i < len(candidate_rows) else {}
        row.update(
            {
                "y": int(y_aug[i]),
                "cA_transport_error_logeuc": float(bridge_meta.get("transport_error_logeuc", 0.0)),
                "cA_transport_error_fro": float(bridge_meta.get("transport_error_fro", 0.0)),
            }
        )
        if dual_mode:
            bridge_meta_dm = bridge_metrics_dm[-1]
            row.update(
                {
                    "cDm_transport_error_logeuc": float(bridge_meta_dm.get("transport_error_logeuc", 0.0)),
                    "cDm_transport_error_fro": float(bridge_meta_dm.get("transport_error_fro", 0.0)),
                    "dual_transport_error_logeuc": 0.5
                    * (
                        float(bridge_meta.get("transport_error_logeuc", 0.0))
                        + float(bridge_meta_dm.get("transport_error_logeuc", 0.0))
                    ),
                }
            )
        audit_rows.append(row)
    if bridge_metrics_a:
        import pandas as pd

        avg_bridge = {
            f"cA_{k}": float(v)
            for k, v in pd.DataFrame(bridge_metrics_a).mean(numeric_only=True).to_dict().items()
        }
        if bridge_metrics_dm:
            dm_avg = {
                f"cDm_{k}": float(v)
                for k, v in pd.DataFrame(bridge_metrics_dm).mean(numeric_only=True).to_dict().items()
            }
            avg_bridge.update(dm_avg)
            avg_bridge["dual_transport_error_logeuc"] = 0.5 * (
                float(avg_bridge.get("cA_transport_error_logeuc", 0.0))
                + float(avg_bridge.get("cDm_transport_error_logeuc", 0.0))
            )
    else:
        avg_bridge = {}
    return {
        "aug_trials": aug_trials,
        "X_aug_raw": np.stack([trial["x"] for trial in aug_trials]) if aug_trials else None,
        "y_aug_np": np.asarray([trial["y"] for trial in aug_trials], dtype=np.int64) if aug_trials else None,
        "avg_bridge": avg_bridge,
        "audit_rows": audit_rows,
    }
