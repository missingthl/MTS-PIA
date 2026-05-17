from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PIA.telm2 import TELM2Config, TELM2Transformer


def _act(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sine":
        return np.sin(x)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"unknown activation: {kind}")


def _orthonormalize_rows(W: np.ndarray) -> np.ndarray:
    X = np.asarray(W, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("W must be 2D for row orthonormalization")
    if X.shape[0] == 0:
        return X.astype(np.float64)
    q, _ = np.linalg.qr(X.T)
    rows = q.T[: X.shape[0]]
    out = np.asarray(rows, dtype=np.float64)
    for i in range(out.shape[0]):
        nrm = float(np.linalg.norm(out[i]))
        if not np.isfinite(nrm) or nrm <= 1e-12:
            out[i] = 0.0
            out[i, 0] = 1.0
        else:
            out[i] /= nrm
    return out


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
    }


def _infer_sym_dim_from_utri_size(size: int) -> int:
    if size <= 0:
        raise ValueError("upper-triangle vector size must be positive")
    disc = 1 + 8 * int(size)
    root = int(np.sqrt(float(disc)))
    if root * root != disc:
        raise ValueError(f"cannot infer symmetric matrix dim from upper-triangle size {size}")
    dim = (-1 + root) // 2
    if dim * (dim + 1) // 2 != int(size):
        raise ValueError(f"invalid upper-triangle vector size {size}")
    return int(dim)


def _unvec_utri_sym(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).ravel()
    idx = np.triu_indices(int(dim))
    if v.size != idx[0].size:
        raise ValueError(f"upper-triangle vector size mismatch: got {v.size}, expected {idx[0].size}")
    out = np.zeros((int(dim), int(dim)), dtype=np.float64)
    out[idx] = v
    out[(idx[1], idx[0])] = v
    return out


def _vec_utri_sym(mat: np.ndarray) -> np.ndarray:
    mm = np.asarray(mat, dtype=np.float64)
    if mm.ndim != 2 or mm.shape[0] != mm.shape[1]:
        raise ValueError("mat must be a square symmetric matrix")
    idx = np.triu_indices(mm.shape[0])
    return np.asarray(mm[idx], dtype=np.float64)


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    mm = np.asarray(mat, dtype=np.float64)
    return 0.5 * (mm + mm.T)


def _orthonormalize_sym_mats(mats: np.ndarray) -> np.ndarray:
    arr = np.asarray(mats, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("mats must be [K, C, C]")
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        cur = _symmetrize(arr[i])
        for j in range(i):
            denom = float(np.sum(out[j] * out[j])) + 1e-12
            proj = float(np.sum(cur * out[j])) / denom
            cur = cur - proj * out[j]
        cur = _symmetrize(cur)
        nrm = float(np.sqrt(np.sum(cur * cur)))
        if not np.isfinite(nrm) or nrm <= 1e-12:
            cur = np.zeros_like(cur, dtype=np.float64)
            cur[0, 0] = 1.0
        else:
            cur = cur / nrm
        out[i] = cur
    return out


@dataclass(frozen=True)
class PIACoreConfig:
    r_dimension: int = 3
    n_iters: int = 3
    C_repr: float = 1.0
    activation: str = "sine"
    bias_lr: float = 0.25
    orthogonalize: bool = True
    enable_repr_learning: bool = True
    bias_update_mode: str = "residual"
    seed: Optional[int] = None


@dataclass(frozen=True)
class PIACoreArtifacts:
    W: np.ndarray
    b: np.ndarray
    recon_err: List[float]
    mu: np.ndarray
    directions: np.ndarray


@dataclass(frozen=True)
class PIAOperatorResult:
    X_aug: np.ndarray
    meta: Dict[str, object]


class PIACore:
    """Template-driven closed-form PIA core.

    This module is the method core of PIA/TELM2:
    - fit only on the training representation
    - learn template matrix W and bias b via closed-form updates
    - expose template response P = act(XW^T + b)
    - expose an explicit affine operator that stretches samples along template axes

    It is intentionally not a generic policy/controller.
    """

    def __init__(self, cfg: PIACoreConfig):
        self.cfg = cfg
        self._telm: Optional[TELM2Transformer] = None
        self._arts: Optional[PIACoreArtifacts] = None

    def fit(self, X_tr: np.ndarray) -> "PIACore":
        X = np.asarray(X_tr, dtype=np.float64)
        telm_cfg = TELM2Config(
            r_dimension=int(self.cfg.r_dimension),
            n_iters=int(self.cfg.n_iters),
            C_repr=float(self.cfg.C_repr),
            activation=str(self.cfg.activation),
            bias_lr=float(self.cfg.bias_lr),
            orthogonalize=bool(self.cfg.orthogonalize),
            enable_repr_learning=bool(self.cfg.enable_repr_learning),
            bias_update_mode=str(self.cfg.bias_update_mode),
            seed=None if self.cfg.seed is None else int(self.cfg.seed),
        )
        self._telm = TELM2Transformer(telm_cfg).fit(X)
        raw = self._telm.get_artifacts()
        W = np.asarray(raw.W, dtype=np.float64)
        if W.ndim != 2:
            W = np.asarray(W, dtype=np.float64).reshape(1, -1)
        dirs = _orthonormalize_rows(W)
        self._arts = PIACoreArtifacts(
            W=np.asarray(W, dtype=np.float64),
            b=np.asarray(raw.b, dtype=np.float64),
            recon_err=list(raw.recon_err),
            mu=np.mean(X, axis=0, keepdims=True).astype(np.float64),
            directions=np.asarray(dirs, dtype=np.float64),
        )
        return self

    def get_artifacts(self) -> PIACoreArtifacts:
        if self._arts is None:
            raise RuntimeError("PIACore.fit() must be called first.")
        return self._arts

    def transform(self, X: np.ndarray) -> np.ndarray:
        arts = self.get_artifacts()
        xx = np.asarray(X, dtype=np.float64)
        return _act(xx @ arts.W.T + arts.b[None, :], str(self.cfg.activation)).astype(np.float64)

    def fit_transform(self, X_tr: np.ndarray) -> np.ndarray:
        return self.fit(X_tr).transform(X_tr)

    def rank_axes_by_energy(self, X: np.ndarray) -> Tuple[List[int], Dict[str, object]]:
        arts = self.get_artifacts()
        xx = np.asarray(X, dtype=np.float64)
        centered = xx - arts.mu
        dirs = np.asarray(arts.directions, dtype=np.float64)
        if dirs.size == 0:
            return [], {"axis_energy_stats": _stats([]), "ranked_axis_ids": []}
        coeff = centered @ dirs.T
        axis_energy = np.sum(coeff * coeff, axis=0).astype(np.float64)
        order = np.argsort(axis_energy)[::-1]
        ranked = [int(v) for v in order.tolist()]
        return ranked, {
            "axis_energy_stats": _stats(axis_energy.tolist()),
            "ranked_axis_ids": ranked,
            "axis_energy_vector": [float(v) for v in axis_energy.tolist()],
        }

    def build_two_axis_gamma_vector(
        self,
        *,
        axis_ids: Sequence[int],
        gamma_main: float,
        second_axis_scale: float = 1.0,
    ) -> Tuple[List[float], Dict[str, object]]:
        axis_ids_arr = np.asarray(list(axis_ids), dtype=np.int64).ravel()
        if axis_ids_arr.size == 0:
            raise ValueError("axis_ids cannot be empty")
        gamma_main_f = float(gamma_main)
        second_scale_f = float(second_axis_scale)
        if gamma_main_f < 0.0:
            raise ValueError("gamma_main must be >= 0")
        if second_scale_f < 0.0:
            raise ValueError("second_axis_scale must be >= 0")

        gamma_arr = np.full((axis_ids_arr.size,), gamma_main_f, dtype=np.float64)
        if axis_ids_arr.size >= 2:
            gamma_arr[1] = gamma_main_f * second_scale_f
        return [float(v) for v in gamma_arr.tolist()], {
            "axis_strength_mode": "two_axis_refine_v1",
            "gamma_main": gamma_main_f,
            "second_axis_scale": second_scale_f,
            "effective_second_axis_gamma": float(gamma_arr[1]) if gamma_arr.size >= 2 else gamma_main_f,
        }

    def apply_affine(
        self,
        X: np.ndarray,
        *,
        gamma_vector: Sequence[float],
        axis_ids: Optional[Sequence[int]] = None,
        pullback_alpha: float = 1.0,
    ) -> PIAOperatorResult:
        arts = self.get_artifacts()
        xx = np.asarray(X, dtype=np.float64)
        centered = xx - arts.mu
        dirs = np.asarray(arts.directions, dtype=np.float64)

        if axis_ids is None:
            axis_ids_arr = np.arange(dirs.shape[0], dtype=np.int64)
        else:
            axis_ids_arr = np.asarray(list(axis_ids), dtype=np.int64).ravel()
        if axis_ids_arr.size == 0:
            axis_ids_arr = np.arange(dirs.shape[0], dtype=np.int64)

        gamma_arr = np.asarray(list(gamma_vector), dtype=np.float64).ravel()
        if gamma_arr.size == 0:
            raise ValueError("gamma_vector cannot be empty")
        if gamma_arr.size != axis_ids_arr.size:
            raise ValueError("gamma_vector size must match axis_ids size")

        selected_dirs = dirs[axis_ids_arr]
        coeff = centered @ selected_dirs.T
        comps = coeff[:, :, None] * selected_dirs[None, :, :]
        residual = centered - np.sum(comps, axis=1)
        scaled = np.sum((1.0 + gamma_arr[None, :, None]) * comps, axis=1)
        x_affine = arts.mu + scaled + residual
        alpha = float(pullback_alpha)
        if alpha < 0.0:
            raise ValueError("pullback_alpha must be >= 0")
        x_aug = xx + alpha * (x_affine - xx)

        axis_energy = np.sum(coeff * coeff, axis=0) if coeff.size else np.zeros((0,), dtype=np.float64)
        residual_energy = np.sum(residual * residual, axis=1)
        total_centered_energy = np.sum(centered * centered, axis=1)
        aligned_ratio = (
            float(np.mean(np.sum(coeff * coeff, axis=1) / (total_centered_energy + 1e-12)))
            if coeff.size
            else 0.0
        )
        meta = {
            "operator_type": "pia_explicit_affine",
            "operator_semantics": "sample_conditional_template_axis_stretch_with_residual_preservation",
            "selected_axis_ids": [int(v) for v in axis_ids_arr.tolist()],
            "gamma_vector": [float(v) for v in gamma_arr.tolist()],
            "pullback_alpha": alpha,
            "center_mode": "train_mean",
            "residual_preservation_strength": 1.0,
            "axis_energy_stats": _stats(axis_energy.tolist()),
            "residual_energy_mean": float(np.mean(residual_energy)) if residual_energy.size else 0.0,
            "aligned_energy_ratio_mean": aligned_ratio,
        }
        return PIAOperatorResult(X_aug=np.asarray(x_aug, dtype=np.float32), meta=meta)

    def apply_logeuclidean_affine(
        self,
        X: np.ndarray,
        *,
        gamma_vector: Sequence[float],
        axis_ids: Optional[Sequence[int]] = None,
        pullback_alpha: float = 1.0,
    ) -> PIAOperatorResult:
        arts = self.get_artifacts()
        xx = np.asarray(X, dtype=np.float64)
        if xx.ndim != 2:
            raise ValueError("X must be [N, D]")
        dim = _infer_sym_dim_from_utri_size(int(xx.shape[1]))
        centered_logs = np.stack([_unvec_utri_sym(v, dim) for v in xx], axis=0)

        dirs = np.asarray(arts.directions, dtype=np.float64)
        if axis_ids is None:
            axis_ids_arr = np.arange(dirs.shape[0], dtype=np.int64)
        else:
            axis_ids_arr = np.asarray(list(axis_ids), dtype=np.int64).ravel()
        if axis_ids_arr.size == 0:
            axis_ids_arr = np.arange(dirs.shape[0], dtype=np.int64)

        gamma_arr = np.asarray(list(gamma_vector), dtype=np.float64).ravel()
        if gamma_arr.size == 0:
            raise ValueError("gamma_vector cannot be empty")
        if gamma_arr.size != axis_ids_arr.size:
            raise ValueError("gamma_vector size must match axis_ids size")

        selected_vec_dirs = dirs[axis_ids_arr]
        selected_sym_dirs = np.stack([_unvec_utri_sym(v, dim) for v in selected_vec_dirs], axis=0)
        selected_sym_dirs = _orthonormalize_sym_mats(selected_sym_dirs)

        coeff = np.einsum("nij,kij->nk", centered_logs, selected_sym_dirs)
        comps = coeff[:, :, None, None] * selected_sym_dirs[None, :, :, :]
        residual = centered_logs - np.sum(comps, axis=1)
        scaled = np.sum((1.0 + gamma_arr[None, :, None, None]) * comps, axis=1)
        mats_affine = scaled + residual

        alpha = float(pullback_alpha)
        if alpha < 0.0:
            raise ValueError("pullback_alpha must be >= 0")
        mats_aug = centered_logs + alpha * (mats_affine - centered_logs)
        mats_aug = np.asarray([_symmetrize(m) for m in mats_aug], dtype=np.float64)
        x_aug = np.stack([_vec_utri_sym(m) for m in mats_aug], axis=0)

        axis_energy = np.sum(coeff * coeff, axis=0) if coeff.size else np.zeros((0,), dtype=np.float64)
        residual_energy = np.sum(residual * residual, axis=(1, 2))
        total_centered_energy = np.sum(centered_logs * centered_logs, axis=(1, 2))
        aligned_ratio = (
            float(np.mean(np.sum(coeff * coeff, axis=1) / (total_centered_energy + 1e-12)))
            if coeff.size
            else 0.0
        )
        meta = {
            "operator_type": "pia_logeuclidean_affine",
            "operator_semantics": "sample_conditional_logeuclidean_axis_stretch_with_residual_preservation",
            "geometry_mode": "centered_log_covariance_matrix_domain",
            "selected_axis_ids": [int(v) for v in axis_ids_arr.tolist()],
            "gamma_vector": [float(v) for v in gamma_arr.tolist()],
            "pullback_alpha": alpha,
            "center_mode": "mean_log_train_zero_centered",
            "residual_preservation_strength": 1.0,
            "axis_energy_stats": _stats(axis_energy.tolist()),
            "residual_energy_mean": float(np.mean(residual_energy)) if residual_energy.size else 0.0,
            "aligned_energy_ratio_mean": aligned_ratio,
            "matrix_axis_dim": int(dim),
        }
        return PIAOperatorResult(X_aug=np.asarray(x_aug, dtype=np.float32), meta=meta)
