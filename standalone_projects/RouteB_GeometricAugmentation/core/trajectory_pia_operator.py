from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from PIA.telm2 import TELM2Config, TELM2Transformer


@dataclass(frozen=True)
class TrajectoryPIAOperatorConfig:
    r_dimension: int = 1
    n_iters: int = 3
    C_repr: float = 1.0
    activation: str = "sine"
    bias_lr: float = 0.25
    orthogonalize: bool = True
    enable_repr_learning: bool = True
    bias_update_mode: str = "residual"
    seed: int | None = None


@dataclass
class TrajectoryPIAOperatorArtifacts:
    W: np.ndarray
    b: np.ndarray
    recon_err: List[float]
    mu: np.ndarray
    direction: np.ndarray
    pooled_window_count: int
    z_dim: int
    meta: Dict[str, object] = field(default_factory=dict)


def _normalize_direction(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("direction vector cannot be empty")
    nrm = float(np.linalg.norm(arr))
    if not np.isfinite(nrm) or nrm <= 1e-12:
        out = np.zeros_like(arr, dtype=np.float64)
        out[0] = 1.0
        return out
    return np.asarray(arr / nrm, dtype=np.float64)


def _flatten_train_windows(z_seq_list: Sequence[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(seq, dtype=np.float64) for seq in z_seq_list if np.asarray(seq).size > 0]
    if not rows:
        raise ValueError("z_seq_list cannot be empty for trajectory operator fit")
    x = np.concatenate(rows, axis=0).astype(np.float64)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError("pooled train windows must be a non-empty 2D array")
    return x


def _smooth_delta(delta: np.ndarray, smooth_lambda: float) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("delta must be [K, D]")
    lam = float(smooth_lambda)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("smooth_lambda must be in [0, 1]")
    if arr.shape[0] <= 1 or lam <= 0.0:
        return np.asarray(arr, dtype=np.float64)

    padded = np.pad(arr, ((1, 1), (0, 0)), mode="edge")
    prev_delta = padded[:-2]
    cur_delta = padded[1:-1]
    next_delta = padded[2:]
    smoothed = (1.0 - lam) * cur_delta + 0.5 * lam * (prev_delta + next_delta)
    return np.asarray(smoothed, dtype=np.float64)


def _seq_step_change_mean(z_seq: np.ndarray) -> float:
    arr = np.asarray(z_seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    delta = np.diff(arr, axis=0)
    return float(np.mean(np.linalg.norm(delta, axis=1)))


class TrajectoryPIAOperator:
    def __init__(self, cfg: TrajectoryPIAOperatorConfig):
        if int(cfg.r_dimension) != 1:
            raise ValueError("T2a first version only supports r_dimension=1")
        self.cfg = cfg
        self._telm: TELM2Transformer | None = None
        self._arts: TrajectoryPIAOperatorArtifacts | None = None

    def get_artifacts(self) -> TrajectoryPIAOperatorArtifacts:
        if self._arts is None:
            raise RuntimeError("TrajectoryPIAOperator.fit() must be called first.")
        return self._arts

    def fit(self, train_z_seq_list: Sequence[np.ndarray]) -> "TrajectoryPIAOperator":
        x = _flatten_train_windows(train_z_seq_list)
        telm_cfg = TELM2Config(
            r_dimension=1,
            n_iters=int(self.cfg.n_iters),
            C_repr=float(self.cfg.C_repr),
            activation=str(self.cfg.activation),
            bias_lr=float(self.cfg.bias_lr),
            orthogonalize=bool(self.cfg.orthogonalize),
            enable_repr_learning=bool(self.cfg.enable_repr_learning),
            bias_update_mode=str(self.cfg.bias_update_mode),
            seed=None if self.cfg.seed is None else int(self.cfg.seed),
        )
        self._telm = TELM2Transformer(telm_cfg).fit(np.asarray(x, dtype=np.float64))
        raw = self._telm.get_artifacts()
        w = np.asarray(raw.W, dtype=np.float64)
        if w.ndim != 2:
            w = np.asarray(w, dtype=np.float64).reshape(1, -1)
        direction = _normalize_direction(w[0])
        mu = np.mean(x, axis=0).astype(np.float64)
        self._arts = TrajectoryPIAOperatorArtifacts(
            W=np.asarray(w, dtype=np.float64),
            b=np.asarray(raw.b, dtype=np.float64),
            recon_err=list(raw.recon_err),
            mu=np.asarray(mu, dtype=np.float64),
            direction=np.asarray(direction, dtype=np.float64),
            pooled_window_count=int(x.shape[0]),
            z_dim=int(x.shape[1]),
            meta={
                "shared_basis_mode": "global_single_axis",
                "basis_learning_scope": "pooled_train_windows_only",
            },
        )
        return self

    def transform(
        self,
        z_seq: np.ndarray,
        *,
        gamma_main: float,
        smooth_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        arts = self.get_artifacts()
        seq = np.asarray(z_seq, dtype=np.float64)
        if seq.ndim != 2 or seq.shape[1] != arts.z_dim:
            raise ValueError("z_seq must be [K, D] and match fitted z_dim")
        gamma = float(gamma_main)
        centered = seq - arts.mu[None, :]
        coeff = centered @ arts.direction[:, None]
        comp = coeff * arts.direction[None, :]
        delta_raw = gamma * comp
        delta_smooth = _smooth_delta(delta_raw, float(smooth_lambda))
        z_aug = seq + delta_smooth

        base_step = _seq_step_change_mean(seq)
        aug_step = _seq_step_change_mean(z_aug)
        continuity_ratio = float(aug_step / (base_step + 1e-12)) if seq.shape[0] > 1 else 1.0

        meta = {
            "gamma_main": float(gamma),
            "smooth_lambda": float(smooth_lambda),
            "base_step_change_mean": float(base_step),
            "aug_step_change_mean": float(aug_step),
            "continuity_distortion_ratio": float(continuity_ratio),
        }
        return np.asarray(z_aug, dtype=np.float32), np.asarray(delta_smooth, dtype=np.float32), meta

    def transform_many(
        self,
        z_seq_list: Sequence[np.ndarray],
        *,
        gamma_main: float,
        smooth_lambda: float,
    ) -> tuple[List[np.ndarray], List[np.ndarray], Dict[str, object]]:
        z_aug_list: List[np.ndarray] = []
        delta_list: List[np.ndarray] = []
        continuity_rows: List[float] = []
        for seq in z_seq_list:
            z_aug, delta_seq, meta = self.transform(
                seq,
                gamma_main=float(gamma_main),
                smooth_lambda=float(smooth_lambda),
            )
            z_aug_list.append(z_aug)
            delta_list.append(delta_seq)
            continuity_rows.append(float(meta["continuity_distortion_ratio"]))
        return z_aug_list, delta_list, {
            "gamma_main": float(gamma_main),
            "smooth_lambda": float(smooth_lambda),
            "mean_continuity_distortion_ratio": float(np.mean(continuity_rows)) if continuity_rows else 1.0,
            "n_aug_sequences": int(len(z_aug_list)),
        }

