from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from core.trajectory_pia_operator import (
    TrajectoryPIAOperator,
    TrajectoryPIAOperatorArtifacts,
    TrajectoryPIAOperatorConfig,
)


@dataclass(frozen=True)
class TrajectoryPIAT2B0Config:
    gamma_base: float = 0.05
    smooth_lambda: float = 0.50
    low_multiplier: float = 0.5
    mid_multiplier: float = 1.0
    high_multiplier: float = 1.5
    seed: int = 0


@dataclass
class TrajectoryPIAT2B0Artifacts:
    base_artifacts: TrajectoryPIAOperatorArtifacts
    gamma_base: float
    smooth_lambda: float
    multipliers: Dict[str, float]
    meta: Dict[str, object] = field(default_factory=dict)


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
    return np.asarray((1.0 - lam) * cur_delta + 0.5 * lam * (prev_delta + next_delta), dtype=np.float64)


def _seq_step_change_mean(z_seq: np.ndarray) -> float:
    arr = np.asarray(z_seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    delta = np.diff(arr, axis=0)
    return float(np.mean(np.linalg.norm(delta, axis=1)))


def _compute_saliency(seq: np.ndarray) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("z_seq must be [K, D]")
    if arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    padded = np.pad(arr, ((1, 1), (0, 0)), mode="edge")
    prev_arr = padded[:-2]
    cur_arr = padded[1:-1]
    next_arr = padded[2:]
    step = np.linalg.norm(cur_arr - prev_arr, axis=1)
    curvature = np.linalg.norm(next_arr - 2.0 * cur_arr + prev_arr, axis=1)
    return np.asarray(step + curvature, dtype=np.float64)


def _quantile_levels(saliency: np.ndarray) -> np.ndarray:
    arr = np.asarray(saliency, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if arr.size <= 2 or np.allclose(arr, arr[0]):
        return np.ones((arr.size,), dtype=np.int64)
    q1, q2 = np.percentile(arr, [33.333333, 66.666667])
    levels = np.ones((arr.size,), dtype=np.int64)
    levels[arr <= q1] = 0
    levels[arr > q2] = 2
    return levels


def _levels_to_gamma(levels: np.ndarray, *, gamma_base: float, multipliers: np.ndarray) -> np.ndarray:
    arr = np.asarray(levels, dtype=np.int64).ravel()
    mult = np.asarray(multipliers, dtype=np.float64)
    if np.any(arr < 0) or np.any(arr >= mult.size):
        raise ValueError("levels out of multiplier range")
    return np.asarray(float(gamma_base) * mult[arr], dtype=np.float64)


def _level_ratios(levels: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(levels, dtype=np.int64).ravel()
    n = float(arr.size)
    if n <= 0:
        return {"saliency_low_ratio": 0.0, "saliency_mid_ratio": 0.0, "saliency_high_ratio": 0.0}
    return {
        "saliency_low_ratio": float(np.mean(arr == 0)),
        "saliency_mid_ratio": float(np.mean(arr == 1)),
        "saliency_high_ratio": float(np.mean(arr == 2)),
    }


def _stable_seed(text: str, base_seed: int) -> int:
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(h, 16)


class TrajectoryPIAT2B0Operator:
    def __init__(
        self,
        base_cfg: TrajectoryPIAOperatorConfig,
        t2b_cfg: TrajectoryPIAT2B0Config,
    ) -> None:
        self.base_cfg = base_cfg
        self.t2b_cfg = t2b_cfg
        self._base_operator: TrajectoryPIAOperator | None = None
        self._arts: TrajectoryPIAT2B0Artifacts | None = None

    def fit(
        self,
        train_z_seq_list: Sequence[np.ndarray],
        *,
        prefit_base_operator: TrajectoryPIAOperator | None = None,
    ) -> "TrajectoryPIAT2B0Operator":
        self._base_operator = prefit_base_operator if prefit_base_operator is not None else TrajectoryPIAOperator(self.base_cfg).fit(train_z_seq_list)
        base_artifacts = self._base_operator.get_artifacts()
        self._arts = TrajectoryPIAT2B0Artifacts(
            base_artifacts=base_artifacts,
            gamma_base=float(self.t2b_cfg.gamma_base),
            smooth_lambda=float(self.t2b_cfg.smooth_lambda),
            multipliers={
                "low": float(self.t2b_cfg.low_multiplier),
                "mid": float(self.t2b_cfg.mid_multiplier),
                "high": float(self.t2b_cfg.high_multiplier),
            },
            meta={
                "operator_family": "t2b0_fixed_rule_local_saliency_probe",
                "basis_mode": "shared_global_single_axis_from_t2a",
            },
        )
        return self

    def get_artifacts(self) -> TrajectoryPIAT2B0Artifacts:
        if self._arts is None or self._base_operator is None:
            raise RuntimeError("TrajectoryPIAT2B0Operator.fit() must be called first.")
        return self._arts

    def _randomize_levels(self, levels: np.ndarray, *, trial_id: str) -> np.ndarray:
        arr = np.asarray(levels, dtype=np.int64).copy()
        if arr.size <= 1:
            return arr
        rng = np.random.default_rng(_stable_seed(str(trial_id), int(self.t2b_cfg.seed)))
        return np.asarray(rng.permutation(arr), dtype=np.int64)

    def transform(
        self,
        z_seq: np.ndarray,
        *,
        mode: str,
        trial_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        arts = self.get_artifacts()
        seq = np.asarray(z_seq, dtype=np.float64)
        if seq.ndim != 2 or seq.shape[1] != int(arts.base_artifacts.z_dim):
            raise ValueError("z_seq must be [K, D] and match fitted z_dim")
        if str(mode) not in {"saliency", "randomized"}:
            raise ValueError("mode must be 'saliency' or 'randomized'")

        centered = seq - arts.base_artifacts.mu[None, :]
        coeff = centered @ arts.base_artifacts.direction[:, None]
        comp = coeff * arts.base_artifacts.direction[None, :]

        saliency = _compute_saliency(seq)
        levels = _quantile_levels(saliency)
        applied_levels = levels if str(mode) == "saliency" else self._randomize_levels(levels, trial_id=str(trial_id))
        gamma_seq = _levels_to_gamma(
            applied_levels,
            gamma_base=float(arts.gamma_base),
            multipliers=np.asarray(
                [
                    float(self.t2b_cfg.low_multiplier),
                    float(self.t2b_cfg.mid_multiplier),
                    float(self.t2b_cfg.high_multiplier),
                ],
                dtype=np.float64,
            ),
        )

        delta_raw = gamma_seq[:, None] * comp
        delta_smooth = _smooth_delta(delta_raw, float(arts.smooth_lambda))
        z_aug = seq + delta_smooth

        base_step = _seq_step_change_mean(seq)
        aug_step = _seq_step_change_mean(z_aug)
        continuity_ratio = float(aug_step / (base_step + 1e-12)) if seq.shape[0] > 1 else 1.0
        meta = {
            "gamma_effective_mean": float(np.mean(gamma_seq)) if gamma_seq.size else 0.0,
            "continuity_distortion_ratio": float(continuity_ratio),
            "saliency_step_change_mean": float(np.mean(np.linalg.norm(np.diff(seq, axis=0), axis=1))) if seq.shape[0] > 1 else 0.0,
            **_level_ratios(applied_levels),
        }
        return (
            np.asarray(z_aug, dtype=np.float32),
            np.asarray(delta_smooth, dtype=np.float32),
            np.asarray(gamma_seq, dtype=np.float32),
            meta,
        )

    def transform_many(
        self,
        z_seq_list: Sequence[np.ndarray],
        *,
        mode: str,
        trial_ids: Sequence[str],
    ) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, object]]:
        if len(list(z_seq_list)) != len(list(trial_ids)):
            raise ValueError("z_seq_list and trial_ids must have same length")
        z_aug_list: List[np.ndarray] = []
        delta_list: List[np.ndarray] = []
        gamma_list: List[np.ndarray] = []
        continuity_rows: List[float] = []
        low_rows: List[float] = []
        mid_rows: List[float] = []
        high_rows: List[float] = []
        gamma_mean_rows: List[float] = []
        for seq, tid in zip(z_seq_list, trial_ids):
            z_aug, delta_seq, gamma_seq, meta = self.transform(seq, mode=str(mode), trial_id=str(tid))
            z_aug_list.append(z_aug)
            delta_list.append(delta_seq)
            gamma_list.append(gamma_seq)
            continuity_rows.append(float(meta["continuity_distortion_ratio"]))
            low_rows.append(float(meta["saliency_low_ratio"]))
            mid_rows.append(float(meta["saliency_mid_ratio"]))
            high_rows.append(float(meta["saliency_high_ratio"]))
            gamma_mean_rows.append(float(meta["gamma_effective_mean"]))
        return z_aug_list, delta_list, gamma_list, {
            "mode": str(mode),
            "gamma_base": float(self.t2b_cfg.gamma_base),
            "smooth_lambda": float(self.t2b_cfg.smooth_lambda),
            "mean_continuity_distortion_ratio": float(np.mean(continuity_rows)) if continuity_rows else 1.0,
            "saliency_low_ratio": float(np.mean(low_rows)) if low_rows else 0.0,
            "saliency_mid_ratio": float(np.mean(mid_rows)) if mid_rows else 0.0,
            "saliency_high_ratio": float(np.mean(high_rows)) if high_rows else 0.0,
            "gamma_effective_mean": float(np.mean(gamma_mean_rows)) if gamma_mean_rows else 0.0,
            "n_aug_sequences": int(len(z_aug_list)),
        }
