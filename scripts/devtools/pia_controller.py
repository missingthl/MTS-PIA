from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _softmax(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    if xx.size == 0:
        return np.asarray([], dtype=np.float64)
    m = float(np.max(xx))
    ex = np.exp(xx - m)
    z = float(np.sum(ex))
    if not np.isfinite(z) or z <= 1e-12:
        return np.ones_like(xx, dtype=np.float64) / float(max(1, xx.size))
    return ex / z


@dataclass
class PiaControllerConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    tau: float = 1.0
    kappa: float = 0.25
    lambda_ema: float = 0.2
    freeze_M: int = 3
    gamma_scale_min: float = 0.5
    gamma_scale_max: float = 1.5
    enable_weight_update: bool = True
    enable_gamma_update: bool = False
    enable_freeze: bool = False


class PiaControllerLite:
    def __init__(
        self,
        *,
        classes: Iterable[int],
        n_dirs: int,
        cfg: PiaControllerConfig,
    ) -> None:
        self.cfg = cfg
        self.classes = [int(c) for c in classes]
        self.class_to_idx = {int(c): i for i, c in enumerate(self.classes)}
        self.n_dirs = int(n_dirs)
        n_cls = len(self.classes)

        self.accept_ema = np.zeros((n_cls, self.n_dirs), dtype=np.float64)
        self.intrusion_ema = np.zeros((n_cls, self.n_dirs), dtype=np.float64)
        self.flip_ema = np.zeros((n_cls, self.n_dirs), dtype=np.float64)
        self.margin_drop_ema = np.zeros((n_cls, self.n_dirs), dtype=np.float64)
        self.usage_count = np.zeros((n_cls, self.n_dirs), dtype=np.int64)
        self.reward = np.zeros((n_cls, self.n_dirs), dtype=np.float64)
        self.sampling_weight = np.ones((n_cls, self.n_dirs), dtype=np.float64) / float(max(1, self.n_dirs))
        self.gamma_scale = np.ones((n_cls, self.n_dirs), dtype=np.float64)
        self.frozen_flag = np.zeros((n_cls, self.n_dirs), dtype=bool)
        self.bad_streak = np.zeros((n_cls, self.n_dirs), dtype=np.int64)

    def sample_direction_ids(self, y: np.ndarray, rs: np.random.RandomState) -> np.ndarray:
        y_arr = np.asarray(y).astype(int).ravel()
        out = np.empty((len(y_arr),), dtype=np.int64)
        for cls in np.unique(y_arr).tolist():
            ci = self.class_to_idx[int(cls)]
            mask = y_arr == int(cls)
            p = self.sampling_weight[ci].copy()
            p[self.frozen_flag[ci]] = 0.0
            s = float(np.sum(p))
            if not np.isfinite(s) or s <= 1e-12:
                p = np.ones((self.n_dirs,), dtype=np.float64)
                p[self.frozen_flag[ci]] = 0.0
                s = float(np.sum(p))
                if s <= 1e-12:
                    p = np.ones((self.n_dirs,), dtype=np.float64)
                    s = float(np.sum(p))
            p = p / max(1e-12, s)
            out[mask] = rs.choice(self.n_dirs, size=int(np.sum(mask)), replace=True, p=p)
        return out

    def lookup_gamma_scale(self, y: np.ndarray, dir_ids: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y).astype(int).ravel()
        d_arr = np.asarray(dir_ids).astype(int).ravel()
        out = np.ones((len(y_arr),), dtype=np.float64)
        for i in range(len(y_arr)):
            ci = self.class_to_idx[int(y_arr[i])]
            out[i] = float(self.gamma_scale[ci, int(d_arr[i])])
        return out

    def update_from_round(self, round_stats: pd.DataFrame) -> None:
        if round_stats.empty:
            return
        lam = float(self.cfg.lambda_ema)
        for _, row in round_stats.iterrows():
            cls = int(row["class_id"])
            did = int(row["direction_id"])
            ci = self.class_to_idx[cls]
            self.accept_ema[ci, did] = (1.0 - lam) * self.accept_ema[ci, did] + lam * float(row["accept_rate"])
            self.intrusion_ema[ci, did] = (1.0 - lam) * self.intrusion_ema[ci, did] + lam * float(row["intrusion"])
            self.flip_ema[ci, did] = (1.0 - lam) * self.flip_ema[ci, did] + lam * float(row["flip_rate"])
            self.margin_drop_ema[ci, did] = (1.0 - lam) * self.margin_drop_ema[ci, did] + lam * float(row["margin_drop_median"])
            self.usage_count[ci, did] += int(row["usage_count"])

        bad_margin = np.maximum(0.0, -self.margin_drop_ema)
        self.reward = (
            float(self.cfg.alpha) * self.accept_ema
            - float(self.cfg.beta) * self.intrusion_ema
            - float(self.cfg.gamma) * self.flip_ema
            - float(self.cfg.eta) * bad_margin
        )

        if self.cfg.enable_gamma_update:
            self.gamma_scale = np.clip(
                self.gamma_scale * np.exp(float(self.cfg.kappa) * self.reward),
                float(self.cfg.gamma_scale_min),
                float(self.cfg.gamma_scale_max),
            )

        if self.cfg.enable_freeze:
            bad_mask = (
                ((self.accept_ema < 0.2) & (self.intrusion_ema > 0.4))
                | ((self.flip_ema > 0.1) & (self.margin_drop_ema < 0.0))
            )
            self.bad_streak = np.where(bad_mask, self.bad_streak + 1, 0)
            self.frozen_flag = self.frozen_flag | (self.bad_streak >= int(self.cfg.freeze_M))

        if self.cfg.enable_weight_update:
            for ci in range(len(self.classes)):
                logits = float(self.cfg.tau) * self.reward[ci]
                logits = logits.copy()
                logits[self.frozen_flag[ci]] = -1e9
                self.sampling_weight[ci] = _softmax(logits)

    def state_dataframe(self, *, base_gamma: float | None = None) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for cls, ci in self.class_to_idx.items():
            for did in range(self.n_dirs):
                gamma_scale = float(self.gamma_scale[ci, did])
                rows.append(
                    {
                        "class_id": int(cls),
                        "direction_id": int(did),
                        "usage_count": int(self.usage_count[ci, did]),
                        "accept_ema": float(self.accept_ema[ci, did]),
                        "intrusion_ema": float(self.intrusion_ema[ci, did]),
                        "flip_ema": float(self.flip_ema[ci, did]),
                        "margin_drop_ema": float(self.margin_drop_ema[ci, did]),
                        "reward_i": float(self.reward[ci, did]),
                        "sampling_weight_i": float(self.sampling_weight[ci, did]),
                        "gamma_scale_i": gamma_scale,
                        "gamma_eff_i": float(base_gamma) * gamma_scale if base_gamma is not None else None,
                        "frozen_flag_i": bool(self.frozen_flag[ci, did]),
                        "bad_streak": int(self.bad_streak[ci, did]),
                    }
                )
        return pd.DataFrame(rows).sort_values(["class_id", "direction_id"]).reset_index(drop=True)
