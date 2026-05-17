from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .telm2 import TELM2Config, TELM2Transformer


@dataclass
class PIADirectionalAffineAugmenter:
    """PIA directional affine augmentation for sample-level features."""

    gamma: float = 0.2
    n_iters: int = 3
    activation: str = "sine"
    bias_update_mode: str = "residual"
    C_repr: float = 1.0
    seed: Optional[int] = None

    w_dir_: Optional[np.ndarray] = None
    mu_: Optional[np.ndarray] = None
    recon_err_: Optional[List[float]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PIADirectionalAffineAugmenter":
        X = np.asarray(X, dtype=np.float64)
        cfg = TELM2Config(
            r_dimension=1,
            n_iters=int(self.n_iters),
            activation=self.activation,
            bias_update_mode=self.bias_update_mode,
            C_repr=float(self.C_repr),
            enable_repr_learning=True,
            seed=None if self.seed is None else int(self.seed),
        )
        telm = TELM2Transformer(cfg).fit(X)
        arts = telm.get_artifacts()
        W = np.asarray(arts.W, dtype=np.float64)
        w = W.reshape(-1)
        norm = np.linalg.norm(w) + 1e-12
        self.w_dir_ = w / norm
        self.mu_ = X.mean(axis=0, keepdims=True)
        self.recon_err_ = list(getattr(arts, "recon_err", []))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.w_dir_ is None or self.mu_ is None:
            raise RuntimeError("PIADirectionalAffineAugmenter.fit() must be called first.")
        X = np.asarray(X, dtype=np.float64)
        X_c = X - self.mu_
        t = X_c @ self.w_dir_.reshape(-1, 1)
        comp = t @ self.w_dir_.reshape(1, -1)
        res = X_c - comp
        return (self.mu_ + (1.0 + self.gamma) * comp + res).astype(np.float64)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)

    def state(self) -> dict:
        return {
            "w_dir": None if self.w_dir_ is None else np.asarray(self.w_dir_),
            "mu": None if self.mu_ is None else np.asarray(self.mu_),
            "recon_err": None if self.recon_err_ is None else list(self.recon_err_),
        }
