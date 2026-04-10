from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from .base import BaseTransform


@dataclass
class RiemannianAlignTransform(BaseTransform):
    """Riemannian Alignment (RA) for trial-level raw/DE sequences."""

    mode: str = "auto"  # auto | band | flat | pca->flat
    input_repr: str = "signal"  # signal | cov
    n_channels: int = 62
    n_bands: int = 5
    zscore: bool = True
    zscore_eps: float = 1e-6
    shrinkage: float = 0.0
    trace_normalize: bool = True
    cov_eps: float = 1e-6
    mean_mode: str = "logeuclidean"  # logeuclidean | euclidean
    min_eig: float = 1e-6
    verbose: bool = True

    reference_: Optional[np.ndarray] = None
    reference_bands_: Optional[List[np.ndarray]] = None
    ref_inv_sqrt_: Optional[np.ndarray] = None
    ref_inv_sqrt_bands_: Optional[List[np.ndarray]] = None
    mode_resolved_: Optional[str] = None
    _warned: bool = field(default=False, init=False, repr=False)

    def _log(self, msg: str) -> None:
        if self.verbose and not self._warned:
            print(msg)
            self._warned = True

    def _resolve_mode(self, feat_dim: int) -> str:
        mode = (self.mode or "auto").lower()
        if mode == "auto":
            return "band" if feat_dim == self.n_channels * self.n_bands else "flat"
        if mode == "pca":
            return "flat"
        if mode in {"band", "flat"}:
            return mode
        raise ValueError(f"Unknown RiemannianAlignTransform mode: {self.mode}")

    def _preprocess_trial(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"RiemannianAlignTransform expects 2D (T, C), got {X.shape}")
        if not self.zscore:
            return X
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < self.zscore_eps, 1.0, std)
        return (X - mu) / std

    def _covariance(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        if T < 2:
            d = X.shape[1]
            return np.eye(d, dtype=np.float64)
        C = (X.T @ X) / max(1, T - 1)
        if self.shrinkage > 0:
            d = C.shape[0]
            tr = float(np.trace(C))
            C = (1.0 - self.shrinkage) * C + self.shrinkage * (tr / d) * np.eye(d)
        if self.trace_normalize:
            tr = float(np.trace(C))
            if tr > 0:
                C = C / tr
        if self.cov_eps > 0:
            d = C.shape[0]
            C = C + self.cov_eps * np.eye(d)
        return C

    def _as_cov(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"RiemannianAlignTransform expects 2D input, got {X.shape}")
        if X.shape[0] != X.shape[1]:
            raise ValueError(f"RiemannianAlignTransform cov input must be square, got {X.shape}")
        C = (X + X.T) * 0.5
        if self.cov_eps > 0:
            d = C.shape[0]
            C = C + self.cov_eps * np.eye(d)
        return C

    def _logm_spd(self, C: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eigh(C)
        w = np.clip(w, self.min_eig, None)
        logC = (v * np.log(w)) @ v.T
        return (logC + logC.T) * 0.5

    def _expm_spd(self, C: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eigh(C)
        expC = (v * np.exp(w)) @ v.T
        return (expC + expC.T) * 0.5

    def _mean_cov(self, covs: List[np.ndarray]) -> np.ndarray:
        if len(covs) == 0:
            raise ValueError("RiemannianAlignTransform received empty covariance list")
        mode = (self.mean_mode or "logeuclidean").lower()
        if mode == "euclidean":
            C = np.mean(covs, axis=0)
            return (C + C.T) * 0.5
        if mode == "logeuclidean":
            logs = [self._logm_spd(C) for C in covs]
            mean_log = np.mean(logs, axis=0)
            return self._expm_spd(mean_log)
        raise ValueError(f"Unknown mean_mode: {self.mean_mode}")

    def _inv_sqrt(self, C: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eigh(C)
        w = np.clip(w, self.min_eig, None)
        inv_sqrt = (v * (w ** -0.5)) @ v.T
        return (inv_sqrt + inv_sqrt.T) * 0.5

    def _split_bands(self, X: np.ndarray) -> List[np.ndarray]:
        T, feat = X.shape
        expected = self.n_channels * self.n_bands
        if feat != expected:
            raise ValueError(f"band mode expects feat={expected}, got {feat}")
        Xb = X.reshape(T, self.n_channels, self.n_bands)
        return [Xb[:, :, b] for b in range(self.n_bands)]

    def _merge_bands(self, bands: List[np.ndarray]) -> np.ndarray:
        Xb = np.stack(bands, axis=2)
        return Xb.reshape(Xb.shape[0], self.n_channels * self.n_bands)

    def fit_trials(self, trials: Iterable[np.ndarray]) -> "RiemannianAlignTransform":
        trials = list(trials)
        if len(trials) == 0:
            raise ValueError("RiemannianAlignTransform.fit_trials received empty trials")
        if not (0.0 <= self.shrinkage <= 1.0):
            raise ValueError("RiemannianAlignTransform shrinkage must be in [0, 1]")

        feat_dim = np.asarray(trials[0]).shape[1]
        mode = self._resolve_mode(feat_dim)
        self.mode_resolved_ = mode
        input_repr = (self.input_repr or "signal").lower()
        if input_repr not in {"signal", "cov"}:
            raise ValueError(f"Unknown RiemannianAlignTransform input_repr: {self.input_repr}")
        if input_repr == "cov" and mode != "flat":
            raise ValueError("RiemannianAlignTransform input_repr=cov requires mode=flat")
        self._log(f"[RA] enabled, mode={mode}, mean={self.mean_mode}, input={input_repr}")

        if mode == "band":
            covs_by_band: List[List[np.ndarray]] = [[] for _ in range(self.n_bands)]
            for X in trials:
                if np.asarray(X).shape[1] != feat_dim:
                    raise ValueError("RiemannianAlignTransform trials must share feature dimension")
                Xp = self._preprocess_trial(X)
                bands = self._split_bands(Xp)
                for b in range(self.n_bands):
                    covs_by_band[b].append(self._covariance(bands[b]))
            self.reference_bands_ = []
            self.ref_inv_sqrt_bands_ = []
            for covs in covs_by_band:
                ref = self._mean_cov(covs)
                self.reference_bands_.append(ref)
                self.ref_inv_sqrt_bands_.append(self._inv_sqrt(ref))
            self.reference_ = None
            self.ref_inv_sqrt_ = None
        else:
            covs = []
            for X in trials:
                if input_repr == "cov":
                    covs.append(self._as_cov(X))
                else:
                    if np.asarray(X).shape[1] != feat_dim:
                        raise ValueError("RiemannianAlignTransform trials must share feature dimension")
                    covs.append(self._covariance(self._preprocess_trial(X)))
            ref = self._mean_cov(covs)
            self.reference_ = ref
            self.ref_inv_sqrt_ = self._inv_sqrt(ref)
            self.reference_bands_ = None
            self.ref_inv_sqrt_bands_ = None
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, meta: Optional[dict] = None) -> "RiemannianAlignTransform":
        if isinstance(X, (list, tuple)):
            return self.fit_trials(X)
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return self.fit_trials([X])
        raise ValueError("RiemannianAlignTransform.fit expects trials list or (T,C) array")

    def transform(self, X: np.ndarray, meta: Optional[dict] = None) -> np.ndarray:
        if self.mode_resolved_ is None:
            raise RuntimeError("RiemannianAlignTransform.fit_trials() must be called first.")
        input_repr = (self.input_repr or "signal").lower()
        if input_repr not in {"signal", "cov"}:
            raise ValueError(f"Unknown RiemannianAlignTransform input_repr: {self.input_repr}")
        if input_repr == "cov" and self.mode_resolved_ != "flat":
            raise ValueError("RiemannianAlignTransform input_repr=cov requires mode=flat")
        if input_repr == "cov":
            C = self._as_cov(X)
            if self.ref_inv_sqrt_ is None:
                raise RuntimeError("RiemannianAlignTransform reference is missing")
            W = self.ref_inv_sqrt_
            return (W @ C @ W).astype(np.float64)
        Xp = self._preprocess_trial(X)
        if self.mode_resolved_ == "band":
            if self.ref_inv_sqrt_bands_ is None:
                raise RuntimeError("RiemannianAlignTransform band reference is missing")
            bands = self._split_bands(Xp)
            aligned = [b @ W for b, W in zip(bands, self.ref_inv_sqrt_bands_)]
            return self._merge_bands(aligned)
        if self.ref_inv_sqrt_ is None:
            raise RuntimeError("RiemannianAlignTransform reference is missing")
        return Xp @ self.ref_inv_sqrt_

    def transform_trials(self, trials: Iterable[np.ndarray]) -> List[np.ndarray]:
        return [self.transform(t) for t in trials]
