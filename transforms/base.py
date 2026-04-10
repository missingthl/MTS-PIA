from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


class BaseTransform:
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, meta: Optional[dict] = None) -> "BaseTransform":
        return self

    def transform(self, X: np.ndarray, meta: Optional[dict] = None) -> np.ndarray:
        return X

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, meta: Optional[dict] = None
    ) -> np.ndarray:
        return self.fit(X, y=y, meta=meta).transform(X, meta=meta)


class NoOpTransform(BaseTransform):
    def fit_trials(
        self, trials: Iterable[np.ndarray], y: Optional[np.ndarray] = None, meta: Optional[dict] = None
    ) -> "NoOpTransform":
        return self

    def transform_trials(self, trials: Iterable[np.ndarray]) -> List[np.ndarray]:
        return [np.asarray(t) for t in trials]
