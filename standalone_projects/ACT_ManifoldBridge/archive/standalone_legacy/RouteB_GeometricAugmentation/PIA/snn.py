from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


Activation = Literal["sigmoid", "sine"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _inv_sigmoid(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def _inv_sine(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.clip(y, -1.0 + eps, 1.0 - eps)
    return np.arcsin(y)


def _activate(x: np.ndarray, kind: Activation) -> np.ndarray:
    if kind == "sigmoid":
        return _sigmoid(x)
    if kind == "sine":
        return np.sin(x)
    raise ValueError(f"Unknown activation: {kind}")


def _inv_activate(y: np.ndarray, kind: Activation) -> np.ndarray:
    if kind == "sigmoid":
        return _inv_sigmoid(y)
    if kind == "sine":
        return _inv_sine(y)
    raise ValueError(f"Unknown activation: {kind}")


@dataclass
class _MinMaxState:
    a: float
    b: float
    xmin: np.ndarray  # (K,)
    xmax: np.ndarray  # (K,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        span = np.maximum(self.xmax - self.xmin, 1e-12)
        return (X - self.xmin) / span * (self.b - self.a) + self.a

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        span = np.maximum(self.xmax - self.xmin, 1e-12)
        return (X - self.a) / (self.b - self.a) * span + self.xmin


def _fit_minmax(X: np.ndarray, a: float, b: float) -> _MinMaxState:
    X = np.asarray(X, dtype=np.float64)
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    return _MinMaxState(a=float(a), b=float(b), xmin=xmin, xmax=xmax)


def _onehot_pm1(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-hot in {-1, +1}. Shape: (N, K)."""
    y = np.asarray(y).astype(int).ravel()
    idx = np.searchsorted(classes, y)
    N = y.shape[0]
    K = classes.shape[0]
    oh = np.zeros((N, K), dtype=np.float64)
    oh[np.arange(N), idx] = 1.0
    return oh * 2.0 - 1.0


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(np.clip(x, -50.0, 50.0))
    return e / np.sum(e, axis=1, keepdims=True)


class SNNClassifier:
    """
    SNNs classifier (paper-style, gradient-free):
    - Iteratively fits residual targets via a pseudo-inverse / ridge closed form.
    - Uses u(.) normalization (MinMax) + invertible activation g(.) (sigmoid/sine).

    This matches the spirit of the paper's Eq.(8)(9) and the original MATLAB/OS-ELM
    implementation, but fixes the common "single scaler reused across layers" bug
    by storing per-layer scaling state.
    """

    def __init__(
        self,
        n_nodes: int = 3,
        C: float = 4.0,
        activation: Activation = "sigmoid",
        eps_residual: float = 1e-7,
    ):
        self.n_nodes = int(n_nodes)
        self.C = float(C)
        self.activation: Activation = activation
        self.eps_residual = float(eps_residual)

        self.classes_: Optional[np.ndarray] = None
        self.layers_: List[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SNNClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)

        T = _onehot_pm1(y, self.classes_)  # (N, K)
        Y = np.zeros_like(T)  # prediction accumulator

        D = X.shape[1]
        lam = 1.0 / max(self.C, 1e-12)
        A = X.T @ X + lam * np.eye(D, dtype=np.float64)  # shared across nodes

        self.layers_ = []

        for _ in range(self.n_nodes):
            E = (T - Y) + self.eps_residual  # residual

            if self.activation == "sigmoid":
                u_a, u_b = 0.01, 0.99
            else:
                u_a, u_b = -1.0, 1.0

            mm = _fit_minmax(E, a=u_a, b=u_b)
            Eu = mm.transform(E)
            Einv = _inv_activate(Eu, self.activation)  # (N, K)

            B = X.T @ Einv  # (D, K)
            W = np.linalg.solve(A, B)  # (D, K)

            bias = float(np.mean(X @ W - Einv))
            Gu = _activate(X @ W - bias, self.activation)  # (N, K) in u-range
            F = mm.inverse_transform(Gu)  # back to residual scale (approx [-1,1])

            Y = Y + F
            self.layers_.append({"W": W, "bias": bias, "mm": mm})

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or not self.layers_:
            raise RuntimeError("SNNClassifier.fit() must be called first.")
        X = np.asarray(X, dtype=np.float64)

        Y = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        for layer in self.layers_:
            W = layer["W"]
            bias = layer["bias"]
            mm: _MinMaxState = layer["mm"]
            Gu = _activate(X @ W - bias, self.activation)
            Y = Y + mm.inverse_transform(Gu)
        return Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax_rows(self.decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).astype(int).ravel()
        return float(np.mean(self.predict(X) == y))

