from typing import List

import numpy as np


def smooth_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=False)
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(x.astype(np.float64), kernel, mode="same")


def smooth_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    x = x.astype(np.float64)
    if x.size == 0:
        return x
    alpha = float(alpha)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def smooth_kalman_1d(x: np.ndarray, q_over_r: float) -> np.ndarray:
    x = x.astype(np.float64)
    if x.size == 0:
        return x
    q = float(q_over_r)
    r = 1.0
    x_hat = x[0]
    P = 1.0
    out = np.empty_like(x)
    out[0] = x_hat
    for i in range(1, x.size):
        # predict
        P = P + q
        # update
        K = P / (P + r)
        x_hat = x_hat + K * (x[i] - x_hat)
        P = (1.0 - K) * P
        out[i] = x_hat
    return out

