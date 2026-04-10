# src/telm_aec/telm2.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ========== 配置 ==========
@dataclass
class TELM2Config:
    r_dimension: int = 1            # 模板数 L
    n_iters: int = 3                # 迭代次数（=0/负数 将视为不开启迭代）
    C_repr: float = 1.0             # 表征岭系数
    activation: str = "sine"        # "sine" | "sigmoid"
    bias_lr: float = 0.25           # 偏置更新步长
    orthogonalize: bool = True      # 回写后是否行正交
    enable_repr_learning: bool = True  # 是否启用表征学习
    bias_update_mode: str = "act_mean" # ★ "off" | "act_mean" | "residual"
    seed: Optional[int] = None      # 随机种子（None 表示非确定）

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TELM2Config":
        fields = {}
        for k in cls.__dataclass_fields__.keys():  # type: ignore
            fields[k] = d.get(k, getattr(cls, k))
        return cls(**fields)

# ========== 工具 ==========
def _act(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sine":
        return np.sin(x)
    elif kind == "sigmoid":
        arr = np.asarray(x, dtype=np.float64)
        out = np.empty_like(arr, dtype=np.float64)
        pos = arr >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
        exp_arr = np.exp(arr[~pos])
        out[~pos] = exp_arr / (1.0 + exp_arr)
        return out
    else:
        raise ValueError(f"unknown activation: {kind}")

def _inv_act(y: np.ndarray, kind: str) -> np.ndarray:
    eps = 1e-6
    if kind == "sine":
        y_clip = np.clip(y, -1 + eps, 1 - eps)
        return np.arcsin(y_clip)
    elif kind == "sigmoid":
        y_clip = np.clip(y, eps, 1 - eps)
        return np.log(y_clip / (1 - y_clip))
    else:
        raise ValueError(f"unknown activation: {kind}")

def _scale_to_range(X: np.ndarray, a=-1.0, b=1.0) -> np.ndarray:
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    eps = 1e-8
    scale = (xmax - xmin)
    scale[scale < eps] = eps
    return (X - xmin) / scale * (b - a) + a

def _row_orth(W: np.ndarray) -> np.ndarray:
    # 简单行正交（Gram-Schmidt）
    Wo = W.copy()
    for i in range(W.shape[0]):
        for j in range(i):
            denom = np.dot(Wo[j], Wo[j]) + 1e-12
            proj = np.dot(Wo[i], Wo[j]) / denom
            Wo[i] -= proj * Wo[j]
        n = np.linalg.norm(Wo[i]) + 1e-12
        Wo[i] /= n
    return Wo

# ========== 产物 ==========
@dataclass
class TELM2Artifacts:
    W: np.ndarray                  # (L, d)
    b: np.ndarray                  # (L,)
    recon_err: List[float]         # 每次迭代的重构误差

# ========== 主类 ==========
class TELM2Transformer:
    def __init__(self, cfg: TELM2Config):
        self.cfg = cfg
        self._arts: Optional[TELM2Artifacts] = None

    def get_artifacts(self) -> TELM2Artifacts:
        if self._arts is None:
            raise RuntimeError("TELM2Transformer.fit() must be called first.")
        return self._arts

    def fit(
        self,
        X_tr: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        target_override: Optional[np.ndarray] = None,
    ) -> "TELM2Transformer":
        """严格隔离：仅用训练集 X_tr 学模板。"""
        X_tr = np.asarray(X_tr, dtype=np.float64)
        L = int(self.cfg.r_dimension)
        d = X_tr.shape[1]
        n_iters = int(self.cfg.n_iters)
        seed = None if self.cfg.seed is None else int(self.cfg.seed)
        weight_arr: Optional[np.ndarray] = None
        mean_weight_arr: Optional[np.ndarray] = None
        sqrt_weight_arr: Optional[np.ndarray] = None
        if sample_weights is not None:
            weight_arr = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            if weight_arr.shape[0] != X_tr.shape[0]:
                raise ValueError("sample_weights must align with X_tr rows")
            if np.any(~np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
                raise ValueError("sample_weights must be finite and nonnegative")
            if float(np.sum(weight_arr)) <= 0.0:
                raise ValueError("sample_weights must have positive total mass")
            mean_weight_arr = weight_arr / float(np.sum(weight_arr))
            sqrt_weight_arr = np.sqrt(weight_arr)[:, None]
        target_override_arr: Optional[np.ndarray] = None
        if target_override is not None:
            target_override_arr = np.asarray(target_override, dtype=np.float64)
            if target_override_arr.ndim != 2:
                raise ValueError("target_override must be a 2D matrix")
            if target_override_arr.shape[0] != X_tr.shape[0]:
                raise ValueError("target_override must align with X_tr rows")
            if target_override_arr.shape[1] != d:
                raise ValueError("target_override must match X_tr feature dimension")
            if np.any(~np.isfinite(target_override_arr)):
                raise ValueError("target_override must be finite")

        # A 开关：关闭表征学习 → 随机模板/最弱基线
        if not self.cfg.enable_repr_learning or n_iters <= 0:
            rng = np.random.default_rng(seed)
            W = rng.uniform(-1.0, 1.0, size=(L, d)).astype(np.float64)
            b = np.zeros((L,), dtype=np.float64)
            self._arts = TELM2Artifacts(W=W, b=b, recon_err=[])
            return self

        # 初始化
        rng = np.random.default_rng(seed)
        W = rng.uniform(-1.0, 1.0, size=(L, d)).astype(np.float64)
        b = np.zeros((L,), dtype=np.float64)
        if self.cfg.orthogonalize and L > 1:
            W = _row_orth(W)

        recon_curve: List[float] = []

        for it in range(n_iters):
            # 1) 前向：P = f(XW^T + b)
            P = _act(X_tr @ W.T + b[None, :], self.cfg.activation)   # (N, L)

            # 2) 反激活目标 Y4（把输入缩放到激活域后取反函数）
            if target_override_arr is None:
                X_scaled = _scale_to_range(X_tr, a=-1.0, b=1.0)
                Y4 = _inv_act(X_scaled, self.cfg.activation)         # (N, d)
            else:
                Y4 = np.asarray(target_override_arr, dtype=np.float64)

            # 3) 岭回归闭式解：YYM = argmin ||P·M - Y4||^2 + (1/C)||M||^2
            C = float(self.cfg.C_repr)
            if sqrt_weight_arr is None:
                A = P.T @ P + (1.0 / C) * np.eye(L)
                B = P.T @ Y4
            else:
                Pw = P * sqrt_weight_arr
                Yw = Y4 * sqrt_weight_arr
                A = Pw.T @ Pw + (1.0 / C) * np.eye(L)
                B = Pw.T @ Yw
            YYM = np.linalg.solve(A, B)                              # (L, d)

            # 4) 回写
            W = YYM

            # 5) 偏置更新（三模式：off/act_mean/residual）
            mode = self.cfg.bias_update_mode.lower()
            if mode == "act_mean":
                # 目标：把每个模板通道的平均激活拉回目标均值
                if mean_weight_arr is None:
                    mu = P.mean(axis=0)                              # (L,)
                else:
                    mu = np.sum(P * mean_weight_arr[:, None], axis=0)
                target = 0.0 if self.cfg.activation == "sine" else 0.5
                b = b - self.cfg.bias_lr * (mu - target)
            elif mode == "residual":
                # 吸收“反域重构残差”的平均值（保守标量）
                resid = Y4 - P @ YYM                                  # (N, d)
                if mean_weight_arr is None:
                    delta_b = float(resid.mean())                     # scalar
                else:
                    delta_b = float(np.sum(resid * mean_weight_arr[:, None]))
                b = b + self.cfg.bias_lr * delta_b
            # else: "off" 不更新

            # 6) 行正交（可选）
            if self.cfg.orthogonalize and L > 1:
                W = _row_orth(W)

            # 7) 记录重构误差（相对）
            resid_eval = P @ YYM - Y4
            if sqrt_weight_arr is None:
                num = np.linalg.norm(resid_eval)
                den = np.linalg.norm(Y4) + 1e-12
            else:
                num = np.linalg.norm(resid_eval * sqrt_weight_arr)
                den = np.linalg.norm(Y4 * sqrt_weight_arr) + 1e-12
            recon_curve.append(float(num / den))

        self._arts = TELM2Artifacts(W=W, b=b, recon_err=recon_curve)
        return self
