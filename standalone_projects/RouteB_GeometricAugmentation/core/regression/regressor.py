from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegressorConfig:
    regressor_type: str
    alpha: float
    l1_ratio: float = 0.5
    max_iter: int = 5000
    seed: int = 0


def build_regressor(cfg: RegressorConfig) -> Pipeline:
    kind = str(cfg.regressor_type).strip().lower()
    if kind == "ridge":
        model = Ridge(alpha=float(cfg.alpha))
    elif kind == "elasticnet":
        model = ElasticNet(
            alpha=float(cfg.alpha),
            l1_ratio=float(cfg.l1_ratio),
            max_iter=int(cfg.max_iter),
            random_state=int(cfg.seed),
        )
    else:
        raise ValueError(f"Unsupported regressor_type: {cfg.regressor_type}")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def regressor_params_dict(cfg: RegressorConfig) -> Dict[str, object]:
    return {
        "regressor_type": str(cfg.regressor_type).strip().lower(),
        "alpha": float(cfg.alpha),
        "l1_ratio": float(cfg.l1_ratio),
        "max_iter": int(cfg.max_iter),
        "seed": int(cfg.seed),
    }
