from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.regression.representation import RegressionRepresentationState
from core.regression.regressor import RegressorConfig, build_regressor, regressor_params_dict


@dataclass
class RegressionEvalResult:
    dataset: str
    regressor: str
    seed: int
    rmse: float
    mae: float
    r2: float
    y_pred: np.ndarray
    params: Dict[str, object] = field(default_factory=dict)


def evaluate_regression(
    rep_state: RegressionRepresentationState,
    reg_cfg: RegressorConfig,
) -> RegressionEvalResult:
    model = build_regressor(reg_cfg)
    model.fit(rep_state.X_train_z, rep_state.y_train)
    y_pred = np.asarray(model.predict(rep_state.X_test_z), dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(rep_state.y_test, y_pred)))
    mae = float(mean_absolute_error(rep_state.y_test, y_pred))
    r2 = float(r2_score(rep_state.y_test, y_pred))
    return RegressionEvalResult(
        dataset=str(rep_state.dataset),
        regressor=str(reg_cfg.regressor_type).strip().lower(),
        seed=int(reg_cfg.seed),
        rmse=rmse,
        mae=mae,
        r2=r2,
        y_pred=y_pred,
        params=regressor_params_dict(reg_cfg),
    )
