"""Regression probe branch for Route B geometry front-end validation."""

from .evaluator import RegressionEvalResult, evaluate_regression
from .regressor import RegressorConfig, build_regressor, regressor_params_dict
from .representation import (
    RegressionRepresentationConfig,
    RegressionRepresentationState,
    build_regression_representation,
)

__all__ = [
    "RegressionEvalResult",
    "RegressionRepresentationConfig",
    "RegressionRepresentationState",
    "RegressorConfig",
    "build_regression_representation",
    "build_regressor",
    "evaluate_regression",
    "regressor_params_dict",
]
