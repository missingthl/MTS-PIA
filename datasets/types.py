from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class FoldData:
    """Sample-level fold data with optional trial_id metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    trial_id_train: Optional[np.ndarray] = None
    trial_id_test: Optional[np.ndarray] = None


@dataclass
class TrialFoldData:
    """Trial-level fold data for manifold streams."""

    trials_train: List[np.ndarray]
    y_trial_train: np.ndarray
    trials_test: List[np.ndarray]
    y_trial_test: np.ndarray
    trial_id_train: Optional[np.ndarray] = None
    trial_id_test: Optional[np.ndarray] = None
