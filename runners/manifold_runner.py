from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os
import sys

import numpy as np

from datasets.types import TrialFoldData

# Ensure telm_aec is importable without installation.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_TELM_SRC = os.path.join(_PROJECT_ROOT, "archive", "legacy_code", "block", "src")
if _TELM_SRC not in sys.path:
    sys.path.append(_TELM_SRC)

from telm_aec.manifold import LinearManifoldClassifier


@dataclass
class ManifoldRunner:
    mode: str = "band"
    pca_dim: Optional[int] = None
    eps: float = 1e-3
    classifier: str = "svm"
    C: float = 1.0
    C_grid: Optional[list] = None
    standardize: bool = True
    min_eig: Optional[float] = None
    ra_mode: str = "euclidean"
    align_mode: str = "ra"
    domain_level: Optional[str] = None
    mdm_metric: str = "logeuclidean"
    mdm_mean: str = "logeuclidean"
    debug: bool = False
    max_iter: int = 5000
    expected_channels: Optional[int] = None
    input_kind: str = "signal"
    n_bands: int = 5

    def _domain_from_trial_id(self, trial_id: str) -> str:
        tid = str(trial_id)
        if "_s" not in tid or "_t" not in tid:
            raise ValueError(f"Invalid trial_id format: {trial_id}")
        subject, rest = tid.rsplit("_s", 1)
        session_str, _trial_str = rest.split("_t", 1)
        if self.domain_level == "subject":
            return subject
        if self.domain_level == "session":
            return f"{subject}_s{session_str}"
        raise ValueError(f"Unknown domain_level: {self.domain_level}")

    def fit_predict(self, fold: TrialFoldData) -> Dict[str, np.ndarray]:
        domains_train = None
        domains_test = None
        if self.domain_level:
            if fold.trial_id_train is None or fold.trial_id_test is None:
                raise ValueError("domain alignment requires trial_id in fold")
            domains_train = [self._domain_from_trial_id(t) for t in fold.trial_id_train]
            domains_test = [self._domain_from_trial_id(t) for t in fold.trial_id_test]
        clf = LinearManifoldClassifier(
            mode=self.mode,
            classifier=self.classifier,
            C=self.C,
            C_list=self.C_grid,
            eps=self.eps,
            pca_dim=self.pca_dim,
            standardize=self.standardize,
            min_eig=self.min_eig if self.min_eig is not None else 1e-6,
            ra_mode=self.ra_mode,
            align_mode=self.align_mode,
            mdm_metric=self.mdm_metric,
            mdm_mean=self.mdm_mean,
            debug=self.debug,
            max_iter=self.max_iter,
            expected_channels=self.expected_channels,
            input_kind=self.input_kind,
            n_bands=self.n_bands,
        )
        clf.fit(fold.trials_train, fold.y_trial_train, domains=domains_train)
        proba_test = clf.predict_proba(fold.trials_test, domains=domains_test)
        proba_train = clf.predict_proba(fold.trials_train, domains=domains_train)

        return {
            "trial_proba_test": np.asarray(proba_test, dtype=np.float64),
            "trial_proba_train": np.asarray(proba_train, dtype=np.float64),
            "y_trial_test": np.asarray(fold.y_trial_test).astype(int).ravel(),
            "y_trial_train": np.asarray(fold.y_trial_train).astype(int).ravel(),
        }
