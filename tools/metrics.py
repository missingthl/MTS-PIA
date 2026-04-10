from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    return float(np.mean(y_true == y_pred))


def accuracy_from_proba(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_pred = np.argmax(proba, axis=1)
    return accuracy(y_true, y_pred)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    return float(f1_score(y_true, y_pred, average="macro"))


def macro_f1_from_proba(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_pred = np.argmax(proba, axis=1)
    return macro_f1(y_true, y_pred)
