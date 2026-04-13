from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class LocalKNNGateConfig:
    k: int = 20
    tau_purity: float = 0.7
    metric: str = "euclidean"
    algorithm: str = "auto"
    query_batch_size: int = 4096


class ReadOnlyLocalKNNGate:
    def __init__(self, cfg: LocalKNNGateConfig) -> None:
        self.cfg = cfg
        self._X_anchor: Optional[np.ndarray] = None
        self._y_anchor: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self._k_eff: Optional[int] = None

    def fit(self, X_anchor: np.ndarray, y_anchor: np.ndarray) -> "ReadOnlyLocalKNNGate":
        X = np.asarray(X_anchor, dtype=np.float32)
        y = np.asarray(y_anchor).astype(int).ravel()
        if X.ndim != 2:
            raise ValueError("X_anchor must be 2D.")
        if len(X) != len(y):
            raise ValueError("X_anchor and y_anchor length mismatch.")
        if len(X) == 0:
            raise ValueError("X_anchor must be non-empty.")
        k_eff = int(min(max(1, int(self.cfg.k)), len(X)))
        nn = NearestNeighbors(
            n_neighbors=k_eff,
            metric=str(self.cfg.metric),
            algorithm=str(self.cfg.algorithm),
        )
        nn.fit(X)
        self._X_anchor = X
        self._y_anchor = y
        self._nn = nn
        self._k_eff = k_eff
        return self

    @property
    def fitted(self) -> bool:
        return self._nn is not None

    def evaluate_batch(
        self,
        X_cand: np.ndarray,
        source_labels: np.ndarray,
        *,
        direction_ids: Optional[np.ndarray] = None,
        gamma_used: Optional[np.ndarray] = None,
        source_tids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if self._nn is None or self._y_anchor is None:
            raise RuntimeError("Gate must be fit before evaluate_batch.")

        X = np.asarray(X_cand, dtype=np.float32)
        y = np.asarray(source_labels).astype(int).ravel()
        if X.ndim != 2:
            raise ValueError("X_cand must be 2D.")
        if len(X) != len(y):
            raise ValueError("X_cand and source_labels length mismatch.")

        n = int(len(y))
        if n == 0:
            return np.zeros((0,), dtype=bool), {
                "gate3_accept_count": 0,
                "gate3_reject_count": 0,
                "gate3_accept_rate": 0.0,
                "gate3_reject_reason_summary": {
                    "majority_mismatch_only": 0,
                    "purity_below_tau_only": 0,
                    "both": 0,
                },
                "candidate_rows": [],
            }

        dir_arr = (
            np.full((n,), -1, dtype=np.int64)
            if direction_ids is None
            else np.asarray(direction_ids).astype(np.int64).ravel()
        )
        gamma_arr = (
            np.zeros((n,), dtype=np.float64)
            if gamma_used is None
            else np.asarray(gamma_used, dtype=np.float64).ravel()
        )
        tid_arr = (
            np.asarray([""] * n, dtype=object)
            if source_tids is None
            else np.asarray(source_tids)
        )
        if not (len(dir_arr) == len(gamma_arr) == len(tid_arr) == n):
            raise ValueError("Gate3 auxiliary arrays length mismatch.")

        t0 = time.perf_counter()
        batch_size = max(1, int(self.cfg.query_batch_size))
        d_parts: List[np.ndarray] = []
        i_parts: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            d_b, i_b = self._nn.kneighbors(X[start:stop], return_distance=True)
            d_parts.append(np.asarray(d_b, dtype=np.float64))
            i_parts.append(np.asarray(i_b, dtype=np.int64))
        dists = np.vstack(d_parts) if d_parts else np.empty((0, int(self._k_eff or 0)), dtype=np.float64)
        idx = np.vstack(i_parts) if i_parts else np.empty((0, int(self._k_eff or 0)), dtype=np.int64)
        y_nb = self._y_anchor[idx]
        same_class = (y_nb == y[:, None]).astype(np.float64)
        purity = np.mean(same_class, axis=1).astype(np.float64)
        intrusion = (1.0 - purity).astype(np.float64)

        majority = np.empty((n,), dtype=np.int64)
        for i in range(n):
            labels = y_nb[i]
            classes, counts = np.unique(labels, return_counts=True)
            majority[i] = int(classes[np.argmax(counts)])

        accept_majority = majority == y
        accept_purity = purity >= float(self.cfg.tau_purity)
        keep = accept_majority & accept_purity

        mean_same = np.zeros((n,), dtype=np.float64)
        mean_other = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            same_mask = y_nb[i] == y[i]
            other_mask = ~same_mask
            mean_same[i] = float(np.mean(dists[i][same_mask])) if np.any(same_mask) else float("inf")
            mean_other[i] = float(np.mean(dists[i][other_mask])) if np.any(other_mask) else float("inf")

        reject_rows: List[Dict[str, object]] = []
        majority_only = 0
        purity_only = 0
        both = 0
        for i in range(n):
            maj_fail = not bool(accept_majority[i])
            pur_fail = not bool(accept_purity[i])
            if not keep[i]:
                if maj_fail and pur_fail:
                    both += 1
                    reason = "majority_mismatch_and_purity"
                elif maj_fail:
                    majority_only += 1
                    reason = "majority_mismatch"
                else:
                    purity_only += 1
                    reason = "purity_below_tau"
                reject_rows.append(
                    {
                        "reject_by_gate3": True,
                        "reject_reason": reason,
                        "source_label": int(y[i]),
                        "source_direction_id": int(dir_arr[i]),
                        "gamma_used": float(gamma_arr[i]),
                        "source_tid": str(tid_arr[i]),
                        "knn_majority_label": int(majority[i]),
                        "knn_same_class_purity": float(purity[i]),
                        "knn_intrusion_ratio": float(intrusion[i]),
                        "mean_distance_same_class": float(mean_same[i]),
                        "mean_distance_other_class": float(mean_other[i]),
                    }
                )

        runtime_sec = float(time.perf_counter() - t0)
        gamma_stats: Dict[str, Dict[str, float]] = {}
        if gamma_arr.size:
            for g in sorted(set(np.round(gamma_arr.astype(np.float64), 8).tolist())):
                mask_g = np.isclose(gamma_arr, g)
                gamma_stats[str(float(g))] = {
                    "gate3_accept_rate_at_gamma": float(np.mean(keep[mask_g])) if np.any(mask_g) else 0.0,
                    "gate3_mean_purity_at_gamma": float(np.mean(purity[mask_g])) if np.any(mask_g) else 0.0,
                    "gate3_mean_intrusion_at_gamma": float(np.mean(intrusion[mask_g])) if np.any(mask_g) else 0.0,
                    "n_candidates_at_gamma": int(np.sum(mask_g)),
                }

        return keep.astype(bool), {
            "gate3_accept_count": int(np.sum(keep)),
            "gate3_reject_count": int(np.sum(~keep)),
            "gate3_accept_rate": float(np.mean(keep)),
            "gate3_anchor_n": int(len(self._y_anchor)),
            "gate3_k": int(self._k_eff),
            "gate3_tau_purity": float(self.cfg.tau_purity),
            "gate3_query_mode": "batch",
            "gate3_knn_algorithm": str(self.cfg.algorithm),
            "gate3_query_batch_size": int(batch_size),
            "gate3_runtime_sec": float(runtime_sec),
            "gate3_reject_reason_summary": {
                "majority_mismatch_only": int(majority_only),
                "purity_below_tau_only": int(purity_only),
                "both": int(both),
            },
            "gamma_stats": gamma_stats,
            "candidate_rows": reject_rows,
            "candidate_diag_arrays": {
                "direction_ids": dir_arr.tolist(),
                "gamma_used": gamma_arr.tolist(),
                "source_label": y.tolist(),
                "knn_majority_label": majority.tolist(),
                "knn_same_class_purity": purity.tolist(),
                "knn_intrusion_ratio": intrusion.tolist(),
                "mean_distance_same_class": mean_same.tolist(),
                "mean_distance_other_class": mean_other.tolist(),
            },
        }
