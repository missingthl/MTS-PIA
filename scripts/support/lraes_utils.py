from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from scripts.support.fisher_pia_utils import FisherPIAConfig, compute_fisher_pia_terms


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0:
        return {"min": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "min": float(np.min(xx)),
        "mean": float(np.mean(xx)),
        "std": float(np.std(xx)),
        "max": float(np.max(xx)),
    }


def _canonicalize_axis(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel().copy()
    nrm = float(np.linalg.norm(x))
    if not np.isfinite(nrm) or nrm <= 1e-12:
        out = np.zeros_like(x, dtype=np.float64)
        if out.size:
            out[0] = 1.0
        return out
    x /= nrm
    idx = int(np.argmax(np.abs(x)))
    if x[idx] < 0.0:
        x *= -1.0
    return x


def _stats_string(x: np.ndarray, *, fmt: str = ".4f") -> str:
    s = _summary_stats(np.asarray(x, dtype=np.float64))
    return (
        f"min={format(float(s['min']), fmt)}|"
        f"mean={format(float(s['mean']), fmt)}|"
        f"std={format(float(s['std']), fmt)}|"
        f"max={format(float(s['max']), fmt)}"
    )


@dataclass(frozen=True)
class LRAESConfig:
    beta: float = 0.5
    reg_lambda: float = 1e-4
    top_k_per_class: int = 3
    rank_tol: float = 1e-8
    eig_pos_eps: float = 1e-9


def build_lraes_direction_bank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k_dir: int,
    fisher_cfg: FisherPIAConfig,
    lraes_cfg: LRAESConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train).astype(int).ravel()
    if X.ndim != 2:
        raise ValueError("X_train must be 2D for LRAES.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_train / y_train size mismatch for LRAES.")

    class_terms, terms_meta = compute_fisher_pia_terms(X, y, cfg=fisher_cfg)
    d = int(X.shape[1])
    I = np.eye(d, dtype=np.float64)
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    total_cov = (X_centered.T @ X_centered) / max(1, int(X.shape[0]))

    candidates: List[Dict[str, object]] = []
    solver_rows: List[Dict[str, object]] = []
    top_k_local = int(max(1, min(int(lraes_cfg.top_k_per_class), d)))
    eig_eps = float(lraes_cfg.eig_pos_eps)

    for cls in sorted(class_terms.keys()):
        t = class_terms[int(cls)]
        S_expand = np.asarray(t["S_expand"], dtype=np.float64)
        S_risk = np.asarray(t["S_risk"], dtype=np.float64)
        S_expand_reg = 0.5 * (S_expand + S_expand.T) + float(lraes_cfg.reg_lambda) * I
        S_risk_reg = 0.5 * (S_risk + S_risk.T) + float(lraes_cfg.reg_lambda) * I
        M = S_expand_reg - float(lraes_cfg.beta) * S_risk_reg
        M = 0.5 * (M + M.T)

        eigvals, eigvecs = np.linalg.eigh(M)
        order = np.argsort(eigvals)[::-1]
        eigvals_desc = eigvals[order]
        top_eigs = eigvals_desc[:top_k_local]
        top1 = float(top_eigs[0]) if top_eigs.size else 0.0
        pos_count = int(np.sum(top_eigs > eig_eps))
        nonpos_count = int(top_eigs.size - pos_count)
        max_pos = bool(top1 > eig_eps)
        if top1 <= eig_eps:
            solver_state = "fully_risk_dominated"
        elif nonpos_count > 0:
            solver_state = "marginal"
        else:
            solver_state = "safe_expandable"

        matrix_rank = int(np.linalg.matrix_rank(M, tol=float(lraes_cfg.rank_tol)))
        selected_axis_var: List[float] = []
        for j in range(min(top_k_local, eigvecs.shape[1])):
            eig_idx = int(order[j])
            eigval = float(eigvals[eig_idx])
            axis = _canonicalize_axis(eigvecs[:, eig_idx])
            axis_var = float(axis @ total_cov @ axis)
            selected_axis_var.append(axis_var)
            candidates.append(
                {
                    "class_id": int(cls),
                    "axis_rank_in_class": int(j + 1),
                    "eigval": eigval,
                    "axis": axis,
                    "axis_variance": axis_var,
                    "class_weight": float(t["class_weight"]),
                    "candidate_score": float(float(t["class_weight"]) * eigval),
                    "solver_state": solver_state,
                }
            )

        solver_rows.append(
            {
                "class_id": int(cls),
                "beta": float(lraes_cfg.beta),
                "matrix_rank": matrix_rank,
                "top1_eigenvalue": top1,
                "topk_eigenvalues": "|".join(format(float(v), ".6f") for v in top_eigs.tolist()),
                "topk_positive_count": pos_count,
                "topk_nonpositive_count": nonpos_count,
                "max_eigenvalue_is_positive": max_pos,
                "solver_state": solver_state,
                "selected_axis_variance_summary": _stats_string(np.asarray(selected_axis_var, dtype=np.float64)),
                "low_quality_axis_count": nonpos_count,
                "class_weight": float(t["class_weight"]),
            }
        )

    if not candidates:
        raise RuntimeError("LRAES failed to generate any candidate direction.")

    candidates_sorted = sorted(candidates, key=lambda row: float(row["candidate_score"]), reverse=True)
    selected = candidates_sorted[: int(max(1, min(int(k_dir), len(candidates_sorted))))]
    bank = np.vstack([np.asarray(row["axis"], dtype=np.float64) for row in selected]).astype(np.float32)
    eigvals_sel = np.asarray([float(row["eigval"]) for row in selected], dtype=np.float64)
    axis_var_sel = np.asarray([float(row["axis_variance"]) for row in selected], dtype=np.float64)
    prior_frozen_mask = eigvals_sel <= eig_eps
    if np.all(prior_frozen_mask) and prior_frozen_mask.size:
        prior_frozen_mask[int(np.argmax(eigvals_sel))] = False

    rank_arr = np.asarray([int(row["matrix_rank"]) for row in solver_rows], dtype=np.float64)
    top1_arr = np.asarray([float(row["top1_eigenvalue"]) for row in solver_rows], dtype=np.float64)
    pos_arr = np.asarray([int(row["topk_positive_count"]) for row in solver_rows], dtype=np.float64)
    nonpos_arr = np.asarray([int(row["topk_nonpositive_count"]) for row in solver_rows], dtype=np.float64)

    bank_meta: Dict[str, object] = {
        "bank_source": "lraes_local_risk_aware_eigensolver",
        "k_dir": int(bank.shape[0]),
        "beta": float(lraes_cfg.beta),
        "reg_lambda": float(lraes_cfg.reg_lambda),
        "top_k_per_class": int(top_k_local),
        "rank_tol": float(lraes_cfg.rank_tol),
        "eig_pos_eps": float(lraes_cfg.eig_pos_eps),
        "terms_meta": terms_meta,
        "selected_class_ids": [int(row["class_id"]) for row in selected],
        "selected_axis_rank_in_class": [int(row["axis_rank_in_class"]) for row in selected],
        "selected_eigenvalues": eigvals_sel.tolist(),
        "selected_axis_variances": axis_var_sel.tolist(),
        "local_matrix_rank_summary": _stats_string(rank_arr, fmt=".2f"),
        "top1_eigenvalue_summary": _stats_string(top1_arr, fmt=".6f"),
        "topk_positive_count_summary": _stats_string(pos_arr, fmt=".2f"),
        "topk_nonpositive_count_summary": _stats_string(nonpos_arr, fmt=".2f"),
        "selected_axis_variance_summary": _stats_string(axis_var_sel, fmt=".6f"),
        "low_quality_axis_count": int(np.sum(eigvals_sel <= eig_eps)),
        "solver_state_counts": {
            key: int(sum(1 for row in solver_rows if row["solver_state"] == key))
            for key in ["safe_expandable", "marginal", "fully_risk_dominated"]
        },
    }
    return bank, np.asarray(prior_frozen_mask, dtype=bool), bank_meta, solver_rows
