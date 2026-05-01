from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .curriculum import estimate_local_manifold_margins
from .pia import build_zpia_direction_bank


@dataclass(frozen=True)
class PIAOperatorConfig:
    """Configuration for the CSTA-internal PIA operator facade.

    WARNING: This module acts as a policy interface and metadata facade for 
    the paper's "Four-Step Operator" narrative. It is NOT a full production-level
    replacement for the low-level generation engine.

    Known deviations from production path:
    1. Single-sample per anchor (Production supports dual-sign slots);
    2. No bridge realization logic (Uses simple additive shifts);
    3. No group-consensus functional implementation (Implemented as metadata tag only).
    """

    k_dir: int = 10
    gamma: float = 0.1
    eta_safe: float = 0.5
    activation_policy: str = "top1"
    activation_topk: int = 1
    activation_tau: Optional[float] = None
    seed: int = 42
    telm2_n_iters: int = 3
    telm2_c_repr: float = 1.0
    telm2_activation: str = "sine"
    telm2_bias_update_mode: str = "residual"


def normalize_activation_policy(policy: str) -> Dict[str, object]:
    """Return canonical PIA activation metadata for a public policy string."""

    name = str(policy or "top1")
    if name in {"top1", "top_response", "anchor_top_response"}:
        return {"activation_policy": "top1", "activation_scope": "single_anchor", "activation_topk": 1, "activation_tau": np.nan}
    if name.startswith("topk_uniform_top"):
        topk = int(name.split("top")[-1])
        return {
            "activation_policy": "uniform_topk",
            "activation_scope": "single_anchor",
            "activation_topk": topk,
            "activation_tau": np.nan,
        }
    if name.startswith("topk_softmax_tau_"):
        tau = float(name.split("_")[-1])
        return {
            "activation_policy": "softmax_topk",
            "activation_scope": "single_anchor",
            "activation_topk": 5,
            "activation_tau": tau,
        }
    if name == "group_top":
        return {
            "activation_policy": "group_top",
            "activation_scope": "neighborhood_consensus",
            "activation_topk": 1,
            "activation_tau": np.nan,
        }
    return {
        "activation_policy": name,
        "activation_scope": "single_anchor",
        "activation_topk": np.nan,
        "activation_tau": np.nan,
    }


def pia_operator_metadata(policy: str, *, dictionary_estimator: str = "TELM2") -> Dict[str, object]:
    meta = normalize_activation_policy(policy)
    meta.update(
        {
            "operator_name": "PIA",
            "dictionary_estimator": dictionary_estimator,
            "safe_generator": "local_margin_safe_step",
            "bridge_realizer": "whitening_coloring",
        }
    )
    return meta


def estimate_template_dictionary(
    Z_train: np.ndarray,
    *,
    cfg: PIAOperatorConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Estimate the train-only TELM2 tangent dictionary used by PIA."""

    bank, meta = build_zpia_direction_bank(
        Z_train,
        k_dir=int(cfg.k_dir),
        seed=int(cfg.seed),
        telm2_n_iters=int(cfg.telm2_n_iters),
        telm2_c_repr=float(cfg.telm2_c_repr),
        telm2_activation=str(cfg.telm2_activation),
        telm2_bias_update_mode=str(cfg.telm2_bias_update_mode),
    )
    meta = dict(meta)
    meta.update(pia_operator_metadata(cfg.activation_policy))
    return bank, meta


def activate_templates(
    z_anchor: np.ndarray,
    dictionary: np.ndarray,
    *,
    policy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select template ids according to the canonical PIA activation policy."""

    D = np.asarray(dictionary, dtype=np.float64)
    z = np.asarray(z_anchor, dtype=np.float64).ravel()
    if D.ndim != 2 or D.shape[0] <= 0:
        raise ValueError("PIA dictionary must be a non-empty 2D array.")
    meta = normalize_activation_policy(policy)
    responses = np.abs(z @ D.T)
    order = np.lexsort((np.arange(D.shape[0]), -responses))
    topk = int(meta["activation_topk"]) if np.isfinite(meta["activation_topk"]) else 1
    topk = max(1, min(topk, D.shape[0]))
    top_ids = order[:topk]
    if meta["activation_policy"] == "uniform_topk":
        return np.asarray([int(rng.choice(top_ids))], dtype=np.int64)
    if meta["activation_policy"] == "softmax_topk":
        tau = max(float(meta["activation_tau"]), 1e-12)
        logits = responses[top_ids] / tau
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        probs = weights / np.sum(weights)
        return np.asarray([int(rng.choice(top_ids, p=probs))], dtype=np.int64)
    return np.asarray([int(top_ids[0])], dtype=np.int64)


def generate_safe_vicinal_states(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    dictionary: np.ndarray,
    *,
    cfg: PIAOperatorConfig,
) -> Tuple[np.ndarray, np.ndarray, Iterable[Dict[str, object]], Dict[str, object]]:
    """Generate one safe PIA state per anchor using the existing safe-step rule."""

    Z = np.asarray(Z_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64).ravel()
    D = np.asarray(dictionary, dtype=np.float64)
    margins = estimate_local_manifold_margins(Z, y)
    rng = np.random.default_rng(int(cfg.seed))
    z_aug = []
    rows = []
    for idx, z in enumerate(Z):
        template_id = int(activate_templates(z, D, policy=cfg.activation_policy, rng=rng)[0])
        direction = D[template_id]
        direction_norm = float(np.linalg.norm(direction))
        safe_upper = float(cfg.eta_safe) * float(margins[idx]) / (direction_norm + 1e-12)
        gamma_used = min(float(cfg.gamma), safe_upper)
        delta = gamma_used * direction
        z_aug.append(z + delta)
        rows.append(
            {
                "anchor_index": int(idx),
                "label": int(y[idx]),
                "template_id": template_id,
                "gamma_requested": float(cfg.gamma),
                "gamma_used": float(gamma_used),
                "safe_radius_ratio": float(abs(gamma_used) * direction_norm / (float(cfg.eta_safe) * float(margins[idx]) + 1e-12)),
                "is_clipped": float(float(cfg.gamma) > safe_upper + 1e-9),
            }
        )
    meta = pia_operator_metadata(cfg.activation_policy)
    return np.asarray(z_aug, dtype=np.float32), y.copy(), rows, meta


def run_pia_operator(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    *,
    cfg: PIAOperatorConfig,
) -> Dict[str, object]:
    """Convenience wrapper for the paper-facing PIA operator contract."""

    dictionary, dictionary_meta = estimate_template_dictionary(Z_train, cfg=cfg)
    z_aug, y_aug, audit_rows, operator_meta = generate_safe_vicinal_states(
        Z_train,
        y_train,
        dictionary,
        cfg=cfg,
    )
    operator_meta.update(dictionary_meta)
    return {
        "z_aug": z_aug,
        "y_aug": y_aug,
        "audit_rows": list(audit_rows),
        "operator_meta": operator_meta,
        "dictionary": dictionary,
    }
