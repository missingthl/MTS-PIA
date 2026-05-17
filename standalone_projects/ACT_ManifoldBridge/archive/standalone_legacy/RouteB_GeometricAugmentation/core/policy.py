from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from core.curriculum_utils import _build_direction_bank_d1, _update_direction_budget
from core.types import EvaluatorPosterior, PolicyAction, PolicyState, PolicyUpdateSummary
from utils.fisher_pia_utils import FisherPIAConfig
from utils.lraes_utils import LRAESConfig, build_lraes_direction_bank


@dataclass(frozen=True)
class UnifiedPolicyConfig:
    variant: str
    n_rounds: int = 3
    k_dir: int = 5
    select_top_k: int = 3
    gamma_init: float = 0.06
    expand_factor: float = 1.25
    shrink_factor: float = 0.70
    gamma_max: float = 0.16
    freeze_eps: float = 0.02
    stop_patience: int = 2
    feedback_enabled: bool = False
    feedback_eta_reward: float = 0.75
    feedback_eta_penalty: float = 1.00
    beta: float = 0.5
    reg_lambda: float = 1e-4
    top_k_per_class: int = 3
    rank_tol: float = 1e-8
    eig_pos_eps: float = 1e-9
    lraes_knn_k: int = 20
    lraes_boundary_quantile: float = 0.30
    lraes_interior_quantile: float = 0.70
    lraes_hetero_k: int = 3
    pia_n_iters: int = 2
    pia_activation: str = "sine"
    pia_bias_update_mode: str = "residual"
    pia_c_repr: float = 1.0


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64).ravel().copy()
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.ones_like(x, dtype=np.float64)
    xmin = float(np.min(x[finite]))
    xmax = float(np.max(x[finite]))
    if abs(xmax - xmin) <= 1e-12:
        return np.ones_like(x, dtype=np.float64)
    out = (x - xmin) / (xmax - xmin)
    return out + 1e-6


def _softmax_like(weights: np.ndarray) -> np.ndarray:
    x = np.asarray(weights, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    if not np.any(np.isfinite(x)):
        return np.full_like(x, 1.0 / float(max(1, x.size)))
    x = x - np.max(x)
    ex = np.exp(x)
    total = float(np.sum(ex))
    if total <= 0:
        return np.full_like(x, 1.0 / float(max(1, x.size)))
    return ex / total


def _entropy(probs: np.ndarray) -> float:
    x = np.asarray(probs, dtype=np.float64).ravel()
    x = x[x > 0]
    if x.size == 0:
        return 0.0
    return float(-np.sum(x * np.log(x)))


def _rank_map(scores: np.ndarray, active_mask: np.ndarray) -> Dict[str, int]:
    idx = np.where(np.asarray(active_mask, dtype=bool))[0]
    if idx.size == 0:
        return {}
    order = idx[np.argsort(np.asarray(scores, dtype=np.float64)[idx])[::-1]]
    return {str(int(i)): int(rank + 1) for rank, i in enumerate(order.tolist())}


def init_policy(rep_state, policy_cfg: UnifiedPolicyConfig) -> PolicyState:
    if policy_cfg.variant == "legacy_multiround":
        direction_bank, _ = _build_direction_bank_d1(
            X_train=rep_state.X_train,
            k_dir=int(policy_cfg.k_dir),
            seed=int(rep_state.seed * 10000 + int(policy_cfg.k_dir) * 113 + 17),
            n_iters=int(policy_cfg.pia_n_iters),
            activation=str(policy_cfg.pia_activation),
            bias_update_mode=str(policy_cfg.pia_bias_update_mode),
            c_repr=float(policy_cfg.pia_c_repr),
        )
        base_scores = np.ones((direction_bank.shape[0],), dtype=np.float64)
        prior_frozen_mask = np.zeros((direction_bank.shape[0],), dtype=bool)
        bank_meta = {"bank_source": "legacy_telm2_d1", "selected_eigenvalues": base_scores.tolist()}
        solver_rows = []
    else:
        fisher_cfg = FisherPIAConfig(
            knn_k=int(policy_cfg.lraes_knn_k),
            interior_quantile=float(policy_cfg.lraes_interior_quantile),
            boundary_quantile=float(policy_cfg.lraes_boundary_quantile),
            hetero_k=int(policy_cfg.lraes_hetero_k),
        )
        lraes_cfg = LRAESConfig(
            beta=float(policy_cfg.beta),
            reg_lambda=float(policy_cfg.reg_lambda),
            top_k_per_class=int(policy_cfg.top_k_per_class),
            rank_tol=float(policy_cfg.rank_tol),
            eig_pos_eps=float(policy_cfg.eig_pos_eps),
        )
        direction_bank, prior_frozen_mask, bank_meta, solver_rows = build_lraes_direction_bank(
            rep_state.X_train,
            rep_state.y_train,
            k_dir=int(policy_cfg.k_dir),
            fisher_cfg=fisher_cfg,
            lraes_cfg=lraes_cfg,
        )
        base_scores = _normalize_scores(np.asarray(bank_meta.get("selected_eigenvalues", []), dtype=np.float64))
    gamma_by_dir = np.full((direction_bank.shape[0],), float(policy_cfg.gamma_init), dtype=np.float64)
    gamma_by_dir[np.asarray(prior_frozen_mask, dtype=bool)] = 0.0
    state = PolicyState(
        variant=str(policy_cfg.variant),
        direction_bank=np.asarray(direction_bank, dtype=np.float32),
        base_scores=np.asarray(base_scores, dtype=np.float64),
        current_scores=np.asarray(base_scores, dtype=np.float64).copy(),
        gamma_by_dir=gamma_by_dir,
        prior_frozen_mask=np.asarray(prior_frozen_mask, dtype=bool),
        bank_meta=dict(bank_meta),
        solver_rows=list(solver_rows),
        rank_history=[list(np.argsort(np.asarray(base_scores, dtype=np.float64))[::-1].tolist())],
    )
    return state


def policy_step(rep_state, policy_state: PolicyState, posterior_prev: EvaluatorPosterior | None, policy_cfg: UnifiedPolicyConfig) -> Tuple[PolicyAction, PolicyState]:
    active_mask = (~np.asarray(policy_state.prior_frozen_mask, dtype=bool)) & (
        np.asarray(policy_state.gamma_by_dir, dtype=np.float64) > float(policy_cfg.freeze_eps)
    )
    if not np.any(active_mask):
        policy_state.stop_flag = True
        policy_state.stop_reason = "all_dirs_frozen"
        return (
            PolicyAction(
                round_index=int(len(policy_state.rank_history)),
                selected_dir_ids=[],
                direction_weights={},
                step_sizes={},
                direction_probs_vector=np.zeros_like(policy_state.gamma_by_dir, dtype=np.float64),
                gamma_vector=np.zeros_like(policy_state.gamma_by_dir, dtype=np.float64),
                entropy=0.0,
                stop_flag=True,
                stop_reason="all_dirs_frozen",
            ),
            policy_state,
        )
    scores = np.asarray(policy_state.current_scores, dtype=np.float64).copy()
    scores[~active_mask] = -np.inf
    active_idx = np.where(active_mask)[0]
    order = active_idx[np.argsort(scores[active_idx])[::-1]]
    select_k = int(max(1, min(int(policy_cfg.select_top_k), int(order.size))))
    selected = order[:select_k]
    selected_scores = np.asarray(policy_state.current_scores, dtype=np.float64)[selected]
    selected_probs = _softmax_like(selected_scores)
    probs = np.zeros_like(policy_state.gamma_by_dir, dtype=np.float64)
    probs[selected] = selected_probs
    gamma_vec = np.asarray(policy_state.gamma_by_dir, dtype=np.float64).copy()
    mask_unselected = np.ones_like(gamma_vec, dtype=bool)
    mask_unselected[selected] = False
    gamma_vec[mask_unselected] = 0.0
    action = PolicyAction(
        round_index=int(len(policy_state.rank_history)),
        selected_dir_ids=[int(v) for v in selected.tolist()],
        direction_weights={int(i): float(probs[int(i)]) for i in selected.tolist()},
        step_sizes={int(i): float(gamma_vec[int(i)]) for i in selected.tolist()},
        direction_probs_vector=probs,
        gamma_vector=gamma_vec,
        entropy=float(_entropy(selected_probs)),
        stop_flag=False,
        stop_reason="active_budget_available",
    )
    return action, policy_state


def apply_target_feedback(
    policy_state: PolicyState,
    *,
    margin_by_dir: Dict[int, float],
    flip_by_dir: Dict[int, float],
    intrusion_by_dir: Dict[int, float],
    policy_cfg: UnifiedPolicyConfig,
) -> Tuple[PolicyState, Dict[int, str], Dict[int, float]]:
    gamma_after, state_by_dir, score_by_dir = _update_direction_budget(
        gamma_before=np.asarray(policy_state.gamma_by_dir, dtype=np.float64),
        margin_by_dir=margin_by_dir,
        flip_by_dir=flip_by_dir,
        intrusion_by_dir=intrusion_by_dir,
        expand_factor=float(policy_cfg.expand_factor),
        shrink_factor=float(policy_cfg.shrink_factor),
        gamma_max=float(policy_cfg.gamma_max),
        freeze_eps=float(policy_cfg.freeze_eps),
    )
    policy_state.gamma_by_dir = np.asarray(gamma_after, dtype=np.float64)
    return policy_state, state_by_dir, score_by_dir


def update_policy(policy_state: PolicyState, posterior_t: EvaluatorPosterior, policy_cfg: UnifiedPolicyConfig) -> Tuple[PolicyState, PolicyUpdateSummary]:
    scores_before = np.asarray(policy_state.current_scores, dtype=np.float64).copy()
    active_mask = ~np.asarray(policy_state.prior_frozen_mask, dtype=bool)
    old_rank = _rank_map(scores_before, active_mask)

    margin_by_dir = posterior_t.direction_metrics.get("margin_drop_median", {})
    flip_by_dir = posterior_t.direction_metrics.get("flip_rate", {})
    intrusion_by_dir = posterior_t.direction_metrics.get("intrusion", {})
    selected = set(int(v) for v in posterior_t.selected_dir_ids)
    round_gain = float(posterior_t.round_gain_proxy)
    classwise_penalty = float(
        np.mean(list(posterior_t.classwise_distortion_summary.values()))
        if posterior_t.classwise_distortion_summary
        else 0.0
    )
    margin_delta = float(posterior_t.inter_class_margin_proxy.get("delta_mean", 0.0))
    entropy = float(posterior_t.direction_usage_entropy)

    delta = np.zeros_like(scores_before, dtype=np.float64)
    reward_values: Dict[str, float] = {}
    penalty_values: Dict[str, float] = {}
    for dir_id in range(len(scores_before)):
        margin = float(margin_by_dir.get(int(dir_id), 0.0))
        flip = float(flip_by_dir.get(int(dir_id), 0.0))
        intr = float(intrusion_by_dir.get(int(dir_id), 0.0))
        sel_weight = float(posterior_t.selected_dir_weights.get(int(dir_id), 0.0))
        reward = max(0.0, margin)
        if round_gain > 0.0 and int(dir_id) in selected:
            reward += float(round_gain) * max(0.1, sel_weight)
        penalty = max(0.0, -margin) + flip + intr
        if posterior_t.worst_dir_id is not None and int(dir_id) == int(posterior_t.worst_dir_id):
            penalty += 0.25
        if int(dir_id) in selected:
            penalty += 0.25 * max(0.0, classwise_penalty)
            if round_gain < 0.0:
                penalty += abs(float(round_gain)) * max(0.1, sel_weight)
        if margin_delta < 0.0:
            penalty += 0.25 * abs(margin_delta)
        if entropy < max(0.2, math.log(max(1, len(selected))) * 0.35) and int(dir_id) in selected:
            penalty += 0.10
        reward_values[str(dir_id)] = float(reward)
        penalty_values[str(dir_id)] = float(penalty)
        delta[dir_id] = float(policy_cfg.feedback_eta_reward) * reward - float(policy_cfg.feedback_eta_penalty) * penalty

    scores_after = scores_before + delta
    scores_after = np.where(active_mask, scores_after, -1e9)
    if np.all(scores_after[active_mask] <= 1e-9):
        best_id = int(np.argmax(np.asarray(policy_state.base_scores, dtype=np.float64)))
        scores_after[best_id] = 1.0
    policy_state.current_scores = scores_after

    if float(posterior_t.macro_f1) > float(policy_state.best_val_f1) + 1e-9:
        policy_state.best_val_f1 = float(posterior_t.macro_f1)
        policy_state.best_round_index = int(posterior_t.round_index)
        policy_state.stale_rounds = 0
    else:
        policy_state.stale_rounds = int(policy_state.stale_rounds) + 1
    if int(policy_state.stale_rounds) >= int(policy_cfg.stop_patience):
        policy_state.stop_flag = True
        policy_state.stop_reason = "no_val_improvement_patience"

    new_rank = _rank_map(scores_after, active_mask)
    policy_state.rank_history.append([int(k) for k, _ in sorted(new_rank.items(), key=lambda kv: kv[1])])
    rank_change_summary = {}
    for key in sorted(set(old_rank.keys()) | set(new_rank.keys()), key=int):
        rank_change_summary[str(key)] = int(new_rank.get(key, 999) - old_rank.get(key, 999))

    rewarded = [int(i) for i, v in enumerate(delta.tolist()) if v > 1e-6]
    penalized = [int(i) for i, v in enumerate(delta.tolist()) if v < -1e-6]
    collapse_warning = "direction_collapse_risk" if entropy < 0.25 else "none"
    overspread_warning = "overspread_without_gain" if entropy > 1.0 and round_gain <= 0.0 else "none"
    summary = PolicyUpdateSummary(
        dataset=str(posterior_t.dataset),
        seed=int(posterior_t.seed),
        variant=str(posterior_t.variant),
        round_index=int(posterior_t.round_index),
        rewarded_dirs=rewarded,
        penalized_dirs=penalized,
        rank_change_summary=rank_change_summary,
        collapse_warning=collapse_warning,
        overspread_warning=overspread_warning,
        reward_summary=reward_values,
        penalty_summary=penalty_values,
    )
    return policy_state, summary
