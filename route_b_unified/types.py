from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from scripts.run_raw_bridge_probe import TrialRecord


@dataclass
class RepresentationState:
    dataset: str
    seed: int
    split_meta: Dict[str, object]
    mean_log_train: np.ndarray
    train_records: List[TrialRecord]
    val_records: List[TrialRecord]
    test_records: List[TrialRecord]
    train_trial_dicts: List[Dict[str, object]]
    val_trial_dicts: List[Dict[str, object]]
    test_trial_dicts: List[Dict[str, object]]
    X_train: np.ndarray
    y_train: np.ndarray
    tid_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    tid_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    tid_test: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class PolicyAction:
    round_index: int
    selected_dir_ids: List[int]
    direction_weights: Dict[int, float]
    step_sizes: Dict[int, float]
    direction_probs_vector: np.ndarray
    gamma_vector: np.ndarray
    entropy: float
    stop_flag: bool
    stop_reason: str


@dataclass
class PolicyState:
    variant: str
    direction_bank: np.ndarray
    base_scores: np.ndarray
    current_scores: np.ndarray
    gamma_by_dir: np.ndarray
    prior_frozen_mask: np.ndarray
    bank_meta: Dict[str, object]
    solver_rows: List[Dict[str, object]]
    best_val_f1: float = float("-inf")
    best_round_index: int = 0
    stale_rounds: int = 0
    rank_history: List[List[int]] = field(default_factory=list)
    stop_flag: bool = False
    stop_reason: str = ""


@dataclass
class TargetRoundState:
    round_index: int
    z_aug: np.ndarray
    y_aug: np.ndarray
    tid_aug: np.ndarray
    mech: Dict[str, object]
    dir_maps: Dict[str, Dict[int, float]]
    aug_meta: Dict[str, object]
    action: PolicyAction
    direction_state: Dict[int, str]
    direction_budget_score: Dict[int, float]


@dataclass
class BridgeResult:
    dataset: str
    seed: int
    variant: str
    round_index: int
    train_trials: List[Dict[str, object]]
    val_trials: List[Dict[str, object]]
    test_trials: List[Dict[str, object]]
    global_fidelity: Dict[str, object]
    classwise_fidelity: Dict[str, object]
    margin_proxy: Dict[str, object]
    task_risk_comment: str
    orig_train_trials: List[Dict[str, object]] = field(default_factory=list)
    aug_train_trials: List[Dict[str, object]] = field(default_factory=list)
    per_aug_bridge_meta: List[Dict[str, object]] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class EvaluatorPosterior:
    dataset: str
    seed: int
    variant: str
    round_index: int
    split_name: str
    acc: float
    macro_f1: float
    metrics: Dict[str, object]
    round_gain_proxy: float
    direction_usage_entropy: float
    worst_dir_id: int | None
    worst_dir_summary: str
    direction_metrics: Dict[str, Dict[int, float]]
    selected_dir_ids: List[int]
    selected_dir_weights: Dict[int, float]
    classwise_distortion_summary: Dict[str, float]
    inter_class_margin_proxy: Dict[str, float]
    task_risk_comment: str
    bridge_meta: Dict[str, object]


@dataclass
class PolicyUpdateSummary:
    dataset: str
    seed: int
    variant: str
    round_index: int
    rewarded_dirs: List[int]
    penalized_dirs: List[int]
    rank_change_summary: Dict[str, int]
    collapse_warning: str
    overspread_warning: str
    reward_summary: Dict[str, float]
    penalty_summary: Dict[str, float]
