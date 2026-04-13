from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from datasets.trial_dataset_factory import (
    DEFAULT_NATOPS_ROOT,
    DEFAULT_SEEDIV_ROOT,
    DEFAULT_SEEDV_ROOT,
    DEFAULT_SELFREGULATIONSCP1_ROOT,
    load_trials_for_dataset,
)
from route_b_unified.trial_records import TrialRecord, _apply_mean_log, _build_trial_records
from route_b_unified.types import RepresentationState
from scripts.support.protocol_split_utils import resolve_inner_train_val_split, resolve_protocol_split


@dataclass(frozen=True)
class RepresentationConfig:
    dataset: str
    seed: int
    val_fraction: float = 0.25
    spd_eps: float = 1e-4
    processed_root: str = "data/SEED/SEED_EEG/Preprocessed_EEG"
    stim_xlsx: str = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
    seediv_root: str = DEFAULT_SEEDIV_ROOT
    seedv_root: str = DEFAULT_SEEDV_ROOT
    natops_root: str = DEFAULT_NATOPS_ROOT
    selfregulationscp1_root: str = DEFAULT_SELFREGULATIONSCP1_ROOT


def _record_to_trial_dict(rec: TrialRecord) -> Dict[str, object]:
    return {
        "trial_id_str": str(rec.tid),
        "label": int(rec.y),
        "x_trial": np.asarray(rec.x_raw, dtype=np.float32),
    }


def _records_to_trial_dicts(records: List[TrialRecord]) -> List[Dict[str, object]]:
    return [_record_to_trial_dict(r) for r in records]


def _stack_feature(records: List[TrialRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.asarray([], dtype=object),
        )
    return (
        np.stack([np.asarray(r.z, dtype=np.float32) for r in records], axis=0),
        np.asarray([int(r.y) for r in records], dtype=np.int64),
        np.asarray([r.tid for r in records], dtype=object),
    )


def build_representation(cfg: RepresentationConfig) -> RepresentationState:
    all_trials = load_trials_for_dataset(
        dataset=str(cfg.dataset).strip().lower(),
        processed_root=cfg.processed_root,
        stim_xlsx=cfg.stim_xlsx,
        seediv_root=cfg.seediv_root,
        seedv_root=cfg.seedv_root,
        natops_root=cfg.natops_root,
        selfregulationscp1_root=cfg.selfregulationscp1_root,
    )
    outer_train_trials, outer_test_trials, split_meta = resolve_protocol_split(
        dataset=str(cfg.dataset).strip().lower(),
        all_trials=list(all_trials),
        seed=int(cfg.seed) + 1701,
        allow_random_fallback=False,
    )
    inner_train_trials, inner_val_trials, inner_meta = resolve_inner_train_val_split(
        train_trials=outer_train_trials,
        seed=int(cfg.seed) + 1701,
        val_fraction=float(cfg.val_fraction),
    )
    train_tmp, mean_log_train = _build_trial_records(inner_train_trials, float(cfg.spd_eps))
    val_tmp, _ = _build_trial_records(inner_val_trials, float(cfg.spd_eps))
    test_tmp, _ = _build_trial_records(outer_test_trials, float(cfg.spd_eps))

    train_records = _apply_mean_log(train_tmp, mean_log_train)
    val_records = _apply_mean_log(val_tmp, mean_log_train)
    test_records = _apply_mean_log(test_tmp, mean_log_train)

    X_train, y_train, tid_train = _stack_feature(train_records)
    X_val, y_val, tid_val = _stack_feature(val_records)
    X_test, y_test, tid_test = _stack_feature(test_records)

    return RepresentationState(
        dataset=str(cfg.dataset).strip().lower(),
        seed=int(cfg.seed),
        split_meta=dict(split_meta),
        mean_log_train=np.asarray(mean_log_train, dtype=np.float32),
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        train_trial_dicts=_records_to_trial_dicts(train_records),
        val_trial_dicts=_records_to_trial_dicts(val_records),
        test_trial_dicts=_records_to_trial_dicts(test_records),
        X_train=X_train,
        y_train=y_train,
        tid_train=tid_train,
        X_val=X_val,
        y_val=y_val,
        tid_val=tid_val,
        X_test=X_test,
        y_test=y_test,
        tid_test=tid_test,
        meta={
            "protocol_type": str(split_meta.get("protocol_type", "")),
            "protocol_note": str(split_meta.get("protocol_note", "")),
            "outer_train_trial_count": int(len(outer_train_trials)),
            "inner_train_trial_count": int(len(inner_train_trials)),
            "inner_val_trial_count": int(len(inner_val_trials)),
            "outer_test_trial_count": int(len(outer_test_trials)),
            "val_fraction": float(cfg.val_fraction),
            **dict(inner_meta),
        },
    )
