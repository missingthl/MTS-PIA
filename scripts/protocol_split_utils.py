from __future__ import annotations

import json
import hashlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
from datasets.trial_dataset_factory import normalize_dataset_name
from sklearn.model_selection import StratifiedShuffleSplit


OFFICIAL_FIXED_SPLIT_DATASETS = {
    "har",
    "natops",
    "fingermovements",
    "selfregulationscp1",
    "basicmotions",
    "handmovementdirection",
    "uwavegesturelibrary",
    "epilepsy",
    "atrialfibrillation",
    "pendigits",
}
MITBIH_DATASETS = {"mitbih"}
SEED_NATIVE_DATASETS = {"seed1", "seediv", "seedv"}


def _split_hash(train_ids: Sequence[str], test_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {"train": [str(x) for x in train_ids], "test": [str(x) for x in test_ids]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_by_split(all_trials: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    train = [t for t in all_trials if str(t.get("split", "")).lower() == "train"]
    test = [t for t in all_trials if str(t.get("split", "")).lower() == "test"]
    if not train or not test:
        raise RuntimeError("Expected both split=train and split=test in dataset-provided protocol.")
    return train, test


def _make_trial_split_local(all_trials: List[Dict], seed: int) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_trials))
    n_train = int(0.8 * len(all_trials))
    train_trials = [all_trials[i] for i in idx[:n_train]]
    test_trials = [all_trials[i] for i in idx[n_train:]]
    train_ids = [str(t["trial_id_str"]) for t in train_trials]
    test_ids = [str(t["trial_id_str"]) for t in test_trials]
    return train_trials, test_trials, {
        "split_hash": _split_hash(train_ids, test_ids),
        "train_count_trials": len(train_ids),
        "test_count_trials": len(test_ids),
        "train_trial_ids": train_ids,
        "test_trial_ids": test_ids,
    }


def _zero_based_trial_index(dataset: str, trial_value: int) -> int:
    ds = normalize_dataset_name(dataset)
    t = int(trial_value)
    if ds == "seed1":
        if 1 <= t <= 15:
            return t - 1
        if 0 <= t <= 14:
            return t
    elif ds == "seediv":
        if 0 <= t <= 23:
            return t
        if 1 <= t <= 24:
            return t - 1
    elif ds == "seedv":
        if 0 <= t <= 14:
            return t
        if 1 <= t <= 15:
            return t - 1
    raise ValueError(f"Unexpected native trial index for dataset={dataset}: {trial_value}")


def _official_seed_split(dataset: str, all_trials: Sequence[Dict]) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    ds = normalize_dataset_name(dataset)
    if ds == "seediv":
        train_last = 15
        protocol = "SEED_IV native protocol: per session first 16 trials train, last 8 trials test"
    else:
        train_last = 8
        protocol = (
            "SEED native protocol: per session first 9 trials train, last 6 trials test"
            if ds == "seed1"
            else "SEED_V native protocol: per session first 9 trials train, last 6 trials test"
        )
    train_trials: List[Dict] = []
    test_trials: List[Dict] = []
    for tr in all_trials:
        t0 = _zero_based_trial_index(ds, int(tr["trial"]))
        if t0 <= int(train_last):
            train_trials.append(tr)
        else:
            test_trials.append(tr)
    train_ids = [str(t["trial_id_str"]) for t in train_trials]
    test_ids = [str(t["trial_id_str"]) for t in test_trials]
    return train_trials, test_trials, {
        "protocol": protocol,
        "train_count_trials": int(len(train_trials)),
        "test_count_trials": int(len(test_trials)),
        "train_trial_ids": train_ids,
        "test_trial_ids": test_ids,
        "split_hash": _split_hash(train_ids, test_ids),
    }


def _inner_train_val_split_local(
    trials: Sequence[Dict],
    seed: int,
    val_fraction: float,
) -> Tuple[List[Dict], List[Dict]]:
    if not (0.0 < float(val_fraction) < 1.0):
        return list(trials), []
    y = np.asarray([int(t["label"]) for t in trials], dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=int(seed))
    idx_train, idx_val = next(splitter.split(np.zeros((len(trials), 1)), y))
    train_core = [trials[int(i)] for i in idx_train.tolist()]
    val_core = [trials[int(i)] for i in idx_val.tolist()]
    return train_core, val_core


def resolve_protocol_split(
    *,
    dataset: str,
    all_trials: Sequence[Dict],
    seed: int,
    allow_random_fallback: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    ds = normalize_dataset_name(dataset)

    if ds in OFFICIAL_FIXED_SPLIT_DATASETS:
        train_trials, test_trials = _select_by_split(all_trials)
        train_ids = [str(t["trial_id_str"]) for t in train_trials]
        test_ids = [str(t["trial_id_str"]) for t in test_trials]
        return train_trials, test_trials, {
            "protocol_type": "fixed_split_official",
            "protocol_note": "dataset-provided official TRAIN/TEST split",
            "train_count_trials": int(len(train_trials)),
            "test_count_trials": int(len(test_trials)),
            "train_trial_ids": train_ids,
            "test_trial_ids": test_ids,
            "split_hash": _split_hash(train_ids, test_ids),
        }

    if ds in MITBIH_DATASETS:
        train_trials, test_trials = _select_by_split(all_trials)
        train_ids = [str(t["trial_id_str"]) for t in train_trials]
        test_ids = [str(t["trial_id_str"]) for t in test_trials]
        return train_trials, test_trials, {
            "protocol_type": "mitbih_npz_train_test",
            "protocol_note": "dataset-provided MITBIH npz train/test split",
            "train_count_trials": int(len(train_trials)),
            "test_count_trials": int(len(test_trials)),
            "train_trial_ids": train_ids,
            "test_trial_ids": test_ids,
            "split_hash": _split_hash(train_ids, test_ids),
        }

    if ds in SEED_NATIVE_DATASETS:
        train_trials, test_trials, meta = _official_seed_split(ds, all_trials)
        return train_trials, test_trials, {
            **dict(meta),
            "protocol_type": "seed_family_native",
            "protocol_note": str(meta.get("protocol", "")),
        }

    if allow_random_fallback:
        train_trials, test_trials, meta = _make_trial_split_local(list(all_trials), int(seed))
        return train_trials, test_trials, {
            **dict(meta),
            "protocol_type": "random_split_fallback",
            "protocol_note": "fallback random split because dataset has no official/native protocol helper",
        }

    raise ValueError(f"No supported protocol split for dataset={dataset}")


def resolve_inner_train_val_split(
    *,
    train_trials: Sequence[Dict],
    seed: int,
    val_fraction: float,
    fallback_fraction: float = 0.25,
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    errors: List[str] = []
    tried: List[float] = []
    for frac in [float(val_fraction), float(fallback_fraction)]:
        if frac <= 0.0 or frac >= 1.0:
            continue
        if frac in tried:
            continue
        tried.append(frac)
        try:
            train_core, val_core = _inner_train_val_split_local(train_trials, seed=int(seed), val_fraction=float(frac))
            if len(train_core) > 0 and len(val_core) > 0:
                status = "ok" if abs(frac - float(val_fraction)) <= 1e-12 else "fallback_fraction"
                return train_core, val_core, {
                    "inner_split_status": status,
                    "inner_val_fraction": float(frac),
                    "pool_allowed": True,
                    "inner_split_error": "",
                }
        except Exception as exc:  # pragma: no cover - defensive wrapper
            errors.append(f"{frac:.3f}:{exc}")

    return list(train_trials), [], {
        "inner_split_status": "disabled_no_valid_val_split",
        "inner_val_fraction": 0.0,
        "pool_allowed": False,
        "inner_split_error": " | ".join(errors),
    }
