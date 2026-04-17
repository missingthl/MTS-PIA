from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Iterable, Optional, Any

@dataclass(frozen=True)
class AeonFixedSplitSpec:
    dataset_key: str
    dataset_name: str
    sfreq: float

AEON_FIXED_SPLIT_SPECS: Dict[str, AeonFixedSplitSpec] = {
    "natops": AeonFixedSplitSpec("natops", "NATOPS", 20.0),
    "har": AeonFixedSplitSpec("har", "UCIHAR", 50.0),
    "fingermovements": AeonFixedSplitSpec("fingermovements", "FingerMovements", 20.0),
    "basicmotions": AeonFixedSplitSpec("basicmotions", "BasicMotions", 10.0),
    "handmovementdirection": AeonFixedSplitSpec("handmovementdirection", "HandMovementDirection", 20.0),
    "uwavegesturelibrary": AeonFixedSplitSpec("uwavegesturelibrary", "UWaveGestureLibrary", 20.0),
    "epilepsy": AeonFixedSplitSpec("epilepsy", "Epilepsy", 16.0),
    "atrialfibrillation": AeonFixedSplitSpec("atrialfibrillation", "AtrialFibrillation", 128.0),
    "pendigits": AeonFixedSplitSpec("pendigits", "PenDigits", 8.0),
    "racketsports": AeonFixedSplitSpec("racketsports", "RacketSports", 1.0),
    "articularywordrecognition": AeonFixedSplitSpec("articularywordrecognition", "ArticularyWordRecognition", 1.0),
    "heartbeat": AeonFixedSplitSpec("heartbeat", "Heartbeat", 1.0),
    "selfregulationscp2": AeonFixedSplitSpec("selfregulationscp2", "SelfRegulationSCP2", 1.0),
    "libras": AeonFixedSplitSpec("libras", "Libras", 1.0),
    "japanesevowels": AeonFixedSplitSpec("japanesevowels", "JapaneseVowels", 1.0),
    "cricket": AeonFixedSplitSpec("cricket", "Cricket", 1.0),
    "handwriting": AeonFixedSplitSpec("handwriting", "Handwriting", 1.0),
    "ering": AeonFixedSplitSpec("ering", "ERing", 1.0),
    "motorimagery": AeonFixedSplitSpec("motorimagery", "MotorImagery", 1.0),
    "ethanolconcentration": AeonFixedSplitSpec("ethanolconcentration", "EthanolConcentration", 1.0),
}

def _resolve_dataset_root(dataset_name: str) -> Path:
    # 1. Check local data folder
    local_data = Path("data").resolve()
    # 2. Check main project data folder
    main_data = Path("../../data").resolve()
    
    # Define aliases for specific datasets
    aliases = {
        "har": ["HAR", "UCIHAR", "UCI-HAR"]
    }
    candidates = [dataset_name] + aliases.get(dataset_name.lower(), [])
    
    for base in [local_data, main_data]:
        for cand in candidates:
            # Try direct name
            p = base / cand
            if (p / f"{dataset_name}_TRAIN.ts").is_file():
                return p
            if (p / f"{cand}_TRAIN.ts").is_file():
                return p
            # Try lowercase name
            p_low = base / cand.lower()
            if (p_low / f"{dataset_name}_TRAIN.ts").is_file():
                return p_low
            if (p_low / f"{cand.lower()}_TRAIN.ts").is_file():
                return p_low
            # Try parent folder
            if (base / f"{dataset_name}_TRAIN.ts").is_file():
                return base
            if (base / f"{cand}_TRAIN.ts").is_file():
                return base

    raise FileNotFoundError(f"Dataset {dataset_name} (candidates: {candidates}) not found in {local_data} or {main_data}")

def _resolve_har_root(dataset_name: str) -> Path:
    local_data = Path("data").resolve()
    main_data = Path("../../data").resolve()
    candidates = ["har", "HAR", "UCIHAR", "UCI-HAR"]
    
    for base in [local_data, main_data]:
        for cand in candidates:
            p = base / cand
            # Check for nesting
            p_tries = [p, p / "UCI HAR Dataset", p / "UCI HAR Dataset" / "UCI HAR Dataset"]
            for pt in p_tries:
                if (pt / "train").is_dir() and (pt / "test").is_dir():
                    return pt
    raise FileNotFoundError(f"Raw HAR directory not found in {local_data} or {main_data}")

def _load_har_split(split_root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    inertial_dir = split_root / split / "Inertial Signals"
    channels = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    signals = []
    for ch in channels:
        fp = inertial_dir / f"{ch}_{split}.txt"
        signals.append(np.loadtxt(fp, dtype=np.float32))
    
    x = np.stack(signals, axis=1)  # [N, 9, T=128]
    y_path = split_root / split / f"y_{split}.txt"
    y = np.loadtxt(y_path, dtype=np.int64).reshape(-1) - 1
    return x, y

class UCIHARTrialDataset:
    def __init__(self, root: Path):
        self.root = root
        self.train_x, self.train_y = _load_har_split(self.root, "train")
        self.test_x, self.test_y = _load_har_split(self.root, "test")

def _load_ts_file(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_data = True
    x_list, y_list = [], []
    class_order = []
    
    for line in lines:
        s = line.strip()
        if not s: continue
        if s.lower().startswith("@classlabel"):
            class_order = [t.strip() for t in s.split()[1:] if t.strip()]
            if class_order and class_order[0].lower() in {"true", "false"}: 
                class_order = class_order[1:]
        if s.lower() == "@data":
            header_data = False
            continue
        if header_data: continue
        
        parts = s.split(":")
        dims_raw = parts[:-1]
        label = parts[-1].strip()
        
        rows = []
        for dim in dims_raw:
            vals = [float(t.strip()) if t.strip() != "?" else 0.0 for t in dim.split(",") if t.strip()]
            rows.append(np.array(vals, dtype=np.float32))
        
        # All dimensions in one sample SHOULD have same length
        row_lens = [len(r) for r in rows]
        if len(set(row_lens)) > 1:
            # Handle rare case of inconsistent dim length within sample
            max_row_len = max(row_lens)
            rows = [np.pad(r, (0, max_row_len - len(r))) for r in rows]
        
        x_list.append(np.stack(rows, axis=0)) # (D, T)
        y_list.append(label)
    
    if not x_list:
        raise ValueError(f"No data found in {path}")
        
    # --- PADDING LOGIC FOR VARIABLE LENGTH ACROSS SAMPLES ---
    max_seq_len = max(x.shape[1] for x in x_list)
    num_samples = len(x_list)
    num_dims = x_list[0].shape[0]
    
    x_padded = np.zeros((num_samples, num_dims, max_seq_len), dtype=np.float32)
    for i, x in enumerate(x_list):
        curr_len = x.shape[1]
        x_padded[i, :, :curr_len] = x
        
    return x_padded, y_list, class_order

class AeonFixedSplitTrialDataset:
    def __init__(self, dataset_key: str):
        spec = AEON_FIXED_SPLIT_SPECS[dataset_key.lower()]
        self.root = _resolve_dataset_root(spec.dataset_name)
        self.sfreq = spec.sfreq
        self.dataset_key = dataset_key
        
        train_x, train_y_raw, train_order = _load_ts_file(self.root / f"{spec.dataset_name}_TRAIN.ts")
        test_x, test_y_raw, test_order = _load_ts_file(self.root / f"{spec.dataset_name}_TEST.ts")
        
        order = train_order or test_order or sorted(set(train_y_raw + test_y_raw))
        label_map = {str(lbl): idx for idx, lbl in enumerate(order)}
        
        self.train_x = train_x
        self.train_y = np.array([label_map[y] for y in train_y_raw], dtype=np.int64)
        self.test_x = test_x
        self.test_y = np.array([label_map[y] for y in test_y_raw], dtype=np.int64)

@dataclass
class Trial:
    tid: str
    x: np.ndarray
    y: int
    split: str

def load_trials_for_dataset(dataset_name: str) -> List[Trial]:
    ds_key = dataset_name.lower()
    if ds_key not in AEON_FIXED_SPLIT_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if ds_key == "har":
        root = _resolve_har_root("har")
        loader: Any = UCIHARTrialDataset(root)
    else:
        loader = AeonFixedSplitTrialDataset(ds_key)
    
    trials = []
    for i in range(loader.train_x.shape[0]):
        trials.append(Trial(tid=f"{ds_key}_train_{i}", x=loader.train_x[i], y=int(loader.train_y[i]), split="train"))
    for i in range(loader.test_x.shape[0]):
        trials.append(Trial(tid=f"{ds_key}_test_{i}", x=loader.test_x[i], y=int(loader.test_y[i]), split="test"))
    return trials

def make_trial_split(trials: List[Trial], seed: int) -> Tuple[List[Trial], List[Trial], None]:
    train = [t for t in trials if t.split == "train"]
    test = [t for t in trials if t.split == "test"]
    return train, test, None
