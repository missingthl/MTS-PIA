from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from datasets.seed_official_mat_dataset import OfficialMatTrialDataset


def _hash_indices(indices: List[int]) -> str:
    payload = ",".join(str(i) for i in indices)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _label_hist(labels: np.ndarray, num_classes: int) -> List[int]:
    return [int((labels == cls).sum()) for cls in range(num_classes)]


def _split_by_subject(
    subjects: List[str], split_seed: int, val_ratio: float
) -> Tuple[List[int], List[int], List[str], List[str]]:
    groups: Dict[str, List[int]] = {}
    for idx, subj in enumerate(subjects):
        groups.setdefault(str(subj), []).append(idx)
    if not groups:
        raise ValueError("No subjects found for split")
    group_list = list(groups.keys())
    rng = np.random.default_rng(int(split_seed))
    rng.shuffle(group_list)
    val_size = int(len(group_list) * val_ratio)
    if val_ratio > 0.0 and val_size == 0 and len(group_list) > 1:
        val_size = 1
    if val_size >= len(group_list):
        val_size = len(group_list) - 1 if len(group_list) > 1 else 0
    val_groups = group_list[:val_size]
    train_groups = group_list[val_size:]
    train_idx = [i for g in train_groups for i in groups[g]]
    val_idx = [i for g in val_groups for i in groups[g]]
    return train_idx, val_idx, train_groups, val_groups


def _split_by_index(
    total: int, split_seed: int, val_ratio: float
) -> Tuple[List[int], List[int]]:
    indices = np.arange(total)
    rng = np.random.default_rng(int(split_seed))
    rng.shuffle(indices)
    val_size = int(total * val_ratio)
    if val_ratio > 0.0 and val_size == 0 and total > 1:
        val_size = 1
    if val_size >= total:
        val_size = total - 1 if total > 1 else 0
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    return train_idx, val_idx


@dataclass
class OfficialBaselineRunner:
    root_dir: str
    feature_base: str = "de_LDS"
    agg_mode: str = "mean_time"
    split_by: str = "subject"
    val_ratio: float = 0.2
    split_seed: int = 42
    model: str = "logreg"
    manifest_path: Optional[str] = None
    run_dir: Optional[str] = None
    seed: Optional[int] = None

    def _resolve_run_dir(self) -> str:
        if self.run_dir:
            return self.run_dir
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("logs", f"official_baseline_{ts}")
        return self.run_dir

    def _write_json(self, name: str, payload: dict) -> str:
        run_dir = self._resolve_run_dir()
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def run(self) -> Dict[str, float]:
        dataset = OfficialMatTrialDataset(
            root_dir=self.root_dir,
            feature_base=self.feature_base,
            agg_mode=self.agg_mode,
            manifest_path=self.manifest_path,
            verbose=True,
        )
        X, y, metas = dataset.build_features()
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values detected in official features")

        subjects = [m["subject"] for m in metas]
        total = len(y)
        if total == 0:
            raise ValueError("Empty official dataset")

        split_by = (self.split_by or "subject").lower()
        if split_by == "subject":
            train_idx, val_idx, train_groups, val_groups = _split_by_subject(
                subjects, self.split_seed, self.val_ratio
            )
        elif split_by == "index":
            train_idx, val_idx = _split_by_index(total, self.split_seed, self.val_ratio)
            train_groups = []
            val_groups = []
        else:
            raise ValueError(f"Unsupported split_by: {self.split_by}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        num_classes = int(np.max(y)) + 1 if y.size else 0

        model_name = (self.model or "logreg").lower()
        if model_name == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    multi_class="multinomial",
                    solver="lbfgs",
                ),
            )
        elif model_name == "svm":
            clf = make_pipeline(StandardScaler(), LinearSVC())
        else:
            raise ValueError(f"Unsupported model: {self.model}")

        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val) if len(val_idx) > 0 else np.array([], dtype=int)

        train_acc = float(accuracy_score(y_train, train_pred))
        train_f1 = float(f1_score(y_train, train_pred, average="macro"))
        val_acc = float(accuracy_score(y_val, val_pred)) if len(val_idx) > 0 else 0.0
        val_f1 = float(f1_score(y_val, val_pred, average="macro")) if len(val_idx) > 0 else 0.0

        label_hist = _label_hist(y, num_classes)
        train_label_hist = _label_hist(y_train, num_classes)
        val_label_hist = _label_hist(y_val, num_classes)

        run_config = {
            "root_dir": self.root_dir,
            "feature_base": dataset.feature_base,
            "agg_mode": self.agg_mode,
            "split_by": split_by,
            "split_seed": int(self.split_seed),
            "val_ratio": float(self.val_ratio),
            "num_subjects": len(set(subjects)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "train_subject_count": len(train_groups) if split_by == "subject" else None,
            "val_subject_count": len(val_groups) if split_by == "subject" else None,
            "train_subjects_preview": train_groups[:10] if train_groups else None,
            "val_subjects_preview": val_groups[:10] if val_groups else None,
            "train_indices_sha1": _hash_indices(train_idx),
            "val_indices_sha1": _hash_indices(val_idx),
            "label_hist": label_hist,
            "train_label_hist": train_label_hist,
            "val_label_hist": val_label_hist,
            "feature_dim": int(X.shape[1]),
            "model": model_name,
            "manifest_path": self.manifest_path,
        }
        metrics = [
            {
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        ]

        run_config_path = self._write_json("run_config.json", run_config)
        metrics_path = self._write_json("metrics.json", metrics)
        print(f"[official_baseline] run_config={run_config_path}", flush=True)
        print(f"[official_baseline] metrics={metrics_path}", flush=True)
        print(
            f"[official_baseline] train_acc={train_acc:.4f} train_macro_f1={train_f1:.4f} "
            f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}",
            flush=True,
        )
        return {
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
