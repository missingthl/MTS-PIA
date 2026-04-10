from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from datasets.seed_official_mat_dataset import build_official_trial_index, _normalize_trial_array


class OfficialMatSequenceDataset:
    """Load SEED official .mat trial arrays with strict shape normalization."""

    def __init__(
        self,
        *,
        manifest_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        feature_base: str = "de_LDS",
        feature_key: str = "de_LDS1",
        mode: str = "trial_spd",
        window_size: int = 20,
        window_step: int = 1,
        label_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.feature_base = feature_base
        self.feature_key = feature_key
        self.mode = (mode or "trial_spd").strip().lower()
        self.window_size = int(window_size)
        self.window_step = int(window_step)
        self.manifest_path = manifest_path
        self.root_dir = root_dir
        self.label_path = label_path
        self.verbose = verbose
        self._mat_cache: Dict[str, dict] = {}
        self.trials, self.meta = self._load_trials()
        if self.mode not in {"trial_spd", "window_spd"}:
            raise ValueError(f"Unsupported mode for OfficialMatSequenceDataset: {self.mode}")
        if self.mode == "window_spd":
            if self.window_size <= 0 or self.window_step <= 0:
                raise ValueError("window_size/window_step must be positive for window_spd")
        if self.verbose:
            self._print_example()

    def _load_trials(self) -> Tuple[List[dict], dict]:
        if self.manifest_path:
            data = json.loads(Path(self.manifest_path).read_text())
            if isinstance(data, dict) and "trials" in data:
                return list(data["trials"]), data.get("meta", {})
            if isinstance(data, list):
                return list(data), {}
            raise ValueError("Unsupported manifest format for official dataset")
        if not self.root_dir:
            raise ValueError("manifest_path or root_dir must be provided")
        return build_official_trial_index(
            root_dir=self.root_dir,
            feature_base=self.feature_base,
            label_path=self.label_path,
        )

    def __len__(self) -> int:
        return len(self.trials)

    def _load_mat(self, path: str) -> dict:
        if path not in self._mat_cache:
            self._mat_cache[path] = scipy.io.loadmat(path)
        return self._mat_cache[path]

    def _resolve_key(self, trial: dict, mat: dict) -> str:
        key_name = trial.get("key_name")
        if key_name and key_name in mat:
            return key_name
        if self.feature_key in mat:
            return self.feature_key
        base = "".join([c for c in self.feature_key if not c.isdigit() and c != "_"])
        trial_idx = int(trial.get("trial", -1)) + 1
        candidate = f"{base}{trial_idx}"
        if candidate in mat:
            return candidate
        raise KeyError(
            f"Cannot resolve key for trial={trial.get('trial')} "
            f"feature_key={self.feature_key} mat_keys={sorted(k for k in mat.keys() if not k.startswith('__'))[:10]}"
        )

    def __getitem__(self, idx: int):
        trial = self.trials[idx]
        mat_path = trial["mat_path"]
        mat = self._load_mat(mat_path)
        key_name = self._resolve_key(trial, mat)
        arr = mat[key_name]
        raw_shape = tuple(arr.shape)
        arr = _normalize_trial_array(arr)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in {mat_path} key={key_name}")
        arr = arr.transpose(1, 0, 2).astype(np.float32, copy=False)  # [T, 62, 5]
        if self.mode == "window_spd":
            arr = self._slice_windows(arr)
        meta = {
            "subject": trial.get("subject"),
            "session": trial.get("session"),
            "trial": trial.get("trial"),
            "label": int(trial.get("label")),
            "mat_path": mat_path,
            "key_name": key_name,
            "raw_shape": raw_shape,
            "final_shape": tuple(arr.shape),
        }
        return arr, int(trial.get("label")), meta

    def _slice_windows(self, arr: np.ndarray) -> np.ndarray:
        t_len = int(arr.shape[0])
        if t_len <= self.window_size:
            return arr[None, ...]
        starts = list(range(0, t_len - self.window_size + 1, self.window_step))
        windows = [arr[s : s + self.window_size] for s in starts]
        return np.stack(windows, axis=0)

    def _print_example(self) -> None:
        if not self.trials:
            return
        arr, _, meta = self[0]
        print(
            "[official_mat_seq] example_key="
            f"{meta['key_name']} raw_shape={list(meta['raw_shape'])} "
            f"final_shape={list(meta['final_shape'])}",
            flush=True,
        )
