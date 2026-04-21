from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np


class MITBIHBeatTrialDataset:
    """Expose preprocessed MIT-BIH beat samples in trial dict format used by Phase14/15 scripts."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        include_splits: Iterable[str] = ("train", "test"),
    ) -> None:
        p = Path(npz_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"MIT-BIH preprocessed npz not found: {p}. "
                "Run datasets/mitbih_preprocess.py first."
            )
        self.path = p

        splits = [str(s).strip().lower() for s in include_splits if str(s).strip()]
        if not splits:
            raise ValueError("include_splits cannot be empty")
        for s in splits:
            if s not in {"train", "test"}:
                raise ValueError(f"Unsupported split: {s}")
        self.include_splits = tuple(splits)

        self._npz = np.load(str(self.path), allow_pickle=False)
        self._sfreq = float(self._npz.get("sfreq", np.asarray([360.0], dtype=np.float32))[0])
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for split in self.include_splits:
            self._cache[split] = self._load_split(split)

    def _load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_key = f"X_{split}"
        y_key = f"y_{split}"
        t_key = f"tid_{split}"
        r_key = f"record_{split}"
        for k in (x_key, y_key, t_key, r_key):
            if k not in self._npz:
                raise KeyError(f"Missing key `{k}` in {self.path}")

        X = np.asarray(self._npz[x_key], dtype=np.float32)
        y = np.asarray(self._npz[y_key], dtype=np.int64).reshape(-1)
        tid = np.asarray(self._npz[t_key]).astype(str).reshape(-1)
        rec = np.asarray(self._npz[r_key], dtype=np.int32).reshape(-1)
        if X.ndim != 3:
            raise ValueError(f"Expected X_{split} shape [N,C,T], got {X.shape}")
        if not (X.shape[0] == y.shape[0] == tid.shape[0] == rec.shape[0]):
            raise ValueError(
                f"Length mismatch in split `{split}`: "
                f"X={X.shape[0]} y={y.shape[0]} tid={tid.shape[0]} rec={rec.shape[0]}"
            )
        return X, y, tid, rec

    def __len__(self) -> int:
        total = 0
        for split in self.include_splits:
            total += int(self._cache[split][0].shape[0])
        return total

    def __iter__(self) -> Iterator[Dict]:
        for split in self.include_splits:
            X, y, tid, rec = self._cache[split]
            sess = 1 if split == "train" else 2
            for i in range(X.shape[0]):
                yield {
                    "trial_id_str": str(tid[i]),
                    "x_trial": X[i],
                    "label": int(y[i]),
                    "sfreq": self._sfreq,
                    "subject": int(rec[i]),
                    "session": sess,
                    "trial": int(i),
                    "split": split,
                    "record": int(rec[i]),
                }

