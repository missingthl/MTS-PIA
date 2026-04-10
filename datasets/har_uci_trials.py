from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np


_HAR_CHANNEL_FILES: Tuple[str, ...] = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def _resolve_har_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    candidates = (
        p,
        p / "UCI HAR Dataset",
        p / "UCI HAR Dataset" / "UCI HAR Dataset",
    )
    for c in candidates:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    raise FileNotFoundError(
        "UCI HAR root not found. Expected a directory containing train/ and test/. "
        f"Got: {p}"
    )


def _load_split(split_root: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inertial_dir = split_root / split / "Inertial Signals"
    if not inertial_dir.is_dir():
        raise FileNotFoundError(f"Inertial Signals folder missing: {inertial_dir}")

    signals: List[np.ndarray] = []
    n_rows_ref: int | None = None
    n_cols_ref: int | None = None
    for ch_name in _HAR_CHANNEL_FILES:
        fp = inertial_dir / f"{ch_name}_{split}.txt"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing channel file: {fp}")
        arr = np.loadtxt(fp, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix in {fp}, got shape={arr.shape}")
        if n_rows_ref is None:
            n_rows_ref, n_cols_ref = int(arr.shape[0]), int(arr.shape[1])
        elif arr.shape[0] != n_rows_ref or arr.shape[1] != n_cols_ref:
            raise ValueError(
                f"Channel file shape mismatch in {fp}. Expected {(n_rows_ref, n_cols_ref)}, got {arr.shape}"
            )
        signals.append(arr)

    x = np.stack(signals, axis=1)  # [N, 9, T=128]

    y_path = split_root / split / f"y_{split}.txt"
    s_path = split_root / split / f"subject_{split}.txt"
    if not y_path.is_file():
        raise FileNotFoundError(f"Missing label file: {y_path}")
    if not s_path.is_file():
        raise FileNotFoundError(f"Missing subject file: {s_path}")

    y = np.loadtxt(y_path, dtype=np.int64).reshape(-1)
    subjects = np.loadtxt(s_path, dtype=np.int64).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] != subjects.shape[0]:
        raise ValueError(
            "HAR split size mismatch: "
            f"x={x.shape[0]} y={y.shape[0]} subjects={subjects.shape[0]} ({split})"
        )

    # UCI HAR labels are 1..6, convert to 0..5
    y = y - 1
    uniq = sorted(set(y.tolist()))
    if uniq and (min(uniq) < 0 or max(uniq) > 5):
        raise ValueError(f"Unexpected HAR labels in {split}: {uniq}")

    return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False), subjects.astype(np.int64, copy=False)


class UCIHARTrialDataset:
    """Expose UCI HAR inertial windows in the same trial dict format as SEED loaders."""

    def __init__(
        self,
        root: str | Path,
        *,
        sfreq: float = 50.0,
        include_splits: Iterable[str] = ("train", "test"),
    ) -> None:
        self.root = _resolve_har_root(root)
        self.sfreq = float(sfreq)
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {sfreq}")

        splits = [str(s).strip().lower() for s in include_splits if str(s).strip()]
        if not splits:
            raise ValueError("include_splits cannot be empty")
        for s in splits:
            if s not in {"train", "test"}:
                raise ValueError(f"Unsupported split: {s}")
        self.include_splits = tuple(splits)

        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for split in self.include_splits:
            self._cache[split] = _load_split(self.root, split)

    def __len__(self) -> int:
        n = 0
        for split in self.include_splits:
            n += int(self._cache[split][0].shape[0])
        return n

    def __iter__(self) -> Iterator[Dict]:
        for split in self.include_splits:
            x, y, subjects = self._cache[split]
            sess = 1 if split == "train" else 2
            for i in range(x.shape[0]):
                subj = int(subjects[i])
                yield {
                    "trial_id_str": f"har_{split}_s{subj:02d}_n{i:05d}",
                    "x_trial": x[i],
                    "label": int(y[i]),
                    "sfreq": self.sfreq,
                    "subject": subj,
                    "session": sess,
                    "trial": int(i),
                    "split": split,
                }
