from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np


def _resolve_extract_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


class IEEEPPGTrialDataset:
    """Expose aeon IEEEPPG regression data in the common trial-dict style."""

    def __init__(
        self,
        root: str | Path,
        *,
        include_splits: Iterable[str] = ("train", "test"),
    ) -> None:
        try:
            from aeon.datasets import load_regression
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "IEEEPPG regression loader requires aeon in the pia environment."
            ) from exc

        self.dataset_key = "ieeeppg"
        self.dataset_name = "IEEEPPG"
        self.root = _resolve_extract_root(root)

        splits = [str(s).strip().lower() for s in include_splits if str(s).strip()]
        if not splits:
            raise ValueError("include_splits cannot be empty")
        for s in splits:
            if s not in {"train", "test"}:
                raise ValueError(f"Unsupported split: {s}")
        self.include_splits = tuple(splits)

        train_x, train_y = load_regression(self.dataset_name, split="train", extract_path=str(self.root))
        test_x, test_y = load_regression(self.dataset_name, split="test", extract_path=str(self.root))

        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "train": (np.asarray(train_x, dtype=np.float32), np.asarray(train_y, dtype=np.float64)),
            "test": (np.asarray(test_x, dtype=np.float32), np.asarray(test_y, dtype=np.float64)),
        }

    def __len__(self) -> int:
        n = 0
        for split in self.include_splits:
            n += int(self._cache[split][0].shape[0])
        return n

    def __iter__(self) -> Iterator[Dict]:
        for split in self.include_splits:
            x, y = self._cache[split]
            sess = 1 if split == "train" else 2
            for i in range(x.shape[0]):
                yield {
                    "trial_id_str": f"{self.dataset_key}_{split}_n{i:05d}",
                    "x_trial": x[i],
                    "y_value": float(y[i]),
                    "sfreq": None,
                    "subject": -1,
                    "session": sess,
                    "trial": int(i),
                    "split": split,
                }
