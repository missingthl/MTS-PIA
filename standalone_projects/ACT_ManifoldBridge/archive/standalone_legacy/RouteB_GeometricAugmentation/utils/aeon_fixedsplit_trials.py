from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class AeonFixedSplitSpec:
    dataset_key: str
    dataset_name: str
    sfreq: float


AEON_FIXED_SPLIT_SPECS: Dict[str, AeonFixedSplitSpec] = {
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


def _resolve_dataset_root(path: str | Path, dataset_name: str) -> Path:
    p = Path(path).expanduser().resolve()
    train_name = f"{dataset_name}_TRAIN.ts"
    test_name = f"{dataset_name}_TEST.ts"
    npz_name = f"{dataset_name}_fixedsplit.npz"
    if (p / npz_name).is_file():
        return p
    if (p / train_name).is_file() and (p / test_name).is_file():
        return p
    nested = p / dataset_name
    if (nested / npz_name).is_file():
        return nested
    if (nested / train_name).is_file() and (nested / test_name).is_file():
        return nested
    raise FileNotFoundError(
        f"{dataset_name} root not found. Expected {train_name}/{test_name} or {npz_name}. Got: {p}"
    )


def _load_fixedsplit_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        train_x = np.asarray(data["X_train"], dtype=np.float32)
        train_y = np.asarray(data["y_train"], dtype=np.int64)
        test_x = np.asarray(data["X_test"], dtype=np.float32)
        test_y = np.asarray(data["y_test"], dtype=np.int64)
    return train_x, train_y, test_x, test_y


def _parse_ts_header(lines: List[str]) -> Dict[str, str]:
    header: Dict[str, str] = {}
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.lower() == "@data":
            break
        if not s.startswith("@"):
            continue
        parts = s.split(None, 1)
        key = parts[0].lower()
        value = parts[1].strip() if len(parts) > 1 else ""
        header[key] = value
    return header


def _parse_class_order(header: Dict[str, str]) -> List[str]:
    raw = header.get("@classlabel", "")
    toks = [t.strip() for t in raw.split() if t.strip()]
    if not toks:
        return []
    if toks[0].lower() in {"true", "false"}:
        toks = toks[1:]
    return toks


def _parse_ts_line(line: str) -> Tuple[np.ndarray, str]:
    parts = line.strip().split(":")
    if len(parts) < 2:
        raise ValueError("Invalid .ts row: expected at least one dimension plus label")
    dims_raw = parts[:-1]
    label_raw = parts[-1].strip()

    rows: List[np.ndarray] = []
    seq_len: int | None = None
    for dim in dims_raw:
        vals: List[float] = []
        for token in [t.strip() for t in dim.split(",")]:
            if token == "":
                continue
            if token == "?":
                vals.append(0.0)
            else:
                vals.append(float(token))
        arr = np.asarray(vals, dtype=np.float32)
        if seq_len is None:
            seq_len = int(arr.shape[0])
        elif int(arr.shape[0]) != seq_len:
            raise ValueError(f"Inconsistent sequence length inside one sample: {arr.shape[0]} vs {seq_len}")
        rows.append(arr)

    if seq_len is None or seq_len <= 0:
        raise ValueError("Invalid .ts row: empty series")
    return np.stack(rows, axis=0), label_raw


def _load_ts_file(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty .ts file: {path}")
    header = _parse_ts_header(lines)

    start_idx = None
    for i, raw in enumerate(lines):
        if raw.strip().lower() == "@data":
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"Missing @data in {path}")

    expected_dims = None
    if "@dimensions" in header:
        try:
            expected_dims = int(float(header["@dimensions"]))
        except Exception:
            expected_dims = None

    x_list: List[np.ndarray] = []
    y_list: List[str] = []
    for raw in lines[start_idx:]:
        s = raw.strip()
        if not s:
            continue
        x, y = _parse_ts_line(s)
        if expected_dims is not None and x.shape[0] != expected_dims:
            raise ValueError(f"Dimension mismatch in {path}: got {x.shape[0]}, expected {expected_dims}")
        x_list.append(x)
        y_list.append(y)

    if not x_list:
        raise ValueError(f"No samples parsed from {path}")

    seq_lens = {int(x.shape[1]) for x in x_list}
    if len(seq_lens) != 1:
        raise ValueError(f"Variable sequence length not supported in this loader: {sorted(seq_lens)}")

    x_all = np.stack(x_list, axis=0).astype(np.float32, copy=False)
    return x_all, y_list, _parse_class_order(header)


class AeonFixedSplitTrialDataset:
    """Generic fixed-split loader for aeon/UEA-style multivariate .ts datasets."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_key: str,
        sfreq: float | None = None,
        include_splits: Iterable[str] = ("train", "test"),
    ) -> None:
        key = str(dataset_key).strip().lower()
        if key not in AEON_FIXED_SPLIT_SPECS:
            raise ValueError(f"Unsupported aeon fixed-split dataset: {dataset_key}")
        spec = AEON_FIXED_SPLIT_SPECS[key]

        self.dataset_key = spec.dataset_key
        self.dataset_name = spec.dataset_name
        self.root = _resolve_dataset_root(root, spec.dataset_name)
        self.sfreq = float(spec.sfreq if sfreq is None else sfreq)
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")

        splits = [str(s).strip().lower() for s in include_splits if str(s).strip()]
        if not splits:
            raise ValueError("include_splits cannot be empty")
        for s in splits:
            if s not in {"train", "test"}:
                raise ValueError(f"Unsupported split: {s}")
        self.include_splits = tuple(splits)

        npz_path = self.root / f"{spec.dataset_name}_fixedsplit.npz"
        if npz_path.is_file():
            train_x, train_y, test_x, test_y = _load_fixedsplit_npz(npz_path)
            self._cache = {
                "train": (train_x, train_y),
                "test": (test_x, test_y),
            }
        else:
            train_x, train_y_raw, train_order = _load_ts_file(self.root / f"{spec.dataset_name}_TRAIN.ts")
            test_x, test_y_raw, test_order = _load_ts_file(self.root / f"{spec.dataset_name}_TEST.ts")

            class_order = train_order or test_order or sorted(set(train_y_raw + test_y_raw))
            label_map = {str(lbl): idx for idx, lbl in enumerate(class_order)}
            missing = sorted((set(train_y_raw) | set(test_y_raw)) - set(label_map.keys()))
            for m in missing:
                label_map[m] = len(label_map)

            self._cache = {
                "train": (train_x, np.asarray([label_map[str(v)] for v in train_y_raw], dtype=np.int64)),
                "test": (test_x, np.asarray([label_map[str(v)] for v in test_y_raw], dtype=np.int64)),
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
                    "label": int(y[i]),
                    "sfreq": self.sfreq,
                    "subject": -1,
                    "session": sess,
                    "trial": int(i),
                    "split": split,
                }
