from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np


def _resolve_fingermovements_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if (p / "FingerMovements_TRAIN.ts").is_file() and (p / "FingerMovements_TEST.ts").is_file():
        return p
    nested = p / "FingerMovements"
    if (nested / "FingerMovements_TRAIN.ts").is_file() and (nested / "FingerMovements_TEST.ts").is_file():
        return nested
    raise FileNotFoundError(
        "FingerMovements root not found. Expected FingerMovements_TRAIN.ts and FingerMovements_TEST.ts. "
        f"Got: {p}"
    )


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
    v = header.get("@classlabel", "")
    toks = [t.strip() for t in v.split() if t.strip()]
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
        tokens = [t.strip() for t in dim.split(",")]
        vals: List[float] = []
        for t in tokens:
            if t == "":
                continue
            if t == "?":
                vals.append(0.0)
            else:
                vals.append(float(t))
        arr = np.asarray(vals, dtype=np.float32)
        if seq_len is None:
            seq_len = int(arr.shape[0])
        elif int(arr.shape[0]) != seq_len:
            raise ValueError(f"Inconsistent sequence length inside one sample: {arr.shape[0]} vs {seq_len}")
        rows.append(arr)
    if seq_len is None or seq_len <= 0:
        raise ValueError("Invalid .ts row: empty series")
    x = np.stack(rows, axis=0)  # [C, T]
    return x, label_raw


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
    class_order = _parse_class_order(header)
    return x_all, y_list, class_order


class FingerMovementsTrialDataset:
    """Expose FingerMovements UEA-format multivariate time series in common trial dict format."""

    def __init__(
        self,
        root: str | Path,
        *,
        sfreq: float = 20.0,
        include_splits: Iterable[str] = ("train", "test"),
    ) -> None:
        self.root = _resolve_fingermovements_root(root)
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

        train_x, train_y_raw, train_order = _load_ts_file(self.root / "FingerMovements_TRAIN.ts")
        test_x, test_y_raw, test_order = _load_ts_file(self.root / "FingerMovements_TEST.ts")

        if train_order:
            class_order = train_order
        elif test_order:
            class_order = test_order
        else:
            class_order = sorted(set(train_y_raw + test_y_raw))
        label_map = {str(lbl): idx for idx, lbl in enumerate(class_order)}

        # Ensure all labels are covered by class_order.
        missing = sorted((set(train_y_raw) | set(test_y_raw)) - set(label_map.keys()))
        if missing:
            for m in missing:
                label_map[m] = len(label_map)

        self._label_map = label_map

        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
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
                    "trial_id_str": f"fingermovements_{split}_n{i:05d}",
                    "x_trial": x[i],
                    "label": int(y[i]),
                    "sfreq": self.sfreq,
                    "subject": -1,
                    "session": sess,
                    "trial": int(i),
                    "split": split,
                }
