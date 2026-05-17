from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import scipy.io


# Labels from SEED-IV ReadMe.txt (session-level fixed order, 24 trials/session).
_SESSION_LABELS: Dict[int, List[int]] = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


def _resolve_seediv_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if (p / "eeg_raw_data").is_dir():
        return p
    raise FileNotFoundError(
        "SEED-IV root not found. Expected directory containing eeg_raw_data/. "
        f"Got: {p}"
    )


def _parse_subject_date(stem: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+)_([0-9]{8})$", stem)
    if not m:
        raise ValueError(f"Invalid SEED-IV filename: {stem}")
    return int(m.group(1)), int(m.group(2))


def _extract_trial_keys(keys: Iterable[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k in keys:
        if k.startswith("__"):
            continue
        m = re.match(r".*eeg(\d+)$", k)
        if not m:
            continue
        trial_idx = int(m.group(1))
        out[trial_idx] = k
    return out


class SEEDIVRawTrialDataset:
    """SEED-IV raw EEG trial dataset in common trial dict format."""

    def __init__(
        self,
        root: str | Path,
        *,
        include_sessions: Iterable[int] = (1, 2, 3),
        sfreq: float = 200.0,
    ) -> None:
        self.root = _resolve_seediv_root(root)
        self.sfreq = float(sfreq)
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {sfreq}")

        sessions = [int(s) for s in include_sessions]
        if not sessions:
            raise ValueError("include_sessions cannot be empty")
        for s in sessions:
            if s not in (1, 2, 3):
                raise ValueError(f"Unsupported session index: {s}")
        self.include_sessions = tuple(sorted(set(sessions)))

        self._records: List[Dict] = []
        eeg_raw_root = self.root / "eeg_raw_data"
        for session in self.include_sessions:
            session_dir = eeg_raw_root / str(session)
            if not session_dir.is_dir():
                raise FileNotFoundError(f"Missing SEED-IV session directory: {session_dir}")
            files = sorted(p for p in session_dir.iterdir() if p.suffix.lower() == ".mat")
            if not files:
                raise FileNotFoundError(f"No .mat files found in {session_dir}")
            for p in files:
                subject, date = _parse_subject_date(p.stem)
                self._records.append(
                    {
                        "session": session,
                        "subject": subject,
                        "date": date,
                        "path": str(p),
                    }
                )

        self._records.sort(key=lambda x: (x["session"], x["subject"], x["date"]))

    def __len__(self) -> int:
        return len(self._records) * 24

    def __iter__(self) -> Iterator[Dict]:
        for rec in self._records:
            session = int(rec["session"])
            subject = int(rec["subject"])
            date = int(rec["date"])
            mat_path = str(rec["path"])

            labels = _SESSION_LABELS[session]
            mat = scipy.io.loadmat(mat_path)
            trial_keys = _extract_trial_keys(mat.keys())
            if len(trial_keys) != 24:
                raise ValueError(
                    f"Expected 24 EEG trials in {mat_path}, found {len(trial_keys)}: "
                    f"{sorted(trial_keys.keys())}"
                )

            for trial_1based in range(1, 25):
                if trial_1based not in trial_keys:
                    raise ValueError(f"Missing trial {trial_1based} in {mat_path}")
                key = trial_keys[trial_1based]
                arr = np.asarray(mat[key], dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Expected 2D EEG array in {mat_path}:{key}, got {arr.shape}")
                if arr.shape[0] != 62 and arr.shape[1] == 62:
                    arr = arr.T
                if arr.shape[0] != 62:
                    raise ValueError(f"Expected 62 channels in {mat_path}:{key}, got {arr.shape}")

                trial = trial_1based - 1
                label = int(labels[trial])
                yield {
                    "trial_id_str": f"seediv_s{session}_sub{subject:02d}_t{trial:02d}",
                    "x_trial": arr,
                    "label": label,
                    "sfreq": self.sfreq,
                    "subject": subject,
                    "session": session,
                    "trial": trial,
                    "record": date,
                    "split": "all",
                }
