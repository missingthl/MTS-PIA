from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np
from scipy.signal import resample_poly

from .seedv_preprocess import iter_seedv_trials_meta


def _resolve_seedv_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if (p / "EEG_raw").is_dir() and (p / "EEG_DE_features").is_dir():
        return p
    raise FileNotFoundError(
        "SEED_V root not found. Expected EEG_raw/ and EEG_DE_features/. "
        f"Got: {p}"
    )


class SEEDVRawTrialDataset:
    """Expose SEED_V raw EEG trials in the common trial dict format."""

    def __init__(
        self,
        root: str | Path,
        *,
        sfreq: float = 200.0,
        include_sessions: Iterable[int] = (1, 2, 3),
    ) -> None:
        self.root = _resolve_seedv_root(root)
        self.sfreq = float(sfreq)
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {sfreq}")
        native_sfreq = 1000.0
        if native_sfreq % self.sfreq != 0:
            raise ValueError(f"SEED_V native sfreq={native_sfreq} not divisible by target sfreq={self.sfreq}")
        decim = int(round(native_sfreq / self.sfreq))

        sessions_in = tuple(sorted(set(int(s) for s in include_sessions)))
        if not sessions_in:
            raise ValueError("include_sessions cannot be empty")
        sessions_1based = set()
        zero_based_mode = 0 in sessions_in
        for s in sessions_in:
            if zero_based_mode:
                if s not in (0, 1, 2):
                    raise ValueError(f"Unsupported 0-based session index: {s}")
                sessions_1based.add(int(s) + 1)
            else:
                if s not in (1, 2, 3):
                    raise ValueError(f"Unsupported 1-based session index: {s}")
                sessions_1based.add(int(s))
        self.include_sessions = tuple(sorted(sessions_1based))

        self._records: List[Dict] = []
        for trial_id, x_trial_tc, label, trial_idx, session_idx, subject in iter_seedv_trials_meta(
            source="raw",
            seedv_de_root=str(self.root / "EEG_DE_features"),
            seedv_raw_root=str(self.root / "EEG_raw"),
            raw_repr="signal",
            raw_cache="off",
            raw_channel_policy="strict",
        ):
            session_1based = int(session_idx) + 1
            if session_1based not in self.include_sessions:
                continue
            x_trial = np.asarray(x_trial_tc, dtype=np.float32)
            if x_trial.ndim != 2:
                raise ValueError(f"Expected 2D SEED_V raw trial, got {x_trial.shape} for {trial_id}")
            # seedv_preprocess yields [T, C]; align to common [C, T] convention.
            if x_trial.shape[0] > x_trial.shape[1]:
                x_trial = x_trial.T
            if x_trial.shape[0] != 62:
                raise ValueError(f"Expected 62 channels for {trial_id}, got {x_trial.shape}")
            # Downsample 1000 Hz -> 200 Hz so SEED / SEED_IV / SEED_V share the same
            # fixed-seconds window semantics in external MiniROCKET alignment runs.
            if decim > 1:
                x_trial = resample_poly(x_trial, up=1, down=decim, axis=1).astype(np.float32, copy=False)
            self._records.append(
                {
                    "trial_id_str": str(trial_id),
                    "x_trial": x_trial.astype(np.float32, copy=False),
                    "label": int(label),
                    "sfreq": self.sfreq,
                    "subject": str(subject),
                    "session": int(session_1based),
                    "trial": int(trial_idx),
                    "split": "all",
                }
            )

        self._records.sort(key=lambda x: str(x["trial_id_str"]))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Dict]:
        yield from self._records
