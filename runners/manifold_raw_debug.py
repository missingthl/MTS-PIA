from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from datasets.seed_raw import RawTrialBatch, load_seed1_raw_trials


def _hash_channels(names: list[str]) -> str:
    joined = ",".join(names).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()


def _format_array_stats(X: np.ndarray) -> str:
    X = np.asarray(X, dtype=np.float64)
    return (
        f"shape={X.shape} dtype={X.dtype} "
        f"min={float(np.nanmin(X)):.6f} max={float(np.nanmax(X)):.6f} "
        f"mean={float(np.nanmean(X)):.6f} std={float(np.nanstd(X)):.6f}"
    )


@dataclass
class ManifoldRawDebugRunner:
    debug_trials: int = 1

    def run(
        self,
        *,
        seed_raw_root: str,
        seed_raw_fs: Optional[int],
        channel_policy: str,
        locs_path: str,
        raw_backend: str = "cnt",
    ) -> str:
        batch: RawTrialBatch = load_seed1_raw_trials(
            seed_raw_root,
            fs=seed_raw_fs,
            channel_policy=channel_policy,
            locs_path=locs_path,
            debug_trials=self.debug_trials,
            max_cnt_files=1,
            raw_backend=raw_backend,
        )

        labels_unique = sorted(set(batch.labels_all.tolist()))
        channel_hash = _hash_channels(batch.channel_names)

        print(f"[seed1][raw] raw_root={batch.raw_root}")
        print(f"[seed1][raw] fs={batch.fs}")
        print(f"[seed1][raw] n_trials={len(batch.manifest)}")
        print(f"[seed1][raw] labels_unique={labels_unique}")
        print(f"[seed1][raw] channel_hash={channel_hash}")
        print(f"[seed1][raw] channel_names_head={batch.channel_names[:10]}")

        for i, (X, meta) in enumerate(zip(batch.X, batch.meta)):
            if i >= self.debug_trials:
                break
            print(f"[seed1][raw][trial{i}] {_format_array_stats(X)}")
            print(
                f"[seed1][raw][trial{i}] meta=subject={meta['subject']} "
                f"session={meta['session']} trial_id={meta['trial_id']} "
                f"label={meta['label']} len_T={meta['len_T']}"
            )

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = log_dir / f"seed1_raw_manifest_{ts}.json"
        manifest = {
            "dataset": "seed1",
            "raw_root": batch.raw_root,
            "fs": batch.fs,
            "n_trials": len(batch.manifest),
            "cnt_files_used": 1,
            "channel_hash": channel_hash,
            "channel_names": batch.channel_names,
            "session_inference": "cnt_filename (subject_session.cnt)",
            "trials": batch.manifest,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[seed1][raw] manifest_json={out_path}")
        return str(out_path)
