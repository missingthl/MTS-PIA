from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, ensure_ascii=False, indent=2)


def _rss_vms_gb() -> Dict[str, float]:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {
            "rss_gb": float(mem.rss) / (1024**3),
            "vms_gb": float(mem.vms) / (1024**3),
        }
    except Exception:
        return {
            "rss_gb": float("nan"),
            "vms_gb": float("nan"),
        }


def _cuda_stats_gb() -> Dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "cuda_alloc_gb": 0.0,
                "cuda_reserved_gb": 0.0,
                "cuda_max_alloc_gb": 0.0,
                "cuda_max_reserved_gb": 0.0,
            }
        return {
            "cuda_alloc_gb": float(torch.cuda.memory_allocated()) / (1024**3),
            "cuda_reserved_gb": float(torch.cuda.memory_reserved()) / (1024**3),
            "cuda_max_alloc_gb": float(torch.cuda.max_memory_allocated()) / (1024**3),
            "cuda_max_reserved_gb": float(torch.cuda.max_memory_reserved()) / (1024**3),
        }
    except Exception:
        return {
            "cuda_alloc_gb": float("nan"),
            "cuda_reserved_gb": float("nan"),
            "cuda_max_alloc_gb": float("nan"),
            "cuda_max_reserved_gb": float("nan"),
        }


@dataclass
class ResourceProbeLogger:
    out_dir: str
    summary_path: str = field(init=False)
    csv_path: str = field(init=False)
    _peak_rss_gb: float = field(default=0.0, init=False)
    _stage_starts: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        _ensure_dir(self.out_dir)
        self.summary_path = os.path.join(self.out_dir, "resource_summary.json")
        self.csv_path = os.path.join(self.out_dir, "resource_stage_log.csv")
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "event",
                        "stage",
                        "elapsed_sec",
                        "rss_gb",
                        "peak_rss_gb",
                        "vms_gb",
                        "cuda_alloc_gb",
                        "cuda_reserved_gb",
                        "cuda_max_alloc_gb",
                        "cuda_max_reserved_gb",
                        "note",
                    ],
                )
                writer.writeheader()

    def _snapshot(self) -> Dict[str, float]:
        mem = _rss_vms_gb()
        cuda = _cuda_stats_gb()
        rss = mem.get("rss_gb", float("nan"))
        if np.isfinite(rss):
            self._peak_rss_gb = max(self._peak_rss_gb, float(rss))
        snap = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rss_gb": float(mem["rss_gb"]),
            "peak_rss_gb": float(self._peak_rss_gb),
            "vms_gb": float(mem["vms_gb"]),
            "cuda_alloc_gb": float(cuda["cuda_alloc_gb"]),
            "cuda_reserved_gb": float(cuda["cuda_reserved_gb"]),
            "cuda_max_alloc_gb": float(cuda["cuda_max_alloc_gb"]),
            "cuda_max_reserved_gb": float(cuda["cuda_max_reserved_gb"]),
        }
        return snap

    def _append_row(self, row: Dict[str, object]) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "event",
                    "stage",
                    "elapsed_sec",
                    "rss_gb",
                    "peak_rss_gb",
                    "vms_gb",
                    "cuda_alloc_gb",
                    "cuda_reserved_gb",
                    "cuda_max_alloc_gb",
                    "cuda_max_reserved_gb",
                    "note",
                ],
            )
            writer.writerow(row)
            f.flush()

    def mark_stage_start(self, stage: str, note: str = "") -> None:
        self._stage_starts[str(stage)] = time.perf_counter()
        snap = self._snapshot()
        self._append_row(
            {
                **snap,
                "event": "start",
                "stage": str(stage),
                "elapsed_sec": 0.0,
                "note": str(note),
            }
        )
        self.write_summary(
            status="running",
            current_stage=str(stage),
            last_completed_stage=self._read_last_completed_stage(),
            failed_stage=None,
            exit_code=None,
        )

    def mark_stage_end(self, stage: str, note: str = "") -> None:
        t0 = self._stage_starts.get(str(stage), time.perf_counter())
        elapsed = float(time.perf_counter() - t0)
        snap = self._snapshot()
        self._append_row(
            {
                **snap,
                "event": "end",
                "stage": str(stage),
                "elapsed_sec": elapsed,
                "note": str(note),
            }
        )
        self.write_summary(
            status="running",
            current_stage=None,
            last_completed_stage=str(stage),
            failed_stage=None,
            exit_code=None,
        )

    def mark_failure(self, stage: str, exc: Exception) -> None:
        snap = self._snapshot()
        self._append_row(
            {
                **snap,
                "event": "failure",
                "stage": str(stage),
                "elapsed_sec": 0.0,
                "note": f"{type(exc).__name__}: {exc}",
            }
        )
        self.write_summary(
            status="failed",
            current_stage=str(stage),
            last_completed_stage=self._read_last_completed_stage(),
            failed_stage=str(stage),
            exit_code=1,
            failure_note=f"{type(exc).__name__}: {exc}",
        )

    def mark_success(self) -> None:
        self.write_summary(
            status="success",
            current_stage=None,
            last_completed_stage=self._read_last_completed_stage(),
            failed_stage=None,
            exit_code=0,
        )

    def _read_last_completed_stage(self) -> Optional[str]:
        last_stage = None
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row.get("event") == "end":
                    last_stage = row.get("stage")
        except Exception:
            pass
        return last_stage

    def write_summary(self, **kwargs) -> None:
        snap = self._snapshot()
        payload = {**snap, **kwargs}
        _write_json(self.summary_path, payload)
