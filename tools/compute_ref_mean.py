from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.manifold_streaming_riemann import RiemannianUtils


def _resolve_file_paths(manifest_path: Path) -> List[str]:
    data = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent

    if isinstance(data, dict):
        paths = data.get("file_paths") or data.get("paths") or data.get("files")
        if paths is None:
            raise ValueError("manifest missing file_paths")
        return _resolve_paths(base_dir, paths)

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            if "file_path" in data[0]:
                paths = [item["file_path"] for item in data]
            elif "path" in data[0]:
                paths = [item["path"] for item in data]
            else:
                raise ValueError("manifest list entries missing file_path/path")
            return _resolve_paths(base_dir, paths)
        raise ValueError("manifest list must contain dict entries with file paths")

    raise ValueError("unsupported manifest format")


def _resolve_paths(base_dir: Path, paths: Sequence[str]) -> List[str]:
    resolved = []
    for p in paths:
        p = str(p)
        cand = base_dir / p
        if cand.exists():
            resolved.append(str(cand))
        elif (base_dir.parent / p).exists():
            resolved.append(str(base_dir.parent / p))
        else:
            resolved.append(p)
    return resolved


def _sample_trial_indices(
    n_trials: int,
    *,
    rng: np.random.Generator,
    max_trials: int,
    sample_mode: str,
) -> np.ndarray:
    all_idx = np.arange(n_trials)
    if max_trials <= 0 or max_trials >= n_trials:
        return all_idx
    if sample_mode == "shuffle":
        rng.shuffle(all_idx)
        return all_idx[:max_trials]
    if sample_mode == "stride":
        return np.linspace(0, n_trials - 1, num=max_trials, dtype=int)
    raise ValueError(f"unknown sample_mode: {sample_mode}")


def _normalize_trial_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"expected 4D array, got shape {arr.shape}")
    if arr.shape[1] == 62 and arr.shape[2] == 62:
        return np.transpose(arr, (0, 3, 1, 2))
    if arr.shape[1:] == (5, 62, 62):
        return arr
    raise ValueError(f"unexpected trial shape: {arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--max-windows-per-trial", type=int, default=20)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="shuffle",
        choices=["shuffle", "stride"],
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    file_paths = _resolve_file_paths(manifest_path)
    total_trials = len(file_paths)
    if total_trials == 0:
        raise ValueError("manifest contains no file paths")

    rng = np.random.default_rng(args.seed)
    trial_indices = _sample_trial_indices(
        total_trials,
        rng=rng,
        max_trials=int(args.max_trials),
        sample_mode=args.sample_mode,
    )
    sampled_paths = [file_paths[i] for i in trial_indices]

    covs_by_band: List[List[np.ndarray]] = [[] for _ in range(5)]
    window_counts: List[int] = []

    for path in sampled_paths:
        arr = np.load(path)
        arr = _normalize_trial_array(arr)
        t_len = int(arr.shape[0])
        if t_len == 0:
            continue

        if args.max_windows_per_trial <= 0 or t_len <= args.max_windows_per_trial:
            win_idx = np.arange(t_len)
        else:
            win_idx = rng.choice(t_len, size=int(args.max_windows_per_trial), replace=False)
        window_counts.append(int(len(win_idx)))

        samples = arr[win_idx]  # [N,5,62,62]
        for b in range(5):
            covs_by_band[b].append(samples[:, b, :, :])

    if not window_counts:
        raise ValueError("no windows sampled from trials")

    means: List[torch.Tensor] = []
    for b in range(5):
        if not covs_by_band[b]:
            raise ValueError(f"no covariances collected for band {b}")
        covs = np.concatenate(covs_by_band[b], axis=0)
        covs_t = torch.from_numpy(covs).float()
        mean = RiemannianUtils.cal_riemannian_mean(covs_t)
        means.append(mean.cpu())

    ref_mean = torch.stack(means, dim=0)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref_mean.cpu(), out_path)

    win_min = int(np.min(window_counts))
    win_mean = float(np.mean(window_counts))
    win_max = int(np.max(window_counts))
    print(
        "[ref_mean] "
        f"total_trials={total_trials} sampled_trials={len(sampled_paths)} "
        f"sample_mode={args.sample_mode} max_trials={args.max_trials} "
        f"max_windows_per_trial={args.max_windows_per_trial}"
    )
    print(
        "[ref_mean] "
        f"window_samples min={win_min} mean={win_mean:.2f} max={win_max}"
    )
    print(
        "[ref_mean] "
        f"ref_mean_shape={tuple(ref_mean.shape)} out_path={out_path}"
    )

    log = {
        "manifest_path": str(manifest_path),
        "total_trials": total_trials,
        "sampled_trials": len(sampled_paths),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "max_trials": int(args.max_trials),
        "max_windows_per_trial": int(args.max_windows_per_trial),
        "window_samples": {
            "min": win_min,
            "mean": win_mean,
            "max": win_max,
        },
        "ref_mean_shape": list(ref_mean.shape),
        "out_path": str(out_path),
    }
    log_path = out_path.with_suffix(".meta.json")
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[ref_mean] log_path={log_path}")


if __name__ == "__main__":
    main()
