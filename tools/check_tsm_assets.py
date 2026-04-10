from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.manifold_streaming_riemann import (
    ManifoldStreamingDataset,
    collate_fn_pad,
)


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


def _normalize_trial_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"expected 4D array, got shape {arr.shape}")
    if arr.shape[1] == 62 and arr.shape[2] == 62:
        return np.transpose(arr, (0, 3, 1, 2))
    if arr.shape[1:] == (5, 62, 62):
        return arr
    raise ValueError(f"unexpected trial shape: {arr.shape}")


def _load_ref_mean(path: Path) -> torch.Tensor:
    ref_mean = torch.load(path, map_location="cpu")
    if isinstance(ref_mean, dict):
        if "ref_mean" in ref_mean:
            ref_mean = ref_mean["ref_mean"]
        elif "mean" in ref_mean:
            ref_mean = ref_mean["mean"]
        else:
            raise ValueError("ref_mean dict missing ref_mean/mean keys")
    ref_mean = torch.as_tensor(ref_mean, dtype=torch.float32)
    return ref_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--ref-mean-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    file_paths = _resolve_file_paths(manifest_path)
    if not file_paths:
        raise ValueError("manifest contains no file paths")

    rng = np.random.default_rng(args.seed)
    sample_idx = int(rng.integers(len(file_paths)))
    sample_path = Path(file_paths[sample_idx])
    if not sample_path.is_file():
        raise FileNotFoundError(f"sample trial not found: {sample_path}")

    arr = np.load(sample_path)
    print(f"[tsm_check] sample_path={sample_path}")
    print(f"[tsm_check] sample_shape_raw={arr.shape}")
    arr = _normalize_trial_array(arr)
    print(f"[tsm_check] sample_shape_norm={arr.shape}")

    t_len = arr.shape[0]
    band_idx = int(rng.integers(5))
    win_idx = int(rng.integers(t_len))
    cov = arr[win_idx, band_idx]
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(eigvals.min())
    print(
        f"[tsm_check] sample_band={band_idx} sample_window={win_idx} min_eig={min_eig:.6e}"
    )

    ref_mean_path = Path(args.ref_mean_path)
    if not ref_mean_path.is_file():
        raise FileNotFoundError(f"ref_mean not found: {ref_mean_path}")
    ref_mean = _load_ref_mean(ref_mean_path)
    print(f"[tsm_check] ref_mean_path={ref_mean_path}")
    print(f"[tsm_check] ref_mean_shape={tuple(ref_mean.shape)}")
    if tuple(ref_mean.shape) != (5, 62, 62):
        raise ValueError(f"ref_mean shape must be (5,62,62), got {tuple(ref_mean.shape)}")

    ds = ManifoldStreamingDataset(str(manifest_path), reference_mean=ref_mean)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_fn_pad)
    x, y, mask = next(iter(dl))
    padding_true = int(mask.sum().item())
    any_nan = bool(torch.isnan(x).any().item())
    all_finite = bool(torch.isfinite(x).all().item())

    print(f"[tsm_check] batch_x_shape={tuple(x.shape)}")
    print(f"[tsm_check] batch_y_shape={tuple(y.shape)}")
    print(f"[tsm_check] batch_mask_shape={tuple(mask.shape)}")
    print(f"[tsm_check] padding_true={padding_true}")
    print(f"[tsm_check] any_nan={any_nan} all_finite={all_finite}")


if __name__ == "__main__":
    main()
