from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.manifold_streaming_riemann import RiemannianUtils
from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from manifold_raw.features import bandpass, cov_shrink, parse_band_spec, window_slices
from manifold_raw.spd_eps import compute_spd_eps


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _eps_stats(values: List[float]) -> Dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "p50": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _alpha_tag(alpha: float) -> str:
    return f"{alpha:.0e}".replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnt",
        type=str,
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/1_1.cnt",
    )
    parser.add_argument("--sec", type=float, default=10.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument(
        "--bands",
        type=str,
        default="delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    )
    parser.add_argument("--cov-method", type=str, default="shrinkage_oas")
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=str, default=None)
    parser.add_argument("--locs", type=str, default="data/SEED/channel_62_pos.locs")
    parser.add_argument("--data-format", type=str, default=None)
    parser.add_argument("--spd-eps", type=float, default=1e-5)
    parser.add_argument(
        "--spd-eps-mode",
        type=str,
        default="relative_trace",
        choices=["absolute", "relative_trace", "relative_diag"],
    )
    parser.add_argument("--spd-eps-alpha", type=float, default=1e-2)
    parser.add_argument("--spd-eps-floor-mult", type=float, default=1e-6)
    parser.add_argument("--spd-eps-ceil-mult", type=float, default=1e-1)
    args = parser.parse_args()

    cnt_path = _resolve_path(args.cnt)
    locs_path = _resolve_path(args.locs)

    alpha_tag = _alpha_tag(float(args.spd_eps_alpha))
    eps_tag = f"{args.spd_eps_mode}_a{alpha_tag}"
    if args.out_prefix:
        out_prefix = _resolve_path(args.out_prefix)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_prefix = _resolve_path(f"logs/min_cov_spd_{eps_tag}_{ts}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    raw = load_one_raw(str(cnt_path), backend="cnt", preload=False, data_format=args.data_format)
    raw62, _meta = build_eeg62_view(raw, locs_path=str(locs_path))
    sfreq = float(raw62.info.get("sfreq", 0.0))
    stop = int(round(float(args.sec) * sfreq)) if sfreq > 0 else 0
    if stop <= 0:
        raise ValueError(f"Invalid stop={stop}; check sec={args.sec} sfreq={sfreq}")

    seg = raw62.get_data(start=0, stop=stop).astype(np.float64, copy=False)
    win_slices = window_slices(seg.shape[1], sfreq, float(args.window_sec), float(args.hop_sec))
    if not win_slices:
        raise ValueError("window_slices returned empty list")

    rng = np.random.default_rng(int(args.seed))
    n_pick = min(int(args.num_windows), len(win_slices))
    pick_idx = rng.choice(len(win_slices), size=n_pick, replace=False)
    pick_idx = sorted(int(i) for i in pick_idx)
    picked = [win_slices[i] for i in pick_idx]

    bands = parse_band_spec(args.bands)
    band_full = {band.name: bandpass(seg, sfreq, band).astype(np.float64, copy=False) for band in bands}

    covs_by_band: Dict[str, List[np.ndarray]] = {band.name: [] for band in bands}
    eps_by_band: Dict[str, List[float]] = {band.name: [] for band in bands}

    for w_start, w_end in picked:
        for band in bands:
            data = band_full[band.name][:, w_start:w_end]
            cov = cov_shrink(data, method=args.cov_method)
            cov = 0.5 * (cov + cov.T)
            eps_val, _base = compute_spd_eps(
                cov,
                mode=args.spd_eps_mode,
                absolute=float(args.spd_eps),
                alpha=float(args.spd_eps_alpha),
                floor_mult=float(args.spd_eps_floor_mult),
                ceil_mult=float(args.spd_eps_ceil_mult),
            )
            eps_by_band[band.name].append(float(eps_val))
            cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * float(eps_val)
            covs_by_band[band.name].append(cov.astype(np.float32, copy=False))

    n_windows = len(picked)
    n_bands = len(bands)
    cov_seq = np.empty((n_windows, n_bands, seg.shape[0], seg.shape[0]), dtype=np.float32)
    for b_idx, band in enumerate(bands):
        cov_seq[:, b_idx] = np.stack(covs_by_band[band.name], axis=0)

    cov_path = out_prefix.with_name(f"{out_prefix.name}_cov_spd_{eps_tag}.npy")
    np.save(cov_path, cov_seq)

    manifest_path = out_prefix.with_name(f"{out_prefix.name}_seq_manifest.json")
    manifest = {"file_paths": [cov_path.name]}
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    means = []
    for b_idx in range(n_bands):
        covs = cov_seq[:, b_idx]
        covs_t = torch.from_numpy(covs).float()
        mean = RiemannianUtils.cal_riemannian_mean(covs_t)
        means.append(mean.cpu())
    ref_mean = torch.stack(means, dim=0)

    ref_mean_path = out_prefix.with_name(f"{out_prefix.name}_ref_mean_{eps_tag}.pt")
    torch.save(ref_mean, ref_mean_path)

    eps_meta = {
        "cnt_path": str(cnt_path),
        "locs_path": str(locs_path),
        "sec": float(args.sec),
        "sfreq": float(sfreq),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "window_indices": pick_idx,
        "bands": [b.name for b in bands],
        "cov_method": str(args.cov_method),
        "spd_eps": float(args.spd_eps),
        "spd_eps_mode": str(args.spd_eps_mode),
        "spd_eps_alpha": float(args.spd_eps_alpha),
        "spd_eps_floor_mult": float(args.spd_eps_floor_mult),
        "spd_eps_ceil_mult": float(args.spd_eps_ceil_mult),
        "eps_stats": {name: _eps_stats(vals) for name, vals in eps_by_band.items()},
        "cov_path": str(cov_path),
        "manifest_path": str(manifest_path),
        "ref_mean_path": str(ref_mean_path),
    }
    eps_meta_path = out_prefix.with_name(f"{out_prefix.name}_eps.meta.json")
    with eps_meta_path.open("w", encoding="utf-8") as f:
        json.dump(eps_meta, f, ensure_ascii=True, indent=2)

    print(f"[export] cov_path={cov_path}")
    print(f"[export] manifest_path={manifest_path}")
    print(f"[export] ref_mean_path={ref_mean_path}")
    print(f"[export] eps_meta_path={eps_meta_path}")
    for band in bands:
        stats = eps_meta["eps_stats"][band.name]
        print(
            f"[export] band={band.name} eps p50={stats['p50']:.6e} "
            f"p95={stats['p95']:.6e}"
        )

    if hasattr(raw, "close"):
        raw.close()
    if hasattr(raw62, "close"):
        raw62.close()


if __name__ == "__main__":
    main()
