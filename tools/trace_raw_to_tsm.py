from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from datasets.manifold_streaming_riemann import RiemannianUtils
from manifold_raw.features import bandpass, cov_shrink, parse_band_spec, window_slices
from runners.manifold_raw_v1 import ManifoldRawV1Runner


def _summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _diag_energy_ratio(mat: np.ndarray) -> float:
    total = float(np.sum(mat * mat))
    if total <= 0.0:
        return 0.0
    diag = np.diag(mat)
    diag_energy = float(np.sum(diag * diag))
    return diag_energy / total


def _offdiag_mean_abs(mat: np.ndarray) -> float:
    off = mat - np.diag(np.diag(mat))
    return float(np.mean(np.abs(off)))


def _cov_stats(mat: np.ndarray) -> Dict[str, float]:
    eigvals = np.linalg.eigvalsh(mat)
    eig_min = float(np.min(eigvals))
    eig_max = float(np.max(eigvals))
    trace = float(np.trace(mat))
    diag = np.diag(mat)
    diag_mean = float(np.mean(diag))
    cond = eig_max / max(eig_min, 1e-12)
    return {
        "eig_min": eig_min,
        "eig_max": eig_max,
        "trace": trace,
        "cond": float(cond),
        "diag_energy_ratio": _diag_energy_ratio(mat),
        "offdiag_mean_abs": _offdiag_mean_abs(mat),
        "diag_mean": diag_mean,
    }


def _load_ref_mean(ref_mean_path: Path) -> torch.Tensor:
    ref_mean = torch.load(ref_mean_path, map_location="cpu")
    if isinstance(ref_mean, dict):
        if "ref_mean" in ref_mean:
            ref_mean = ref_mean["ref_mean"]
        elif "mean" in ref_mean:
            ref_mean = ref_mean["mean"]
    ref_mean = torch.as_tensor(ref_mean, dtype=torch.float32)
    if tuple(ref_mean.shape) != (5, 62, 62):
        raise ValueError(f"ref_mean shape must be (5,62,62), got {tuple(ref_mean.shape)}")
    return ref_mean


def _resolve_meta_path(manifest_path: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    name = manifest_path.name
    if name.endswith("_seq_manifest.json"):
        candidate = manifest_path.with_name(name.replace("_seq_manifest.json", "_meta.json"))
        if candidate.exists():
            return candidate
    fallback = Path("logs/seed1_tsm_cov_spd_full_meta.json")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("unable to resolve meta path; use --meta-path")


def _build_trial_rows(
    *,
    raw_manifest: str | None,
    raw_root: str | None,
    raw_backend: str,
) -> List[dict]:
    if raw_manifest is None:
        raise ValueError("meta missing raw_manifest")
    if raw_root is None:
        raise ValueError("meta missing raw_root")
    runner = ManifoldRawV1Runner(
        raw_manifest=raw_manifest,
        seed_raw_root=raw_root,
        raw_backend=raw_backend,
    )
    return runner._load_manifest()


def _resolve_fif_path(
    path: str,
    *,
    raw_backend: str,
    conv_manifest_path: str | None,
) -> str:
    if raw_backend != "fif":
        return path
    if path.lower().endswith(".fif") and Path(path).is_file():
        return path
    if conv_manifest_path and Path(conv_manifest_path).is_file():
        data = json.loads(Path(conv_manifest_path).read_text())
        if isinstance(data, list) and data and isinstance(data[0], dict):
            mapping = {str(row["cnt_path"]): str(row["out_path"]) for row in data if "out_path" in row}
            mapped = mapping.get(path)
            if mapped and Path(mapped).is_file():
                return mapped
    raise FileNotFoundError(f"FIF path not found for raw backend fif: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="seed1")
    parser.add_argument("--effective-config", type=str, required=True)
    parser.add_argument("--ref-mean", type=str, required=True)
    parser.add_argument("--meta-path", type=str, default=None)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--windows-per-trial", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--conv-manifest", type=str, default="logs/seed_raw_fif_manifest.json")
    args = parser.parse_args()

    config_path = Path(args.effective_config)
    effective = json.loads(config_path.read_text())
    manifest_path = Path(effective["manifest_path"])
    meta_path = _resolve_meta_path(manifest_path, args.meta_path)
    meta = json.loads(meta_path.read_text())

    window_sec = float(effective["window_sec"])
    hop_sec = float(effective["hop_sec"])
    cov_method = str(effective["cov_method"])
    spd_eps = float(effective["spd_eps"])
    bands = parse_band_spec(",".join(effective["bands"]))
    raw_manifest = meta.get("raw_manifest")
    raw_root = meta.get("raw_root")
    raw_backend = str(meta.get("raw_backend", "fif")).lower()

    trials = _build_trial_rows(
        raw_manifest=raw_manifest,
        raw_root=raw_root,
        raw_backend=raw_backend,
    )
    if not trials:
        raise ValueError("no trials resolved from manifest")

    rng = np.random.default_rng(int(args.seed))
    pick = rng.choice(len(trials), size=min(int(args.num_trials), len(trials)), replace=False)

    ref_mean = _load_ref_mean(Path(args.ref_mean))

    raw_cache = {}
    trace_rows = []

    raw_var_values = []
    band_var_values = []
    cov_before_diag_ratio = []
    cov_before_offdiag = []
    cov_after_diag_ratio = []
    ratio_eps_domination = []
    tsm_fro_norms = []

    for trial_idx in pick:
        row = trials[int(trial_idx)]
        source_path = row.get("source_cnt_path") or row.get("cnt_path")
        if not source_path:
            raise ValueError("trial row missing source_cnt_path/cnt_path")
        source_path = _resolve_fif_path(
            str(source_path),
            raw_backend=raw_backend,
            conv_manifest_path=args.conv_manifest,
        )
        if source_path not in raw_cache:
            raw = load_one_raw(source_path, backend=raw_backend, preload=False)
            raw62, _meta = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
            raw_cache[source_path] = (raw, raw62)
        raw, raw62 = raw_cache[source_path]

        fs = float(raw62.info.get("sfreq", 0.0))
        t_start = float(row["t_start_s"])
        t_end = float(row["t_end_s"])
        start_idx = int(round(t_start * fs))
        end_idx = int(round(t_end * fs))
        seg = raw62.get_data(start=start_idx, stop=end_idx).astype(np.float32, copy=False)
        n_samples = seg.shape[1]
        win_slices = window_slices(n_samples, fs, window_sec, hop_sec)
        if not win_slices:
            continue

        win_idx = rng.choice(len(win_slices), size=min(int(args.windows_per_trial), len(win_slices)), replace=False)
        win_idx = sorted(int(i) for i in win_idx)

        trial_entry = {
            "trial_index": int(trial_idx),
            "subject": row.get("subject"),
            "session": row.get("session"),
            "trial": row.get("trial"),
            "label": row.get("label"),
            "source_path": source_path,
            "fs": fs,
            "t_start_s": t_start,
            "t_end_s": t_end,
            "window_indices": win_idx,
            "windows": [],
        }

        band_data_cache = {}
        for w_idx in win_idx:
            w_start, w_end = win_slices[w_idx]
            raw_win = seg[:, w_start:w_end]
            raw_var = np.var(raw_win, axis=1)
            raw_var_stats = _summary(raw_var.tolist())
            raw_rms = float(np.sqrt(np.mean(raw_win.astype(np.float64) ** 2)))
            raw_var_values.extend(raw_var.tolist())

            window_entry = {
                "window_index": int(w_idx),
                "w_start": int(w_start),
                "w_end": int(w_end),
                "n_samples": int(w_end - w_start),
                "raw_stats": {
                    "shape": list(raw_win.shape),
                    "channel_var": raw_var_stats,
                    "rms": raw_rms,
                },
                "bands": [],
            }

            for band in bands:
                band_key = f"{band.name}:{band.lo}-{band.hi}"
                if band_key not in band_data_cache:
                    band_data_cache[band_key] = bandpass(seg, fs, band)
                band_data = band_data_cache[band_key]
                band_win = band_data[:, w_start:w_end]
                band_var = np.var(band_win, axis=1)
                band_var_stats = _summary(band_var.tolist())
                band_var_values.extend(band_var.tolist())

                cov_before = cov_shrink(band_win, method=cov_method)
                cov_before_stats = _cov_stats(cov_before)
                cov_before_diag_ratio.append(cov_before_stats["diag_energy_ratio"])
                cov_before_offdiag.append(cov_before_stats["offdiag_mean_abs"])

                cov_after = 0.5 * (cov_before + cov_before.T)
                cov_after = cov_after + np.eye(cov_after.shape[0], dtype=cov_after.dtype) * spd_eps
                cov_after_stats = _cov_stats(cov_after)
                cov_after_diag_ratio.append(cov_after_stats["diag_energy_ratio"])

                diag_mean_before = cov_before_stats["diag_mean"]
                diag_mean_after = cov_after_stats["diag_mean"]
                delta_diag_mean = diag_mean_after - diag_mean_before
                ratio_eps = spd_eps / diag_mean_before if diag_mean_before > 0 else float("inf")
                ratio_eps_domination.append(ratio_eps)

                ref_b = ref_mean[bands.index(band)]
                cov_after_t = torch.as_tensor(cov_after, dtype=torch.float32)
                logm = RiemannianUtils.tangent_space_mapping(cov_after_t, ref_b)
                logm_np = logm.detach().cpu().numpy()
                tsm_stats = {
                    "fro_norm": float(np.linalg.norm(logm_np, ord="fro")),
                    "diag_energy_ratio": _diag_energy_ratio(logm_np),
                    "min": float(np.min(logm_np)),
                    "max": float(np.max(logm_np)),
                    "std": float(np.std(logm_np)),
                }
                tsm_fro_norms.append(tsm_stats["fro_norm"])

                cov_chan = np.cov(band_win, rowvar=True, bias=False)
                cov_time = np.cov(band_win, rowvar=False, bias=False)
                axis_check = {
                    "band_win_shape": list(band_win.shape),
                    "cov_chan_shape": list(cov_chan.shape),
                    "cov_time_shape": list(cov_time.shape),
                    "cov_shrink_input_shape": list(band_win.shape),
                    "cov_shrink_internal_X_shape": [int(band_win.shape[1]), int(band_win.shape[0])],
                }

                band_entry = {
                    "band": band_key,
                    "band_stats": {
                        "shape": list(band_win.shape),
                        "channel_var": band_var_stats,
                    },
                    "cov_before": cov_before_stats,
                    "cov_after": {
                        **cov_after_stats,
                        "delta_diag_mean": float(delta_diag_mean),
                        "ratio_eps_domination": float(ratio_eps),
                    },
                    "tsm": tsm_stats,
                    "axis_check": axis_check,
                }
                window_entry["bands"].append(band_entry)

            trial_entry["windows"].append(window_entry)

        trace_rows.append(trial_entry)

    for raw, raw62 in raw_cache.values():
        if hasattr(raw, "close"):
            raw.close()

    summary = {
        "raw_var": _summary(raw_var_values),
        "band_var": _summary(band_var_values),
        "cov_before_diag_energy_ratio": _summary(cov_before_diag_ratio),
        "cov_before_offdiag_mean_abs": _summary(cov_before_offdiag),
        "cov_after_diag_energy_ratio": _summary(cov_after_diag_ratio),
        "ratio_eps_domination": _summary(ratio_eps_domination),
        "tsm_fro_norm": _summary(tsm_fro_norms),
    }

    conclusions = []
    band_var_median = summary.get("band_var", {}).get("median")
    if band_var_median is not None and band_var_median < 1e-8:
        conclusions.append(
            "bandpass variance median << 1e-5 -> scale issue, eps likely dominates"
        )
    offdiag_median = summary.get("cov_before_offdiag_mean_abs", {}).get("median")
    diag_ratio_median = summary.get("cov_before_diag_energy_ratio", {}).get("median")
    if offdiag_median is not None and diag_ratio_median is not None:
        if offdiag_median < 1e-10 and diag_ratio_median > 0.98:
            conclusions.append(
                "cov_before already near-diagonal -> covariance/axis issue or near-constant input"
            )
    if (
        diag_ratio_median is not None
        and summary.get("cov_after_diag_energy_ratio", {}).get("median", 0.0) > 0.98
        and diag_ratio_median < 0.95
    ):
        conclusions.append("cov_after degrades vs cov_before -> eps/symm dominates")

    report = {
        "dataset": args.dataset,
        "effective_config_path": str(config_path),
        "meta_path": str(meta_path),
        "manifest_path": str(manifest_path),
        "ref_mean_path": str(args.ref_mean),
        "raw_manifest": raw_manifest,
        "raw_root": raw_root,
        "raw_backend": raw_backend,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "cov_method": cov_method,
        "spd_eps": spd_eps,
        "bands": [f"{b.name}:{b.lo}-{b.hi}" for b in bands],
        "summary": summary,
        "conclusions": conclusions,
        "trials": trace_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("[trace] raw_var median:", summary.get("raw_var", {}).get("median"))
    print("[trace] band_var median:", summary.get("band_var", {}).get("median"))
    print(
        "[trace] cov_before diag_energy_ratio median:",
        summary.get("cov_before_diag_energy_ratio", {}).get("median"),
    )
    print(
        "[trace] cov_before offdiag_mean_abs median:",
        summary.get("cov_before_offdiag_mean_abs", {}).get("median"),
    )
    print(
        "[trace] cov_after diag_energy_ratio median:",
        summary.get("cov_after_diag_energy_ratio", {}).get("median"),
    )
    print("[trace] ratio_eps_domination median:", summary.get("ratio_eps_domination", {}).get("median"))
    print("[trace] tsm fro_norm median:", summary.get("tsm_fro_norm", {}).get("median"))
    if conclusions:
        print("[trace] conclusion:", "; ".join(conclusions))
    else:
        print("[trace] conclusion: no clear degradation point detected")
    print("[trace] out_path:", out_path)


if __name__ == "__main__":
    main()
