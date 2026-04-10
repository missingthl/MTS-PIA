from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.manifold_streaming_riemann import RiemannianUtils


def _resolve_manifest_paths(manifest_path: Path) -> List[str]:
    data = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent

    if isinstance(data, dict):
        paths = data.get("file_paths") or data.get("paths") or data.get("files")
        if paths is None:
            raise ValueError("manifest missing file_paths")
        return _resolve_paths(base_dir, paths)

    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], str):
            return _resolve_paths(base_dir, data)
        if isinstance(data[0], dict):
            if "file_path" in data[0]:
                paths = [item["file_path"] for item in data]
            elif "path" in data[0]:
                paths = [item["path"] for item in data]
            else:
                raise ValueError("manifest list entries missing file_path/path")
            return _resolve_paths(base_dir, paths)
    raise ValueError("unsupported manifest format")


def _resolve_paths(base_dir: Path, paths: List[str]) -> List[str]:
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


def _load_ref_mean(ref_path: Path) -> torch.Tensor:
    ref = torch.load(ref_path, map_location="cpu")
    if isinstance(ref, dict):
        if "ref_mean" in ref:
            ref = ref["ref_mean"]
        elif "mean" in ref:
            ref = ref["mean"]
    ref = torch.as_tensor(ref, dtype=torch.float32)
    if tuple(ref.shape) != (5, 62, 62):
        raise ValueError(f"ref_mean shape must be (5,62,62), got {tuple(ref.shape)}")
    return ref


def _symm(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + np.swapaxes(mat, -1, -2))


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


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"p5": math.nan, "p50": math.nan, "p95": math.nan}
    q = np.percentile(values, [5, 50, 95])
    return {"p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def _stats_tsm(tsm: np.ndarray) -> Dict[str, object]:
    if tsm.ndim != 3:
        raise ValueError(f"tsm must be 3D [N,62,62], got {tsm.shape}")
    n = tsm.shape[0]
    flat = tsm.reshape(n, -1)
    fro = np.linalg.norm(flat, axis=1)
    val_min = tsm.min(axis=(1, 2))
    val_max = tsm.max(axis=(1, 2))
    diag_energy = np.asarray([_diag_energy_ratio(tsm[i]) for i in range(n)], dtype=np.float64)
    offdiag = np.asarray([_offdiag_mean_abs(tsm[i]) for i in range(n)], dtype=np.float64)
    stats = {
        "fro_norm": _quantiles(fro),
        "diag_energy_ratio": _quantiles(diag_energy),
        "offdiag_mean_abs": _quantiles(offdiag),
        "value_min": _quantiles(val_min),
        "value_max": _quantiles(val_max),
    }
    return stats


def _cosine_pairs(flat: np.ndarray, *, rng: np.random.Generator, max_pairs: int) -> Dict[str, float]:
    if flat.ndim != 2:
        raise ValueError(f"flat must be 2D, got {flat.shape}")
    n = flat.shape[0]
    if n < 2:
        return {"mean": math.nan, "std": math.nan, "p5": math.nan, "p50": math.nan, "p95": math.nan, "count": 0}
    norms = np.linalg.norm(flat, axis=1)
    pairs = []
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 10:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            attempts += 1
            continue
        if norms[i] == 0.0 or norms[j] == 0.0:
            attempts += 1
            continue
        pairs.append((i, j))
        attempts += 1
    if not pairs:
        return {"mean": math.nan, "std": math.nan, "p5": math.nan, "p50": math.nan, "p95": math.nan, "count": 0}
    cos_vals = []
    for i, j in pairs:
        cos = float(np.dot(flat[i], flat[j]) / (norms[i] * norms[j]))
        cos_vals.append(cos)
    arr = np.asarray(cos_vals, dtype=np.float64)
    q = np.percentile(arr, [5, 50, 95])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
        "count": int(arr.size),
    }


def _collect_windows(
    *,
    paths: List[str],
    num_trials: int,
    max_windows_per_trial: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[int]]:
    rng = np.random.default_rng(int(seed))
    idx_all = np.arange(len(paths))
    if len(idx_all) == 0:
        return [], []
    if num_trials > 0 and num_trials < len(idx_all):
        rng.shuffle(idx_all)
        idx_all = idx_all[:num_trials]
    windows = []
    lengths = []
    for idx in idx_all:
        arr = np.load(paths[int(idx)])  # [T,5,62,62] or [T,62,62,5]
        if arr.ndim != 4:
            raise ValueError(f"expected 4D array, got shape {arr.shape}")
        if arr.shape[1] == 62 and arr.shape[2] == 62:
            arr = np.transpose(arr, (0, 3, 1, 2))
        if arr.shape[1:] != (5, 62, 62):
            raise ValueError(f"bad shape after transpose: {arr.shape}")
        t_len = int(arr.shape[0])
        if t_len == 0:
            continue
        if max_windows_per_trial <= 0 or t_len <= max_windows_per_trial:
            pick = np.arange(t_len)
        else:
            pick = rng.choice(t_len, size=int(max_windows_per_trial), replace=False)
        pick = np.asarray(pick, dtype=int)
        lengths.append(int(pick.size))
        windows.append(arr[pick])  # [K,5,62,62]
    return windows, lengths


def _tsm_from_cov(
    covs: np.ndarray,
    ref: torch.Tensor,
    *,
    debug_dtype: bool,
) -> np.ndarray:
    cov_t = torch.as_tensor(covs, dtype=torch.float32)
    tsm = RiemannianUtils.tangent_space_mapping(cov_t, ref, debug_dtype=debug_dtype)
    return tsm.cpu().numpy()


def _audit_world(
    *,
    cov_windows: List[np.ndarray],
    ref_mean: torch.Tensor,
    abs_eps_sim: float,
    mode: str,
    rng: np.random.Generator,
    debug_dtype: bool,
    max_pairs: int,
) -> Dict[str, object]:
    per_band: Dict[str, Dict[str, object]] = {}
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    all_metrics = {"fro_norm": [], "diag_energy_ratio": [], "offdiag_mean_abs": [], "value_min": [], "value_max": []}
    nan_count = 0
    inf_count = 0

    for b_idx, band in enumerate(band_names):
        cov_list = []
        for win in cov_windows:
            cov_list.append(win[:, b_idx])
        if not cov_list:
            continue
        covs = np.concatenate(cov_list, axis=0)
        covs = _symm(covs)
        ref = ref_mean[b_idx].cpu().numpy()
        ref = _symm(ref)

        if mode == "old":
            covs = covs + np.eye(covs.shape[-1], dtype=covs.dtype) * float(abs_eps_sim)
            ref = ref + np.eye(ref.shape[-1], dtype=ref.dtype) * float(abs_eps_sim)

        tsm = _tsm_from_cov(covs, torch.as_tensor(ref, dtype=torch.float32), debug_dtype=debug_dtype)
        flat = tsm.reshape(tsm.shape[0], -1)

        nan_count += int(np.isnan(tsm).sum())
        inf_count += int(np.isinf(tsm).sum())

        stats = _stats_tsm(tsm)
        cos = _cosine_pairs(flat, rng=rng, max_pairs=max_pairs)

        per_band[band] = {
            "n_samples": int(tsm.shape[0]),
            "stats": stats,
            "cosine": cos,
        }
        all_metrics["fro_norm"].extend(list(np.linalg.norm(flat, axis=1)))
        all_metrics["diag_energy_ratio"].extend(list(np.asarray([_diag_energy_ratio(tsm[i]) for i in range(tsm.shape[0])])))
        all_metrics["offdiag_mean_abs"].extend(list(np.asarray([_offdiag_mean_abs(tsm[i]) for i in range(tsm.shape[0])])))
        all_metrics["value_min"].extend(list(tsm.min(axis=(1, 2))))
        all_metrics["value_max"].extend(list(tsm.max(axis=(1, 2))))

    global_stats = {k: _quantiles(np.asarray(v, dtype=np.float64)) for k, v in all_metrics.items()}
    return {
        "per_band": per_band,
        "global": global_stats,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def _print_summary(tag: str, band: str, stats: Dict[str, object]) -> None:
    b = stats["per_band"][band]["stats"]
    fro = b["fro_norm"]
    off = b["offdiag_mean_abs"]
    print(
        f"[{tag}] {band}: fro_norm p50={fro['p50']:.6e} p95={fro['p95']:.6e}, "
        f"offdiag_mean_abs p50={off['p50']:.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--ref-mean", type=str, required=True)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--max-windows-per-trial", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abs-eps-sim", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-cos-pairs", type=int, default=500)
    parser.add_argument("--debug-dtype", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_paths(Path.cwd(), [args.manifest])[0]
    ref_path = _resolve_paths(Path.cwd(), [args.ref_mean])[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _resolve_manifest_paths(Path(manifest_path))
    if not paths:
        raise ValueError("manifest resolved to zero paths")
    ref_mean = _load_ref_mean(Path(ref_path))

    windows, lengths = _collect_windows(
        paths=paths,
        num_trials=int(args.num_trials),
        max_windows_per_trial=int(args.max_windows_per_trial),
        seed=int(args.seed),
    )
    if not windows:
        raise ValueError("no windows collected from manifest")

    rng = np.random.default_rng(int(args.seed))
    new_stats = _audit_world(
        cov_windows=windows,
        ref_mean=ref_mean,
        abs_eps_sim=float(args.abs_eps_sim),
        mode="new",
        rng=rng,
        debug_dtype=bool(args.debug_dtype),
        max_pairs=int(args.max_cos_pairs),
    )
    old_stats = _audit_world(
        cov_windows=windows,
        ref_mean=ref_mean,
        abs_eps_sim=float(args.abs_eps_sim),
        mode="old",
        rng=rng,
        debug_dtype=bool(args.debug_dtype),
        max_pairs=int(args.max_cos_pairs),
    )

    new_path = out_dir / "audit_tsm_new_relative64.json"
    old_path = out_dir / f"audit_tsm_oldsim_abs{args.abs_eps_sim:.0e}_64.json"
    with new_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(manifest_path),
                "ref_mean": str(ref_path),
                "num_trials": int(args.num_trials),
                "max_windows_per_trial": int(args.max_windows_per_trial),
                "seed": int(args.seed),
                "mode": "new",
                "stats": new_stats,
                "window_samples": {
                    "min": int(min(lengths)),
                    "mean": float(np.mean(lengths)),
                    "max": int(max(lengths)),
                },
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    with old_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(manifest_path),
                "ref_mean": str(ref_path),
                "num_trials": int(args.num_trials),
                "max_windows_per_trial": int(args.max_windows_per_trial),
                "seed": int(args.seed),
                "mode": "old_sim",
                "abs_eps_sim": float(args.abs_eps_sim),
                "stats": old_stats,
                "window_samples": {
                    "min": int(min(lengths)),
                    "mean": float(np.mean(lengths)),
                    "max": int(max(lengths)),
                },
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    _print_summary("new", "delta", new_stats)
    _print_summary("old", "delta", old_stats)
    _print_summary("new", "gamma", new_stats)
    _print_summary("old", "gamma", old_stats)

    ratio_fro_delta = (
        new_stats["per_band"]["delta"]["stats"]["fro_norm"]["p50"]
        / max(old_stats["per_band"]["delta"]["stats"]["fro_norm"]["p50"], 1e-30)
    )
    ratio_fro_gamma = (
        new_stats["per_band"]["gamma"]["stats"]["fro_norm"]["p50"]
        / max(old_stats["per_band"]["gamma"]["stats"]["fro_norm"]["p50"], 1e-30)
    )
    ratio_off_delta = (
        new_stats["per_band"]["delta"]["stats"]["offdiag_mean_abs"]["p50"]
        / max(old_stats["per_band"]["delta"]["stats"]["offdiag_mean_abs"]["p50"], 1e-30)
    )
    ratio_off_gamma = (
        new_stats["per_band"]["gamma"]["stats"]["offdiag_mean_abs"]["p50"]
        / max(old_stats["per_band"]["gamma"]["stats"]["offdiag_mean_abs"]["p50"], 1e-30)
    )
    print(f"[compare] ratio_fro_p50 delta={ratio_fro_delta:.3f}")
    print(f"[compare] ratio_fro_p50 gamma={ratio_fro_gamma:.3f}")
    print(f"[compare] ratio_offdiag_p50 delta={ratio_off_delta:.3f}")
    print(f"[compare] ratio_offdiag_p50 gamma={ratio_off_gamma:.3f}")

    pass_flag = (
        (ratio_fro_delta >= 10.0 and ratio_off_delta >= 10.0)
        or (ratio_fro_gamma >= 10.0 and ratio_off_gamma >= 10.0)
    )
    verdict = "PASS" if pass_flag else "FAIL"
    print(f"[verdict] {verdict}")

    report_path = out_dir / "tsm_audit_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# TSM Audit Report (float64 protected)\n\n")
        f.write(f"- manifest: {manifest_path}\n")
        f.write(f"- ref_mean: {ref_path}\n")
        f.write(f"- abs_eps_sim: {args.abs_eps_sim:.0e}\n")
        f.write(f"- num_trials: {args.num_trials}\n")
        f.write(f"- max_windows_per_trial: {args.max_windows_per_trial}\n")
        f.write(f"- verdict: {verdict}\n\n")
        f.write("## Delta/Gamma Summary\n")
        f.write(f"- new delta fro p50: {new_stats['per_band']['delta']['stats']['fro_norm']['p50']:.6e}\n")
        f.write(f"- old delta fro p50: {old_stats['per_band']['delta']['stats']['fro_norm']['p50']:.6e}\n")
        f.write(f"- new gamma fro p50: {new_stats['per_band']['gamma']['stats']['fro_norm']['p50']:.6e}\n")
        f.write(f"- old gamma fro p50: {old_stats['per_band']['gamma']['stats']['fro_norm']['p50']:.6e}\n")
        f.write(f"- ratio_fro_p50 delta: {ratio_fro_delta:.3f}\n")
        f.write(f"- ratio_fro_p50 gamma: {ratio_fro_gamma:.3f}\n")
        f.write(f"- ratio_offdiag_p50 delta: {ratio_off_delta:.3f}\n")
        f.write(f"- ratio_offdiag_p50 gamma: {ratio_off_gamma:.3f}\n\n")
        f.write("## NaN/Inf Counts\n")
        f.write(f"- new nan/inf: {new_stats['nan_count']}/{new_stats['inf_count']}\n")
        f.write(f"- old nan/inf: {old_stats['nan_count']}/{old_stats['inf_count']}\n")

    print(f"[output] new_json={new_path}")
    print(f"[output] old_json={old_path}")
    print(f"[output] report={report_path}")


if __name__ == "__main__":
    main()
