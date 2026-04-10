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


def _load_manifest(manifest_path: Path) -> Tuple[List[str], List[int]]:
    data = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent
    if isinstance(data, dict):
        paths = data.get("file_paths") or data.get("paths") or data.get("files")
        if paths is None:
            raise ValueError("manifest missing file_paths")
        if "labels" in data:
            labels = [int(v) for v in data["labels"]]
        elif "trials" in data:
            labels = [int(t["label"]) for t in data["trials"]]
        else:
            raise ValueError("manifest missing labels/trials")
        return _resolve_paths(base_dir, paths), labels
    raise ValueError("manifest must be a dict with file_paths/labels")


def _sample_indices(
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


def _prepare_ref_mean(ref_mean_path: Path) -> np.ndarray:
    ref_mean = torch.load(ref_mean_path, map_location="cpu")
    if isinstance(ref_mean, dict):
        if "ref_mean" in ref_mean:
            ref_mean = ref_mean["ref_mean"]
        elif "mean" in ref_mean:
            ref_mean = ref_mean["mean"]
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    if ref_mean.shape != (5, 62, 62):
        raise ValueError(f"ref_mean shape must be (5,62,62), got {ref_mean.shape}")
    return ref_mean


def _build_A_nhalf(ref_mean: np.ndarray) -> List[np.ndarray]:
    A_nhalf = []
    for b in range(5):
        vals, vecs = np.linalg.eigh(ref_mean[b])
        vals = np.clip(vals, 1e-6, None)
        inv_sqrt = 1.0 / np.sqrt(vals)
        A_nhalf.append((vecs * inv_sqrt) @ vecs.T)
    return A_nhalf


def _log_map(cov: np.ndarray, A_nhalf: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    centered = A_nhalf @ cov @ A_nhalf
    vals, vecs = np.linalg.eigh(centered)
    vals = np.clip(vals, 1e-6, None)
    log_vals = np.log(vals)
    return (vecs * log_vals) @ vecs.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="logs/preprocess_effective_config.json",
        help="path to effective config json",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="logs/audit_tsm_preprocess_seed1.json",
        help="output json path for audit stats",
    )
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--max-windows-per-trial", type=int, default=None)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default=None,
        choices=["shuffle", "stride"],
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config_path)
    config = json.loads(config_path.read_text())
    manifest_path = Path(config["manifest_path"])
    ref_mean_path = Path(config["ref_mean_path"])

    sampling = dict(config.get("ref_mean_sampling") or {})
    max_trials = int(args.max_trials) if args.max_trials is not None else int(sampling.get("max_trials", 0))
    max_windows = (
        int(args.max_windows_per_trial)
        if args.max_windows_per_trial is not None
        else int(sampling.get("max_windows_per_trial", 0))
    )
    sample_mode = args.sample_mode or sampling.get("sample_mode", "shuffle")
    seed = int(args.seed) if args.seed is not None else int(sampling.get("seed", 0))

    file_paths, labels = _load_manifest(manifest_path)
    n_trials = len(file_paths)
    rng = np.random.default_rng(seed)
    trial_indices = _sample_indices(
        n_trials,
        rng=rng,
        max_trials=max_trials,
        sample_mode=sample_mode,
    )

    ref_mean = _prepare_ref_mean(ref_mean_path)
    A_nhalf = _build_A_nhalf(ref_mean)

    cov_eig_min = []
    cov_eig_max = []
    cov_cond = []
    cov_trace = []
    cov_diag_ratio = []
    cov_offdiag_mean = []

    tsm_fro = []
    tsm_diag_ratio = []
    tsm_min = []
    tsm_max = []
    tsm_has_nan = False

    window_counts = []
    trial_vectors = []
    spot_checks = []

    for idx in trial_indices:
        path = file_paths[int(idx)]
        arr = _normalize_trial_array(np.load(path))
        t_len = int(arr.shape[0])
        window_counts.append(t_len)

        if max_windows <= 0 or t_len <= max_windows:
            win_idx = np.arange(t_len)
        else:
            win_idx = rng.choice(t_len, size=int(max_windows), replace=False)
        win_idx = np.asarray(win_idx, dtype=int)

        mean_accum = np.zeros((5, 62, 62), dtype=np.float64)
        for w in win_idx:
            for b in range(5):
                cov = np.asarray(arr[w, b], dtype=np.float64)
                cov = 0.5 * (cov + cov.T)
                eigvals = np.linalg.eigvalsh(cov)
                eig_min = float(np.min(eigvals))
                eig_max = float(np.max(eigvals))
                cov_eig_min.append(eig_min)
                cov_eig_max.append(eig_max)
                cov_cond.append(eig_max / max(eig_min, 1e-12))
                cov_trace.append(float(np.trace(cov)))
                cov_diag_ratio.append(_diag_energy_ratio(cov))
                cov_offdiag_mean.append(_offdiag_mean_abs(cov))

                logm = _log_map(cov, A_nhalf[b])
                if not np.all(np.isfinite(logm)):
                    tsm_has_nan = True
                tsm_fro.append(float(np.linalg.norm(logm, ord="fro")))
                tsm_diag_ratio.append(_diag_energy_ratio(logm))
                tsm_min.append(float(np.min(logm)))
                tsm_max.append(float(np.max(logm)))
                mean_accum[b] += logm

        mean_log = mean_accum / max(1, len(win_idx))
        trial_vectors.append(mean_log.reshape(-1))

        if len(spot_checks) < 5:
            spot_checks.append(
                {
                    "file_path": path,
                    "label": int(labels[int(idx)]) if labels else None,
                    "T": t_len,
                    "shape": list(arr.shape),
                }
            )

    trial_vectors = np.asarray(trial_vectors, dtype=np.float64)
    norms = np.linalg.norm(trial_vectors, axis=1, keepdims=True)
    zero_norm = norms.squeeze() < 1e-8
    if np.any(zero_norm):
        norms[zero_norm] = 1.0
    trial_vectors = trial_vectors / norms
    cos_sim = trial_vectors @ trial_vectors.T
    tri = np.triu_indices(cos_sim.shape[0], k=1)
    cos_vals = cos_sim[tri] if cos_sim.size else np.asarray([])

    label_hist = {}
    for label in labels:
        label = int(label)
        label_hist[label] = label_hist.get(label, 0) + 1

    warnings = []
    if tsm_has_nan:
        warnings.append("TSM contains non-finite values")
    if np.any(zero_norm):
        warnings.append("Some trial mean vectors have near-zero norm")

    diag_ratio_median = float(np.median(cov_diag_ratio)) if cov_diag_ratio else None
    if diag_ratio_median is not None and diag_ratio_median > 0.98:
        warnings.append("cov_spd diag_energy_ratio median > 0.98 (possible over-diagonalized)")

    cos_median = float(np.median(cos_vals)) if cos_vals.size else None
    if cos_median is not None and cos_median > 0.95:
        warnings.append("cosine similarity median > 0.95 (possible trial homogenization)")

    audit = {
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "ref_mean_path": str(ref_mean_path),
        "sampling": {
            "seed": seed,
            "sample_mode": sample_mode,
            "max_trials": max_trials,
            "max_windows_per_trial": max_windows,
            "total_trials": n_trials,
            "sampled_trials": int(len(trial_indices)),
            "window_counts": {
                "min": int(np.min(window_counts)) if window_counts else 0,
                "mean": float(np.mean(window_counts)) if window_counts else 0.0,
                "max": int(np.max(window_counts)) if window_counts else 0,
            },
        },
        "cov_spd_stats": {
            "eig_min": _summary(cov_eig_min),
            "eig_max": _summary(cov_eig_max),
            "cond": _summary(cov_cond),
            "trace": _summary(cov_trace),
            "diag_energy_ratio": _summary(cov_diag_ratio),
            "offdiag_mean_abs": _summary(cov_offdiag_mean),
        },
        "tsm_stats": {
            "fro_norm": _summary(tsm_fro),
            "diag_energy_ratio": _summary(tsm_diag_ratio),
            "value_min": _summary(tsm_min),
            "value_max": _summary(tsm_max),
            "has_non_finite": tsm_has_nan,
        },
        "cosine_similarity": _summary(cos_vals),
        "label_hist": label_hist,
        "spot_checks": spot_checks,
        "warnings": warnings,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2))

    print("[audit] manifest_path:", manifest_path)
    print("[audit] ref_mean_path:", ref_mean_path)
    print("[audit] sampled_trials:", int(len(trial_indices)))
    print("[audit] cov_spd eig_min:", _summary(cov_eig_min))
    print("[audit] cov_spd diag_energy_ratio:", _summary(cov_diag_ratio))
    print("[audit] cov_spd offdiag_mean_abs:", _summary(cov_offdiag_mean))
    print("[audit] tsm fro_norm:", _summary(tsm_fro))
    print("[audit] tsm diag_energy_ratio:", _summary(tsm_diag_ratio))
    print("[audit] tsm value_min:", _summary(tsm_min))
    print("[audit] tsm value_max:", _summary(tsm_max))
    print("[audit] cosine_similarity:", _summary(cos_vals))
    if warnings:
        print("[audit] WARNING:", "; ".join(warnings))
    print("[audit] out_path:", out_path)


if __name__ == "__main__":
    main()
