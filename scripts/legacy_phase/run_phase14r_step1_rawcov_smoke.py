import os
import sys
import json
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from datasets.seed_raw_trials import build_trial_index, slice_raw_trials
from manifold_raw.features import BandSpec, bandpass, parse_band_spec, window_slices, logmap_spd, vec_utri


CONFIG = {
    "seed": 0,
    "dataset": "seed1",
    "seed_raw_root": "data/SEED/SEED_EEG/SEED_RAW_EEG",
    "raw_backend": "cnt",
    "time_unit": None,
    "trial_offset_sec": 0.0,
    "window_sec": 4.0,
    "hop_sec": 4.0,
    "bands": "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    "norm_mode": "per_band_global_z",
    "matrix_mode": "cov",
    "spd_eps": 1e-3,
    "logmap_eps": 1e-6,
    "epochs": 10,
    "batch_size": 8,
    "split_mode": "trial_80_20",
    "out_root": "promoted_results/phase14r/step1/seed1/seed0/raw_cov_tsm_manifold_only",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _parse_cnt_name(cnt_path: str) -> Tuple[int, int, str]:
    base = os.path.splitext(os.path.basename(cnt_path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid CNT filename: {cnt_path}")
    subject_str = parts[0]
    session = int(parts[1])
    try:
        subject_int = int(subject_str)
    except ValueError:
        subject_int = -1
    return subject_int, session, subject_str


def _sorted_cnt_files(raw_root: str, ext: str) -> List[str]:
    paths = [str(p) for p in Path(raw_root).iterdir() if p.suffix.lower() == ext]
    return sorted(paths, key=lambda p: _parse_cnt_name(p))


def _trial_id(t) -> str:
    return f"{t.subject}_s{t.session}_t{t.trial}"


def _write_error(out_dir: str, msg: str) -> None:
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "ERROR.md"), "w") as f:
        f.write(msg)


def _deterministic_split(trials: List[dict], seed: int) -> Tuple[List[dict], List[dict]]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(trials))
    n_train = int(0.8 * len(trials))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    train = [trials[i] for i in train_idx]
    test = [trials[i] for i in test_idx]
    return train, test


def _audit_split(train_ids: List[str], test_ids: List[str]) -> dict:
    train_set = set(train_ids)
    test_set = set(test_ids)
    intersection = sorted(list(train_set.intersection(test_set)))
    violations = []
    if intersection:
        violations.extend(intersection[:5])
    return {
        "n_train_trials": len(train_ids),
        "n_test_trials": len(test_ids),
        "intersection_size": len(intersection),
        "violations": violations,
    }


def _band_norm_global(x: np.ndarray) -> np.ndarray:
    mean = float(x.mean())
    std = float(x.std()) + 1e-6
    return (x - mean) / std


def _cov_from_cat(x_cat: np.ndarray, eps: float) -> np.ndarray:
    x_c = x_cat - x_cat.mean(axis=1, keepdims=True)
    denom = max(1, x_c.shape[1] - 1)
    cov = (x_c @ x_c.T) / float(denom)
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
    return cov


def _spectrum_stats_sample(cov_list: List[np.ndarray], eps: float) -> dict:
    if not cov_list:
        return {}
    eig_mins = []
    conds = []
    eff_ranks = []
    eps_dom = []
    low_eigs = []
    for cov in cov_list:
        vals = np.linalg.eigvalsh(cov)
        vals = np.maximum(vals, 1e-12)
        conds.append(float(vals.max() / vals.min()))
        p = vals / vals.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        eff_ranks.append(float(np.exp(entropy)))
        eps_norm = eps * np.sqrt(cov.shape[0])
        c_norm = float(np.linalg.norm(cov))
        eps_dom.append(float(eps_norm / (c_norm + 1e-9)))
        low_eigs.append(float(np.mean(vals <= 10 * eps) * cov.shape[0]))
        eig_mins.append(float(np.quantile(vals, 0.05)))
    return {
        "pre_eps_min_eig_p05": float(np.mean(eig_mins)),
        "cond_p95": float(np.quantile(conds, 0.95)),
        "eff_rank": float(np.mean(eff_ranks)),
        "eps_dominance": float(np.mean(eps_dom)),
        "eigs_le_10eps_count": float(np.mean(low_eigs)),
    }


def main() -> None:
    cfg = CONFIG
    out_dir = cfg["out_root"]
    ensure_dir(out_dir)

    ext = ".fif" if cfg["raw_backend"] == "fif" else ".cnt"
    try:
        cnt_files = _sorted_cnt_files(cfg["seed_raw_root"], ext)
        if not cnt_files:
            raise FileNotFoundError(f"No {ext} files under {cfg['seed_raw_root']}")
        time_txt = os.path.join(cfg["seed_raw_root"], "time.txt")
        stim_xlsx = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
        if not os.path.isfile(time_txt):
            time_txt = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
        if not os.path.isfile(time_txt):
            raise FileNotFoundError(f"time.txt not found near raw root or default: {cfg['seed_raw_root']}")

        # Build trial list across CNT files
        trials_all = []
        for cnt_path in cnt_files:
            trials = build_trial_index(cnt_path, time_txt, stim_xlsx, time_unit=cfg["time_unit"])
            for t in trials:
                trials_all.append(
                    {
                        "trial_id": _trial_id(t),
                        "label": int(t.label),
                        "trial_obj": t,
                        "cnt_path": cnt_path,
                    }
                )
    except Exception as e:
        _write_error(out_dir, f"ERROR: {e}")
        raise

    # Deterministic order by trial_id
    trials_all = sorted(trials_all, key=lambda r: r["trial_id"])
    trial_ids = [r["trial_id"] for r in trials_all]
    if len(trial_ids) != len(set(trial_ids)):
        _write_error(out_dir, "ERROR: duplicate trial_id_str detected")
        raise RuntimeError("duplicate trial_id_str detected")

    # Split
    train_trials, test_trials = _deterministic_split(trials_all, cfg["seed"])
    train_ids = [r["trial_id"] for r in train_trials]
    test_ids = [r["trial_id"] for r in test_trials]
    audit = _audit_split(train_ids, test_ids)
    if audit["intersection_size"] != 0:
        _write_error(out_dir, f"ERROR: train/test overlap detected: {audit['violations']}")
        raise RuntimeError("train/test overlap detected")

    write_json(os.path.join(out_dir, "alignment_audit.json"), audit)

    bands = parse_band_spec(cfg["bands"])
    bands_order = [b.name for b in bands]

    # Cache raw files
    raw_cache = {}

    def load_raw62(cnt_path: str):
        if cnt_path in raw_cache:
            return raw_cache[cnt_path]
        raw = load_one_raw(cnt_path, backend=cfg["raw_backend"], preload=False)
        raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
        raw_cache[cnt_path] = raw62
        return raw62

    # Helper to process trials into window features
    def process_trials(trial_rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[float]]:
        X_list = []
        y_list = []
        trial_id_list = []
        window_idx_list = []
        cov_samples = []

        # Group by CNT to reuse raw loading
        by_cnt: Dict[str, List[dict]] = {}
        for row in trial_rows:
            by_cnt.setdefault(row["cnt_path"], []).append(row)

        for cnt_path, rows in sorted(by_cnt.items()):
            raw62 = load_raw62(cnt_path)
            trials = [r["trial_obj"] for r in rows]
            segments = slice_raw_trials(raw62, trials, trial_offset_sec=cfg["trial_offset_sec"])
            seg_map = {f"{m['subject']}_s{m['session']}_t{m['trial']}": seg for seg, m in segments}

            for row in rows:
                tid = row["trial_id"]
                if tid not in seg_map:
                    continue
                seg = seg_map[tid]  # (C, T)
                fs = float(raw62.info["sfreq"])
                windows = window_slices(seg.shape[1], fs, cfg["window_sec"], cfg["hop_sec"])
                # Band-pass full trial
                band_full = {b.name: bandpass(seg, fs, b) for b in bands}

                for w_idx, (s, e) in enumerate(windows):
                    band_windows = []
                    for b in bands:
                        bw = band_full[b.name][:, s:e]
                        bw = _band_norm_global(bw)
                        band_windows.append(bw)
                    x_cat = np.concatenate(band_windows, axis=1)  # (62, T*5)
                    cov = _cov_from_cat(x_cat, cfg["spd_eps"])
                    if len(cov_samples) < 256:
                        cov_samples.append(cov)
                    logm = logmap_spd(cov, cfg["logmap_eps"])
                    feat = vec_utri(logm).astype(np.float32)
                    X_list.append(feat)
                    y_list.append(int(row["label"]))
                    trial_id_list.append(tid)
                    window_idx_list.append(w_idx)
        return (
            np.vstack(X_list).astype(np.float32),
            np.asarray(y_list, dtype=np.int64),
            trial_id_list,
            window_idx_list,
            cov_samples,
        )

    # Build train/test features
    X_train, y_train, train_tids, train_widx, cov_samples = process_trials(train_trials)
    X_test, y_test, test_tids, test_widx, cov_samples_test = process_trials(test_trials)
    cov_samples.extend(cov_samples_test)

    # Diagnostics
    spectrum_stats = _spectrum_stats_sample(cov_samples[:256], cfg["spd_eps"])
    diagnostics = {
        "fs": float(raw_cache[next(iter(raw_cache))].info["sfreq"]) if raw_cache else None,
        "window_sec": cfg["window_sec"],
        "hop_sec": cfg["hop_sec"],
        "bands_mode": "all5_timecat_raw",
        "norm_mode": cfg["norm_mode"],
        "eps": cfg["spd_eps"],
        "shrinkage_method": "eps_only",
        "spectrum_stats_sample": spectrum_stats,
    }
    write_json(os.path.join(out_dir, "report_diagnostics.json"), diagnostics)

    # Train shallow classifier on TSM features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=200,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=cfg["seed"],
    )
    clf.fit(X_train_std, y_train)
    proba = clf.predict_proba(X_test_std)
    y_pred = np.argmax(proba, axis=1)

    win_acc = float(np.mean(y_pred == y_test)) if len(y_test) > 0 else 0.0

    # Window-level CSV
    win_rows = []
    for tid, widx, y, p in zip(test_tids, test_widx, y_test, proba):
        win_rows.append(
            {
                "trial_id_str": tid,
                "window_idx": int(widx),
                "y_true": int(y),
                "y_pred": int(np.argmax(p)),
                "p0": float(p[0]),
                "p1": float(p[1]),
                "p2": float(p[2]),
            }
        )
    import pandas as pd

    pd.DataFrame(win_rows).to_csv(os.path.join(out_dir, "manifold_window_pred.csv"), index=False)

    # Trial-level aggregation
    trial_map: Dict[str, dict] = {}
    for tid, y, p in zip(test_tids, y_test, proba):
        entry = trial_map.setdefault(tid, {"y": int(y), "p_sum": np.zeros(3), "n": 0})
        entry["p_sum"] += p
        entry["n"] += 1

    trial_rows = []
    for tid, entry in trial_map.items():
        p_mean = entry["p_sum"] / max(1, entry["n"])
        y_pred_t = int(np.argmax(p_mean))
        trial_rows.append(
            {
                "trial_id_str": tid,
                "y_true": int(entry["y"]),
                "y_pred": y_pred_t,
                "p0": float(p_mean[0]),
                "p1": float(p_mean[1]),
                "p2": float(p_mean[2]),
                "n_windows": int(entry["n"]),
            }
        )
    pd.DataFrame(trial_rows).to_csv(os.path.join(out_dir, "manifold_trial_pred.csv"), index=False)

    trial_acc = float(
        np.mean([r["y_pred"] == r["y_true"] for r in trial_rows]) if trial_rows else 0.0
    )

    metrics = {
        "trial_acc": trial_acc,
        "win_acc": win_acc,
        "n_train_trials": len(train_ids),
        "n_test_trials": len(test_ids),
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(X_test)),
        "split_mode": cfg["split_mode"],
    }
    write_json(os.path.join(out_dir, "report_metrics.json"), metrics)

    # SINGLE_RUN_REPORT.md
    report_lines = []
    report_lines.append("# Phase 14R Step 1: Raw EEG → Cov Pooling → Shallow TSM (Manifold-Only) Smoke")
    report_lines.append("")
    report_lines.append("## Config")
    report_lines.append(f"- seed: {cfg['seed']}")
    report_lines.append(f"- dataset: {cfg['dataset']}")
    report_lines.append(f"- raw_root: {cfg['seed_raw_root']}")
    report_lines.append(f"- raw_backend: {cfg['raw_backend']}")
    report_lines.append(f"- window_sec: {cfg['window_sec']}")
    report_lines.append(f"- hop_sec: {cfg['hop_sec']}")
    report_lines.append(f"- bands: {bands_order}")
    report_lines.append(f"- norm_mode: {cfg['norm_mode']}")
    report_lines.append(f"- matrix_mode: {cfg['matrix_mode']}")
    report_lines.append(f"- eps: {cfg['spd_eps']}")
    report_lines.append("")
    report_lines.append("## Metrics")
    report_lines.append(f"- win_acc: {win_acc:.4f}")
    report_lines.append(f"- trial_acc: {trial_acc:.4f}")
    report_lines.append(f"- n_train_trials: {metrics['n_train_trials']}")
    report_lines.append(f"- n_test_trials: {metrics['n_test_trials']}")
    report_lines.append(f"- n_train_windows: {metrics['n_train_windows']}")
    report_lines.append(f"- n_test_windows: {metrics['n_test_windows']}")
    report_lines.append("")
    report_lines.append("## Diagnostics")
    report_lines.append(f"- cond_p95: {spectrum_stats.get('cond_p95', 'N/A')}")
    report_lines.append(f"- eff_rank: {spectrum_stats.get('eff_rank', 'N/A')}")
    report_lines.append(f"- eps_dominance: {spectrum_stats.get('eps_dominance', 'N/A')}")
    report_lines.append(f"- eigs_le_10eps_count: {spectrum_stats.get('eigs_le_10eps_count', 'N/A')}")
    report_lines.append("")
    report_lines.append("## Artifact Index")
    report_lines.append(f"- manifold_window_pred.csv")
    report_lines.append(f"- manifold_trial_pred.csv")
    report_lines.append(f"- report_metrics.json")
    report_lines.append(f"- report_diagnostics.json")
    report_lines.append(f"- alignment_audit.json")
    report_lines.append("")
    report_lines.append("## Reproduction")
    report_lines.append("/home/thl/miniconda3/envs/pia/bin/python scripts/run_phase14r_step1_rawcov_smoke.py")

    with open(os.path.join(out_dir, "SINGLE_RUN_REPORT.md"), "w") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
