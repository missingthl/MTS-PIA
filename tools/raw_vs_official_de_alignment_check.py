from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import mne

from datasets.seed_official_mat_dataset import _build_session_map
from manifold_raw.features import bandpass, parse_band_spec, window_slices


def _load_manifest(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "trials" in data:
        return data["trials"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported manifest format")


def _load_de_series(mat_path: str, trial_idx: int) -> np.ndarray:
    mat = scipy.io.loadmat(mat_path)
    key = f"de_LDS{trial_idx}"
    if key not in mat:
        raise KeyError(f"{key} not found in {mat_path}")
    arr = np.asarray(mat[key])
    if arr.ndim != 3:
        raise ValueError(f"Unexpected de_LDS shape {arr.shape} in {mat_path}")
    if arr.shape[0] != 62:
        # normalize to [62, T, 5]
        if 62 in arr.shape and 5 in arr.shape:
            ch_axis = arr.shape.index(62)
            band_axis = arr.shape.index(5)
            time_axis = [i for i in range(3) if i not in (ch_axis, band_axis)][0]
            arr = np.moveaxis(arr, [ch_axis, time_axis, band_axis], [0, 1, 2])
        else:
            raise ValueError(f"Cannot normalize de_LDS shape {arr.shape}")
    if arr.shape[2] != 5:
        raise ValueError(f"Unexpected band dimension in {mat_path}: {arr.shape}")
    return arr.mean(axis=0)  # [T, 5]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw vs official DE alignment check.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument(
        "--official-root",
        default="data/SEED/SEED_EEG/ExtractedFeatures_1s",
        help="ExtractedFeatures_1s root",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--trials-per-subject", type=int, default=2)
    parser.add_argument("--out-md", default="logs/manifests/raw_vs_de_alignment_check.md")
    parser.add_argument("--out-json", default="logs/manifests/raw_vs_de_alignment_check.json")
    args = parser.parse_args()

    trials = _load_manifest(Path(args.manifest_path))
    if not trials:
        raise ValueError("manifest is empty")

    session_map, _ = _build_session_map(args.official_root)

    rng = random.Random(args.seed)
    subjects = sorted({t["subject_id"] for t in trials})
    rng.shuffle(subjects)
    subjects = subjects[: args.subjects]

    bands = parse_band_spec("delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50")
    band_names = [b.name for b in bands]

    picked = []
    for subj in subjects:
        subj_trials = [t for t in trials if t["subject_id"] == subj]
        rng.shuffle(subj_trials)
        picked.extend(subj_trials[: args.trials_per_subject])

    results = []
    for row in picked:
        cnt_path = row["raw_file_path"]
        trial_idx = int(row["trial_idx"])
        subject = str(row["subject_id"])
        session = int(row["session_id"])
        start = int(row["start_sample"])
        end = int(row["end_sample"])
        if (subject, session) not in session_map:
            raise ValueError(f"Missing official mat for subject={subject} session={session}")
        mat_path = session_map[(subject, session)]

        raw = mne.io.read_raw_cnt(cnt_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data(start=start, stop=end)  # [C, N]
        n_samples = data.shape[1]
        windows = window_slices(n_samples, sfreq, win_sec=1.0, hop_sec=1.0)
        if not windows:
            raise ValueError("No windows produced for raw segment")
        raw_curve = np.zeros((len(windows), len(bands)), dtype=np.float32)

        for b_idx, band in enumerate(bands):
            band_data = bandpass(data, sfreq, band)
            for w_idx, (s, e) in enumerate(windows):
                seg = band_data[:, s:e]
                var = np.var(seg, axis=1)
                raw_curve[w_idx, b_idx] = float(np.log(np.mean(var) + 1e-8))

        de_curve = _load_de_series(mat_path, trial_idx)
        t_raw = raw_curve.shape[0]
        t_de = de_curve.shape[0]
        t_min = min(t_raw, t_de)
        raw_curve = raw_curve[:t_min]
        de_curve = de_curve[:t_min]

        band_corr = {}
        for b_idx, name in enumerate(band_names):
            band_corr[name] = _pearson(raw_curve[:, b_idx], de_curve[:, b_idx])
        mean_corr = float(np.mean(list(band_corr.values())))

        results.append(
            {
                "subject": subject,
                "session": session,
                "trial_idx": trial_idx,
                "cnt_path": cnt_path,
                "mat_path": mat_path,
                "T_raw": int(t_raw),
                "T_de": int(t_de),
                "T_used": int(t_min),
                "corr": band_corr,
                "mean_corr": mean_corr,
            }
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"samples": results}, indent=2))

    lines = []
    lines.append("# Raw vs Official DE alignment check")
    lines.append(f"sample_count={len(results)}")
    for row in results:
        lines.append(
            f"- subj={row['subject']} sess={row['session']} trial={row['trial_idx']} "
            f"T_raw={row['T_raw']} T_de={row['T_de']} mean_corr={row['mean_corr']:.3f} "
            f"corr={row['corr']}"
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))

    print(f"[align_check] samples={len(results)} out={out_md}", flush=True)


if __name__ == "__main__":
    main()
