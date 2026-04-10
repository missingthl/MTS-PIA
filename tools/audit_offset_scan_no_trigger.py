import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io

from datasets.seed_raw_cnt import build_eeg62_view, load_one_cnt
from datasets.seed_raw_trials import build_trial_index
from manifold_raw.features import bandpass, parse_band_spec, window_slices


BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
DEFAULT_BAND_RANGES = {"delta": (1.0, 4.0), "gamma": (31.0, 50.0)}


def _parse_cnt_subject_session(cnt_path: str) -> Tuple[int, int]:
    base = Path(cnt_path).stem
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid CNT filename: {cnt_path}")
    return int(parts[0]), int(parts[1])


def _resolve_mat_by_session(root: Path, subject: int, session: int) -> Path:
    subject_str = str(subject)
    files = sorted(
        p
        for p in root.iterdir()
        if p.suffix.lower() == ".mat"
        and p.name.startswith(subject_str + "_")
        and p.name.lower() != "label.mat"
    )
    if not files:
        raise FileNotFoundError(f"No .mat files found for subject {subject} in {root}")
    by_date = []
    for p in files:
        parts = p.stem.split("_")
        if len(parts) != 2:
            continue
        try:
            by_date.append((int(parts[1]), p.name))
        except ValueError:
            continue
    if not by_date:
        raise FileNotFoundError(f"No date-coded .mat files for subject {subject} in {root}")
    by_date.sort(key=lambda x: x[0])
    if session < 1 or session > len(by_date):
        raise ValueError(f"Session {session} out of range for subject {subject}")
    return root / by_date[session - 1][1]


def _normalize_trial_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array; got {arr.shape}")
    shape = list(arr.shape)
    if 62 not in shape or 5 not in shape:
        raise ValueError(f"Unexpected trial shape {shape}; missing C=62 or B=5")
    ch_axis = shape.index(62)
    band_axis = shape.index(5)
    time_axis = [i for i in range(3) if i not in (ch_axis, band_axis)]
    if len(time_axis) != 1:
        raise ValueError(f"Cannot infer time axis from shape {shape}")
    return np.moveaxis(arr, [ch_axis, time_axis[0], band_axis], [0, 1, 2])


def _band_index(name: str) -> int:
    return BAND_ORDER.index(name)


def _check_finite(arr: np.ndarray, tag: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values detected in {tag}")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> Tuple[float, int]:
    best_r = float("-inf")
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            aa = a[lag:]
            bb = b[:-lag]
        elif lag < 0:
            aa = a[:lag]
            bb = b[-lag:]
        else:
            aa = a
            bb = b
        n = min(aa.size, bb.size)
        if n == 0:
            continue
        r = _pearson(aa[:n], bb[:n])
        if np.isnan(r):
            continue
        if r > best_r:
            best_r = r
            best_lag = lag
    return best_r, best_lag


def main() -> None:
    parser = argparse.ArgumentParser(description="Offset scan without triggers.")
    parser.add_argument("--cnt", required=True, help="path to CNT file")
    parser.add_argument("--official-root", required=True, help="ExtractedFeatures_1s or _4s")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument("--bands", type=str, default="delta,gamma")
    parser.add_argument("--offset-min", type=float, default=-10.0)
    parser.add_argument("--offset-max", type=float, default=10.0)
    parser.add_argument("--offset-step", type=float, default=0.5)
    parser.add_argument("--lag-max", type=int, default=3)
    parser.add_argument("--eps-var", type=float, default=1e-12)
    parser.add_argument(
        "--out-json",
        default="logs/audit_offset_scan_trial1.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--out-md",
        default="logs/audit_offset_scan_report.md",
        help="output MD path",
    )
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    payload: Dict[str, object] = {}

    raw = load_one_cnt(args.cnt, preload=False)
    sfreq = float(raw.info["sfreq"])
    subject, session = _parse_cnt_subject_session(args.cnt)

    trial_input = int(args.trial)
    trial_zero = trial_input - 1
    trial_list = build_trial_index(
        args.cnt,
        "data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt",
        "data/SEED/SEED_EEG/SEED_stimulation.xlsx",
        time_unit="samples@1000",
    )
    if trial_zero < 0 or trial_zero >= len(trial_list):
        raise ValueError(f"trial out of range: {trial_input}")
    t_meta = trial_list[trial_zero]
    start_sec_base = float(t_meta.t_start_s)

    root_path = Path(args.official_root)
    mat_path = _resolve_mat_by_session(root_path, subject, session)
    mat = scipy.io.loadmat(mat_path)
    key_name = f"de_LDS{trial_input}"
    if key_name not in mat:
        raise KeyError(f"{key_name} not found in {mat_path}")
    arr = _normalize_trial_array(mat[key_name])
    T_off = int(arr.shape[1])

    band_names = [b.strip().lower() for b in args.bands.split(",") if b.strip()]
    for name in band_names:
        if name not in DEFAULT_BAND_RANGES:
            raise ValueError(f"Unsupported band: {name}")

    official_curves: Dict[str, np.ndarray] = {}
    for name in band_names:
        idx = _band_index(name)
        curve = arr[:, :, idx].mean(axis=0).astype(np.float64)
        _check_finite(curve, f"official_curve_{name}")
        official_curves[name] = curve

    duration_sec = float(T_off * args.hop_sec)
    offsets = np.arange(args.offset_min, args.offset_max + 1e-9, args.offset_step)
    raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")

    results: Dict[str, List[dict]] = {name: [] for name in band_names}
    skip_count = 0

    for offset_sec in offsets:
        start_sec = start_sec_base + float(offset_sec)
        end_sec = start_sec + duration_sec
        start_idx = int(round(start_sec * sfreq))
        end_idx = int(round(end_sec * sfreq))
        if start_idx < 0 or end_idx > raw62.n_times or end_idx <= start_idx:
            skip_count += 1
            continue
        seg = raw62.get_data(start=start_idx, stop=end_idx).astype(np.float32)

        slices = window_slices(seg.shape[1], sfreq, args.window_sec, args.hop_sec)
        if not slices:
            skip_count += 1
            continue

        for name in band_names:
            lo, hi = DEFAULT_BAND_RANGES[name]
            band = parse_band_spec(f"{name}:{lo}-{hi}")[0]
            band_data = bandpass(seg, sfreq, band)
            vals = []
            for s, e in slices:
                win = band_data[:, s:e]
                var = np.var(win, axis=1)
                logvar = np.log(var + float(args.eps_var))
                vals.append(float(np.mean(logvar)))
            curve = np.asarray(vals, dtype=np.float64)
            _check_finite(curve, f"raw_curve_{name}")
            r0 = _pearson(official_curves[name], curve)
            best_r, best_lag = _best_lag_corr(
                official_curves[name], curve, int(args.lag_max)
            )
            results[name].append(
                {
                    "offset_sec": float(offset_sec),
                    "r0": float(r0),
                    "best_r": float(best_r),
                    "best_lag": int(best_lag),
                }
            )

    summary: Dict[str, dict] = {}
    for name in band_names:
        ranked = sorted(results[name], key=lambda x: x["best_r"], reverse=True)
        top5 = [(r["offset_sec"], r["best_r"]) for r in ranked[:5]]
        best = ranked[0] if ranked else {}
        summary[name] = {
            "best_offset_sec": best.get("offset_sec"),
            "best_r": best.get("best_r"),
            "best_lag": best.get("best_lag"),
            "top5": top5,
        }

    payload.update(
        {
            "no_events": True,
            "sfreq": sfreq,
            "trial_input": trial_input,
            "trial_zero": trial_zero,
            "subject": subject,
            "session": session,
            "official_root": str(root_path),
            "official_mat": str(mat_path),
            "official_key": key_name,
            "T_off": T_off,
            "window_sec": float(args.window_sec),
            "hop_sec": float(args.hop_sec),
            "duration_sec": duration_sec,
            "bands": band_names,
            "scan_range": {
                "offset_min": float(args.offset_min),
                "offset_max": float(args.offset_max),
                "offset_step": float(args.offset_step),
                "lag_max": int(args.lag_max),
            },
            "skip_count": int(skip_count),
            "summary": summary,
            "results": results,
        }
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Offset Scan (NO_TRIGGER)",
        "",
        f"- no_events: true",
        f"- sfreq: {sfreq:.2f}",
        f"- trial: {trial_input}",
        f"- window_sec: {args.window_sec}",
        f"- hop_sec: {args.hop_sec}",
        f"- duration_sec: {duration_sec:.2f}",
        f"- bands: {band_names}",
        f"- scan_range: [{args.offset_min}, {args.offset_max}] step {args.offset_step}",
        f"- skip_count: {skip_count}",
        "",
        "## Best results",
    ]
    for name in band_names:
        md_lines.append(
            f"- {name}: best_offset_sec={summary[name]['best_offset_sec']} "
            f"best_r={summary[name]['best_r']} best_lag={summary[name]['best_lag']}"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")

    print(
        f"[offset_scan] no_events=true sfreq={sfreq:.2f} trial={trial_input} "
        f"window_sec={args.window_sec}",
        flush=True,
    )
    for name in band_names:
        info = summary[name]
        print(
            f"[offset_scan][{name}] best_offset_sec={info['best_offset_sec']} "
            f"best_r={info['best_r']:.4f} best_lag={info['best_lag']}",
            flush=True,
        )
    if "gamma" in summary:
        print(
            f"[offset_scan][gamma] top5={summary['gamma']['top5']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
