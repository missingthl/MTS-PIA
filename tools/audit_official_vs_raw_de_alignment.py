import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io

from datasets.seed_raw_cnt import build_eeg62_view, load_one_cnt
from datasets.seed_raw_trials import build_trial_index
from manifold_raw.features import bandpass, parse_band_spec, window_slices


BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def _resolve_official_root(root: str) -> Path:
    p = Path(root)
    if p.is_dir():
        return p
    candidate = Path("data/SEED/SEED_EEG") / root
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"official_root not found: {root}")


def _resolve_cnt_path(subject: int, session: int) -> Path:
    path = Path("data/SEED/SEED_EEG/SEED_RAW_EEG") / f"{subject}_{session}.cnt"
    if not path.is_file():
        raise FileNotFoundError(f"CNT file not found: {path}")
    return path


def _resolve_time_txt() -> Path:
    path = Path("data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt")
    if not path.is_file():
        raise FileNotFoundError(f"time.txt not found: {path}")
    return path


def _resolve_stim_xlsx() -> Path:
    path = Path("data/SEED/SEED_EEG/SEED_stimulation.xlsx")
    if not path.is_file():
        raise FileNotFoundError(f"SEED_stimulation.xlsx not found: {path}")
    return path


def _parse_band_list(bands: str) -> List[str]:
    names = [b.strip().lower() for b in bands.split(",") if b.strip()]
    for name in names:
        if name not in BAND_ORDER:
            raise ValueError(f"Unknown band name: {name}")
    return names


def _band_index(name: str) -> int:
    return BAND_ORDER.index(name)


def _normalize_trial_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array; got {arr.shape}")
    shape = arr.shape
    if 62 not in shape or 5 not in shape:
        raise ValueError(f"Unexpected trial shape {shape}; missing C=62 or B=5")
    ch_axis = shape.index(62)
    band_axis = shape.index(5)
    time_axis = [i for i in range(3) if i not in (ch_axis, band_axis)]
    if len(time_axis) != 1:
        raise ValueError(f"Cannot infer time axis from shape {shape}")
    return np.moveaxis(arr, [ch_axis, time_axis[0], band_axis], [0, 1, 2])


def _stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _detrend_curve(x: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return x.astype(np.float64)
    t = np.arange(x.size, dtype=np.float64)
    coef = np.polyfit(t, x.astype(np.float64), deg=1)
    trend = coef[0] * t + coef[1]
    return x.astype(np.float64) - trend


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
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
        if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
            continue
        r = _pearson(aa, bb)
        if np.isnan(r):
            continue
        if r > best_r:
            best_r = r
            best_lag = lag
    return best_r, best_lag


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official de_LDS vs raw DE proxy alignment.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--official-root", type=str, default="ExtractedFeatures_4s")
    parser.add_argument("--feature-base", type=str, default="de_LDS")
    parser.add_argument("--bands", type=str, default="delta,gamma")
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--detrend", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--time-unit",
        type=str,
        default="",
        help="time unit for time.txt (samples@1000, samples@200, seconds)",
    )
    parser.add_argument("--out", type=str, default="logs/audit_official_vs_raw_de.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    payload: Dict[str, object] = {}
    try:
        official_root = _resolve_official_root(args.official_root)
        cnt_path = _resolve_cnt_path(args.subject, args.session)
        time_txt = _resolve_time_txt()
        stim_xlsx = _resolve_stim_xlsx()

        trial_input = int(args.trial)
        if trial_input < 1 or trial_input > 15:
            raise ValueError("trial must be in [1, 15] for de_LDS keys")
        trial_zero = trial_input - 1

        payload.update(
            {
                "official_root": str(official_root),
                "cnt_path": str(cnt_path),
                "time_txt": str(time_txt),
                "stim_xlsx": str(stim_xlsx),
                "subject": int(args.subject),
                "session": int(args.session),
                "trial_input": trial_input,
                "trial_zero": trial_zero,
                "window_sec": float(args.window_sec),
                "hop_sec": float(args.hop_sec),
                "max_lag": int(args.max_lag),
                "detrend": int(args.detrend),
                "eps": float(args.eps),
                "time_unit": args.time_unit or None,
            }
        )
        print(
            "[audit] "
            f"trial_input={trial_input} trial_zero={trial_zero} "
            f"key={args.feature_base}{trial_input} time_unit={args.time_unit or 'default'}",
            flush=True,
        )

        band_names = _parse_band_list(args.bands)
        payload["bands"] = band_names

        # Official .mat (subject/session -> date-sorted filename)
        mat_path = _resolve_mat_by_session(official_root, args.subject, args.session)
        mat = scipy.io.loadmat(mat_path)
        key_name = f"{args.feature_base}{trial_input}"
        if key_name not in mat:
            raise KeyError(f"Key {key_name} not found in {mat_path}")
        arr = _normalize_trial_array(mat[key_name])
        T_off = int(arr.shape[1])

        official_curves: Dict[str, List[float]] = {}
        official_stats: Dict[str, Dict[str, float]] = {}
        for name in band_names:
            b_idx = _band_index(name)
            curve = arr[:, :, b_idx].mean(axis=0)
            _check_finite(curve, f"official_curve_{name}")
            official_curves[name] = curve.astype(np.float64).tolist()
            official_stats[name] = _stats(curve)

        # Raw CNT
        raw = load_one_cnt(str(cnt_path), preload=False)
        sfreq = float(raw.info["sfreq"])
        raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
        trial_index = build_trial_index(
            str(cnt_path),
            str(time_txt),
            str(stim_xlsx),
            time_unit=args.time_unit or None,
        )
        if trial_zero >= len(trial_index):
            raise ValueError(f"trial index out of range: {trial_zero}")
        t_meta = trial_index[trial_zero]
        start_idx = int(round(t_meta.t_start_s * sfreq))
        end_idx = int(round(t_meta.t_end_s * sfreq))
        seg = raw62.get_data(start=start_idx, stop=end_idx)

        payload.update(
            {
                "sfreq": sfreq,
                "trial_start_end": {
                    "t_start_s": float(t_meta.t_start_s),
                    "t_end_s": float(t_meta.t_end_s),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                },
                "raw_trial_seconds": float((end_idx - start_idx) / sfreq),
                "raw_trial_samples": int(seg.shape[1]),
                "official_mat_path": str(mat_path),
                "official_key": key_name,
                "T_off": T_off,
            }
        )
        print(f"[raw] sfreq={sfreq:.2f}", flush=True)
        print(
            "[raw] trial_seconds="
            f"{payload['raw_trial_seconds']:.3f} "
            f"samples={payload['raw_trial_samples']}",
            flush=True,
        )

        band_spec = parse_band_spec("delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50")
        band_map = {b.name: b for b in band_spec}

        win_slices = window_slices(seg.shape[1], sfreq, args.window_sec, args.hop_sec)
        if not win_slices:
            raise ValueError("window_slices returned empty list")
        payload["N_win"] = len(win_slices)
        print(f"[raw] window_sec={args.window_sec} hop_sec={args.hop_sec} N_win={len(win_slices)}", flush=True)

        raw_curves: Dict[str, List[float]] = {}
        raw_stats: Dict[str, Dict[str, float]] = {}

        for name in band_names:
            if name not in band_map:
                raise ValueError(f"Band not in spec: {name}")
            b_data = bandpass(seg, sfreq, band_map[name])
            vals = []
            for s, e in win_slices:
                win = b_data[:, s:e]
                var = np.var(win, axis=1)
                logvar = np.log(var + args.eps)
                vals.append(float(np.mean(logvar)))
            curve = np.asarray(vals, dtype=np.float64)
            _check_finite(curve, f"raw_curve_{name}")
            raw_curves[name] = curve.tolist()
            raw_stats[name] = _stats(curve)

        payload["official_stats"] = official_stats
        payload["raw_stats"] = raw_stats

        corr: Dict[str, Dict[str, float]] = {}
        magnitude: Dict[str, Dict[str, float]] = {}
        curve_head10: Dict[str, Dict[str, List[float]]] = {}
        for name in band_names:
            off_curve = np.asarray(official_curves[name], dtype=np.float64)
            raw_curve = np.asarray(raw_curves[name], dtype=np.float64)
            if args.detrend:
                off_corr = _detrend_curve(off_curve)
                raw_corr = _detrend_curve(raw_curve)
            else:
                off_corr = off_curve
                raw_corr = raw_curve
            r0 = _pearson(off_corr, raw_corr)
            best_r, best_lag = _best_lag_corr(off_corr, raw_corr, args.max_lag)
            corr[name] = {"r0": float(r0), "best_r": float(best_r), "best_lag": int(best_lag)}
            mean_diff_log = float(np.mean(raw_curve) - np.mean(off_curve))
            ratio_linear = float(math.exp(mean_diff_log))
            magnitude[name] = {
                "mean_diff_log": mean_diff_log,
                "ratio_linear": ratio_linear,
            }
            curve_head10[name] = {
                "official": off_curve[:10].tolist(),
                "raw": raw_curve[:10].tolist(),
            }

        payload["corr"] = corr
        payload["magnitude"] = magnitude
        payload["curve_head10"] = curve_head10

        _write_json(out_path, payload)

        for name in band_names:
            o = official_stats[name]
            r = raw_stats[name]
            c = corr[name]
            m = magnitude[name]
            print(
                f"[{name}] official: T={T_off} mean={o['mean']:.6e} std={o['std']:.6e}",
                flush=True,
            )
            print(
                f"[{name}] raw:      T={len(raw_curves[name])} "
                f"mean={r['mean']:.6e} std={r['std']:.6e}",
                flush=True,
            )
            print(
                f"[{name}] corr: r0={c['r0']:.4f} best_r={c['best_r']:.4f} "
                f"best_lag={c['best_lag']}",
                flush=True,
            )
            print(
                f"[{name}] magnitude: mean_diff_log={m['mean_diff_log']:.4f} "
                f"ratio_linear={m['ratio_linear']:.4f}",
                flush=True,
            )
    except Exception as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(out_path, payload)
        print(f"[audit] error: {type(exc).__name__}: {exc}", flush=True)
        raise


def _resolve_subject_date(subject: int, session: int, root: Path) -> str:
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
    return by_date[session - 1][1]


def _resolve_mat_by_session(root: Path, subject: int, session: int) -> Path:
    fname = _resolve_subject_date(subject, session, root)
    return root / fname


def _check_finite(arr: np.ndarray, tag: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values detected in {tag}")


if __name__ == "__main__":
    main()
