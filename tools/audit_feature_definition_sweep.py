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


def _resample_if_needed(data: np.ndarray, fs: float, resample_to: int | None) -> Tuple[np.ndarray, float]:
    if not resample_to or resample_to <= 0 or abs(float(resample_to) - fs) < 1e-6:
        return data, fs
    import mne

    data64 = np.asarray(data, dtype=np.float64)
    resampled = mne.filter.resample(
        data64,
        up=float(resample_to),
        down=float(fs),
        axis=1,
        npad="auto",
        verbose="ERROR",
    )
    return resampled.astype(np.float32, copy=False), float(resample_to)


def _apply_reref(seg: np.ndarray) -> np.ndarray:
    return seg - seg.mean(axis=0, keepdims=True)


def _compute_curve(
    band_data: np.ndarray,
    slices: List[Tuple[int, int]],
    feature: str,
    eps_var: float,
) -> np.ndarray:
    vals = []
    for s, e in slices:
        win = band_data[:, s:e]
        var = np.var(win, axis=1)
        if feature == "logvar":
            feat = np.log(var + eps_var)
        elif feature == "de":
            feat = 0.5 * np.log(2 * math.pi * math.e * (var + eps_var))
        else:
            raise ValueError(f"Unknown feature: {feature}")
        vals.append(float(np.mean(feat)))
    curve = np.asarray(vals, dtype=np.float64)
    _check_finite(curve, f"raw_curve_{feature}")
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature definition sweep for official de_LDS alignment.")
    parser.add_argument("--cnt", required=True, help="path to CNT file")
    parser.add_argument("--official-root", required=True, help="ExtractedFeatures_1s or _4s")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument("--offset-min", type=float, default=-3.0)
    parser.add_argument("--offset-max", type=float, default=3.0)
    parser.add_argument("--offset-step", type=float, default=0.5)
    parser.add_argument("--lag-max", type=int, default=3)
    parser.add_argument("--eps-var", type=float, default=1e-12)
    parser.add_argument(
        "--out-json",
        default="logs/audit_feature_def_sweep_trial1.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--out-md",
        default="logs/audit_feature_def_sweep_report.md",
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

    official_curves = {}
    for name in ("delta", "gamma"):
        idx = _band_index(name)
        curve = arr[:, :, idx].mean(axis=0).astype(np.float64)
        _check_finite(curve, f"official_curve_{name}")
        official_curves[name] = curve

    duration_sec = float(T_off * args.hop_sec)
    offsets = np.arange(args.offset_min, args.offset_max + 1e-9, args.offset_step)

    configs = []
    for resample in [None, 200]:
        for unit in ["V", "uV"]:
            for reref in ["none", "avg"]:
                for bandset in ["A", "B"]:
                    for feature in ["logvar", "de"]:
                        configs.append(
                            {
                                "resample": resample,
                                "unit": unit,
                                "reref": reref,
                                "bandset": bandset,
                                "feature": feature,
                            }
                        )

    bandsets = {
        "A": {"delta": (1.0, 4.0), "gamma": (31.0, 50.0)},
        "B": {"delta": (1.0, 3.0), "gamma": (30.0, 50.0)},
    }

    raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")

    results = []
    skip_count = 0

    for cfg in configs:
        best_gamma = {"best_r": float("-inf")}
        best_delta = {"best_r": float("-inf")}
        cfg_id = (
            f"resample={cfg['resample'] or 'none'}|unit={cfg['unit']}|"
            f"reref={cfg['reref']}|bandset={cfg['bandset']}|feature={cfg['feature']}"
        )
        for offset_sec in offsets:
            start_sec = start_sec_base + float(offset_sec)
            end_sec = start_sec + duration_sec
            start_idx = int(round(start_sec * sfreq))
            end_idx = int(round(end_sec * sfreq))
            if start_idx < 0 or end_idx > raw62.n_times or end_idx <= start_idx:
                skip_count += 1
                continue
            seg = raw62.get_data(start=start_idx, stop=end_idx).astype(np.float32)

            seg, fs_used = _resample_if_needed(seg, sfreq, cfg["resample"])
            if cfg["unit"] == "uV":
                seg = seg * 1e6
            if cfg["reref"] == "avg":
                seg = _apply_reref(seg)

            slices = window_slices(seg.shape[1], fs_used, args.window_sec, args.hop_sec)
            if not slices:
                skip_count += 1
                continue

            for band_name, (lo, hi) in bandsets[cfg["bandset"]].items():
                band = parse_band_spec(f"{band_name}:{lo}-{hi}")[0]
                band_data = bandpass(seg, fs_used, band)
                curve = _compute_curve(band_data, slices, cfg["feature"], float(args.eps_var))
                r0 = _pearson(official_curves[band_name], curve)
                best_r, best_lag = _best_lag_corr(
                    official_curves[band_name], curve, int(args.lag_max)
                )
                entry = {
                    "offset_sec": float(offset_sec),
                    "r0": float(r0),
                    "best_r": float(best_r),
                    "best_lag": int(best_lag),
                    "len_raw": int(curve.size),
                    "len_official": int(official_curves[band_name].size),
                }
                if band_name == "gamma":
                    if entry["best_r"] > best_gamma["best_r"]:
                        best_gamma = entry
                else:
                    if entry["best_r"] > best_delta["best_r"]:
                        best_delta = entry

        results.append(
            {
                "config_id": cfg_id,
                "config": cfg,
                "best_gamma": best_gamma,
                "best_delta": best_delta,
            }
        )

    ranked = sorted(results, key=lambda x: x["best_gamma"]["best_r"], reverse=True)
    top10 = ranked[:10]
    top3_gamma = [
        (r["config_id"], r["best_gamma"]["best_r"], r["best_gamma"]["offset_sec"], r["best_gamma"]["best_lag"])
        for r in ranked[:3]
    ]

    baseline_id = "resample=none|unit=V|reref=none|bandset=A|feature=logvar"
    baseline = next((r for r in results if r["config_id"] == baseline_id), None)

    payload.update(
        {
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
            "scan_range": {
                "offset_min": float(args.offset_min),
                "offset_max": float(args.offset_max),
                "offset_step": float(args.offset_step),
                "lag_max": int(args.lag_max),
            },
            "skip_count": int(skip_count),
            "best_config": ranked[0] if ranked else None,
            "top10": top10,
            "baseline": baseline,
        }
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Feature-Definition Sweep",
        "",
        f"- best_config: {ranked[0]['config_id'] if ranked else 'n/a'}",
        f"- best_r_gamma: {ranked[0]['best_gamma']['best_r'] if ranked else 'n/a'}",
        f"- best_offset_gamma: {ranked[0]['best_gamma']['offset_sec'] if ranked else 'n/a'}",
        f"- best_lag_gamma: {ranked[0]['best_gamma']['best_lag'] if ranked else 'n/a'}",
        f"- best_r_delta: {ranked[0]['best_delta']['best_r'] if ranked else 'n/a'}",
        f"- skip_count: {skip_count}",
        "",
        "## Baseline",
    ]
    if baseline:
        md_lines.extend(
            [
                f"- config: {baseline['config_id']}",
                f"- gamma_best_r: {baseline['best_gamma']['best_r']}",
                f"- gamma_best_offset: {baseline['best_gamma']['offset_sec']}",
                f"- gamma_best_lag: {baseline['best_gamma']['best_lag']}",
                f"- delta_best_r: {baseline['best_delta']['best_r']}",
            ]
        )
    md_lines.append("")
    md_lines.append("## Top3 gamma configs")
    for item in top3_gamma:
        md_lines.append(f"- {item}")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")

    if ranked:
        best = ranked[0]
        print(
            f"[sweep] best_config={best['config_id']} "
            f"best_r_gamma={best['best_gamma']['best_r']:.4f} "
            f"best_offset={best['best_gamma']['offset_sec']} "
            f"best_lag={best['best_gamma']['best_lag']}",
            flush=True,
        )
        print(
            f"[sweep] best_r_delta={best['best_delta']['best_r']:.4f}",
            flush=True,
        )
        print(f"[sweep] top3_gamma={top3_gamma}", flush=True)


if __name__ == "__main__":
    main()
