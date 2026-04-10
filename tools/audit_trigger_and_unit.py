import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io

from datasets.seed_raw_cnt import build_eeg62_view, load_one_cnt
from datasets.seed_raw_trials import build_trial_index, load_seed_time_points
from manifold_raw.features import bandpass, parse_band_spec, window_slices


BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


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


def _stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _check_finite(arr: np.ndarray, tag: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values detected in {tag}")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.size != b.size:
        n = min(a.size, b.size)
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
        if aa.size == 0 or bb.size == 0:
            continue
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


def _candidate_stim_channels(raw) -> List[str]:
    names = list(raw.ch_names)
    types = raw.get_channel_types()
    keys = ["STI", "STIM", "TRIG", "STATUS", "MARKER"]
    candidates = []
    for idx, name in enumerate(names):
        upper = str(name).upper()
        if types[idx] == "stim" or any(k in upper for k in keys):
            score = 0 if types[idx] == "stim" else 1
            candidates.append((score, idx, name))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [c[2] for c in candidates]


def _stim_stats(raw, candidates: List[str]) -> List[dict]:
    stats = []
    for name in candidates:
        data = raw.get_data(picks=[name]).ravel()
        nonzero_ratio = float(np.count_nonzero(data) / max(1, data.size))
        unique_count = int(np.unique(data).size)
        stats.append(
            {
                "name": name,
                "nonzero_ratio": nonzero_ratio,
                "unique_value_count": unique_count,
            }
        )
    return stats


def _extract_events(raw, candidates: List[str]) -> Tuple[np.ndarray, str, str, dict]:
    import mne

    for name in candidates:
        try:
            events = mne.find_events(raw, stim_channel=name, verbose="ERROR")
        except Exception:
            continue
        if events is not None and len(events) > 0:
            return events, "stim_channel", name, {}
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        if events is not None and len(events) > 0:
            return events, "annotations", "annotations", event_id
    except Exception:
        pass
    return np.empty((0, 3), dtype=int), "none", "", {}


def _resample_if_needed(data: np.ndarray, fs: float, resample_to: float) -> Tuple[np.ndarray, float]:
    if not resample_to or resample_to <= 0 or abs(resample_to - fs) < 1e-6:
        return data, fs
    import mne

    resampled = mne.filter.resample(
        data,
        up=resample_to,
        down=fs,
        axis=1,
        npad="auto",
        verbose="ERROR",
    )
    return resampled.astype(np.float32, copy=False), float(resample_to)


def _compute_curves(
    seg: np.ndarray,
    fs: float,
    band_specs: Dict[str, Tuple[float, float]],
    window_sec: float,
    hop_sec: float,
    feature: str,
    eps_var: float,
) -> Tuple[Dict[str, np.ndarray], int]:
    slices = window_slices(seg.shape[1], fs, window_sec, hop_sec)
    curves: Dict[str, np.ndarray] = {}
    for name, (lo, hi) in band_specs.items():
        band = parse_band_spec(f"{name}:{lo}-{hi}")[0]
        band_data = bandpass(seg, fs, band)
        vals = []
        for s, e in slices:
            win = band_data[:, s:e]
            var = np.var(win, axis=1)
            if feature == "logvar":
                feat = np.log(var + eps_var)
            elif feature == "de_gauss":
                feat = 0.5 * np.log(2 * math.pi * math.e * (var + eps_var))
            else:
                raise ValueError(f"Unknown feature: {feature}")
            vals.append(float(np.mean(feat)))
        curve = np.asarray(vals, dtype=np.float64)
        _check_finite(curve, f"raw_curve_{name}_{feature}")
        curves[name] = curve
    return curves, len(slices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger-aligned DE proxy audit.")
    parser.add_argument("--cnt-path", type=str, required=True)
    parser.add_argument("--mat-root-1s", type=str, required=True)
    parser.add_argument("--mat-root-4s", type=str, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--enable-uv", type=int, default=1)
    parser.add_argument("--resample-to", type=float, default=0.0)
    parser.add_argument("--lag-max", type=int, default=10)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--out-md", type=str, required=True)
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    payload: Dict[str, object] = {}
    try:
        raw = load_one_cnt(args.cnt_path, preload=False)
        sfreq = float(raw.info["sfreq"])
        nchan = int(raw.info["nchan"])
        tail_names = raw.ch_names[-10:]
        print(f"[raw] sfreq={sfreq:.2f} nchan={nchan} tail_ch_names={tail_names}", flush=True)
        payload["raw_info"] = {
            "sfreq": sfreq,
            "nchan": nchan,
            "tail_ch_names": tail_names,
        }

        candidates = _candidate_stim_channels(raw)
        stim_stats = _stim_stats(raw, candidates) if candidates else []
        events, strategy, stim_channel, event_id = _extract_events(raw, candidates)
        if events.size == 0:
            payload.update(
                {
                    "no_events": True,
                    "error": "NO_EVENTS_FOUND",
                    "candidate_stats": stim_stats,
                }
            )
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(payload, indent=2))
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("# Trigger Alignment + Unit Audit\n\n- NO_EVENTS_FOUND\n")
            print("NO_EVENTS_FOUND", flush=True)
            return

        events = np.asarray(events, dtype=int)
        events_head = events[:5].tolist()
        events_head_fmt = [(int(s), float(s / sfreq), int(code)) for s, _, code in events_head]
        codes, counts = np.unique(events[:, 2], return_counts=True)
        hist = sorted(zip(codes.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)
        hist_top10 = hist[:10]

        print(f"[events] strategy={strategy} stim_channel={stim_channel}", flush=True)
        print(f"[events] head5={events_head_fmt}", flush=True)
        if len(events) > 10000:
            print(f"[events] code_hist_top10={hist_top10}", flush=True)

        trigger_t0_sec = float(events[0, 0] / sfreq)
        trigger_code = int(events[0, 2])

        start_pts, _end_pts = load_seed_time_points("data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt")
        if args.trial < 1 or args.trial > len(start_pts):
            raise ValueError(f"trial out of range: {args.trial}")
        time_txt_start = int(start_pts[args.trial - 1])
        time_txt_t0_sec = float(time_txt_start / sfreq)
        offset_sec = float(trigger_t0_sec - time_txt_t0_sec)

        print(
            f"[offset] trigger_t0_sec={trigger_t0_sec:.3f} "
            f"time_txt_t0_sec={time_txt_t0_sec:.3f} offset_sec={offset_sec:.3f}",
            flush=True,
        )

        trial_input = int(args.trial)
        trial_zero = trial_input - 1
        trial_list = build_trial_index(
            args.cnt_path,
            "data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt",
            "data/SEED/SEED_EEG/SEED_stimulation.xlsx",
            time_unit="samples@1000",
        )
        trial_meta = trial_list[trial_zero] if 0 <= trial_zero < len(trial_list) else None

        payload.update(
            {
                "cnt_path": args.cnt_path,
                "subject": int(args.subject),
                "session": int(args.session),
                "trial_input": trial_input,
                "trial_zero": trial_zero,
                "trial_meta_t_start_s": float(trial_meta.t_start_s) if trial_meta else None,
                "trial_meta_t_end_s": float(trial_meta.t_end_s) if trial_meta else None,
                "sfreq": sfreq,
                "stim_candidates": stim_stats,
                "event_strategy": strategy,
                "stim_channel": stim_channel,
                "event_id": event_id,
                "events_head5": events_head_fmt,
                "events_count": int(len(events)),
                "events_code_hist_top10": hist_top10,
                "trigger_t0_sec": trigger_t0_sec,
                "trigger_code": trigger_code,
                "time_txt_start": time_txt_start,
                "time_txt_t0_sec": time_txt_t0_sec,
                "offset_sec": offset_sec,
            }
        )

        raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")

        durations = {"1s": 235.0, "4s": 232.0}
        anchors = {
            "trigger": trigger_t0_sec,
            "time_txt": time_txt_t0_sec,
        }
        bandsets = {
            "A": {"delta": (1.0, 4.0), "gamma": (31.0, 50.0)},
            "B": {"delta": (1.0, 3.0), "gamma": (30.0, 50.0)},
        }
        features = ["logvar", "de_gauss"]
        eps_var = 1e-12

        official = {}
        official_curves: Dict[str, Dict[str, np.ndarray]] = {}
        for key, root in [("1s", args.mat_root_1s), ("4s", args.mat_root_4s)]:
            root_path = Path(root)
            mat_path = _resolve_mat_by_session(root_path, args.subject, args.session)
            mat = scipy.io.loadmat(mat_path)
            key_name = f"de_LDS{trial_input}"
            if key_name not in mat:
                raise KeyError(f"{key_name} not found in {mat_path}")
            arr = _normalize_trial_array(mat[key_name])
            T_off = int(arr.shape[1])
            band_index_map = {name: _band_index(name) for name in ("delta", "gamma")}
            curves = {}
            stats = {}
            for name, idx in band_index_map.items():
                curve = arr[:, :, idx].mean(axis=0)
                curve = curve.astype(np.float64)
                _check_finite(curve, f"official_curve_{name}_{key}")
                curves[name] = curve
                stats[name] = _stats(curve)
            official_curves[key] = curves
            official[key] = {
                "root": str(root_path),
                "mat_path": str(mat_path),
                "key_name": key_name,
                "T_off": T_off,
                "band_index_map": band_index_map,
                "stats": stats,
                "head10": {k: v[:10].tolist() for k, v in curves.items()},
            }

        results: Dict[str, Dict[str, dict]] = {}
        best_trigger_gamma: Dict[str, dict] = {}
        baseline_gamma: Dict[str, dict] = {}

        for anchor_name, t0 in anchors.items():
            results[anchor_name] = {}
            for dur_key, dur_sec in durations.items():
                start = int(round(t0 * sfreq))
                end = int(round((t0 + dur_sec) * sfreq))
                end = min(end, raw62.n_times)
                if end <= start:
                    raise ValueError(f"Empty segment for {anchor_name} {dur_key}")
                seg = raw62.get_data(start=start, stop=end).astype(np.float32)
                if args.enable_uv:
                    seg = seg * 1e6
                seg, fs_used = _resample_if_needed(seg, sfreq, float(args.resample_to))

                results[anchor_name][dur_key] = {}
                for bandset_name, band_spec in bandsets.items():
                    results[anchor_name][dur_key][bandset_name] = {}
                    for feat in features:
                        curves, n_win = _compute_curves(
                            seg,
                            fs_used,
                            band_spec,
                            1.0 if dur_key == "1s" else 4.0,
                            1.0 if dur_key == "1s" else 4.0,
                            feat,
                            eps_var,
                        )
                        entry = {}
                        for band_name, curve in curves.items():
                            r0 = _pearson(official_curves[dur_key][band_name], curve)
                            best_r, best_lag = _best_lag_corr(
                                official_curves[dur_key][band_name], curve, int(args.lag_max)
                            )
                            entry[band_name] = {
                                "len_raw": int(curve.size),
                                "stats": _stats(curve),
                                "r0": float(r0),
                                "best_r": float(best_r),
                                "best_lag": int(best_lag),
                                "head10": curve[:10].tolist(),
                                "n_windows": int(n_win),
                            }
                        results[anchor_name][dur_key][bandset_name][feat] = entry

                # capture best trigger gamma per duration
                if anchor_name == "trigger":
                    best = None
                    best_cfg = None
                    for bandset_name, feat_map in results[anchor_name][dur_key].items():
                        for feat, band_map in feat_map.items():
                            gamma = band_map["gamma"]
                            if best is None or gamma["best_r"] > best:
                                best = gamma["best_r"]
                                best_cfg = {
                                    "bandset": bandset_name,
                                    "feature": feat,
                                    "best_r": gamma["best_r"],
                                    "best_lag": gamma["best_lag"],
                                }
                    best_trigger_gamma[dur_key] = best_cfg or {}
                if anchor_name == "time_txt":
                    base = results[anchor_name][dur_key]["A"]["logvar"]["gamma"]
                    baseline_gamma[dur_key] = {
                        "bandset": "A",
                        "feature": "logvar",
                        "best_r": base["best_r"],
                        "best_lag": base["best_lag"],
                    }

        payload.update(
            {
                "official": official,
                "results": results,
                "best_trigger_gamma": best_trigger_gamma,
                "baseline_time_txt_gamma": baseline_gamma,
                "enable_uv": int(bool(args.enable_uv)),
                "resample_to": float(args.resample_to),
                "eps_var": eps_var,
            }
        )

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))

        md_lines = [
            "# Trigger Alignment + Unit Audit",
            "",
            f"- trigger strategy: {strategy} ({stim_channel})",
            f"- trigger_t0_sec: {trigger_t0_sec:.3f}",
            f"- time_txt_t0_sec: {time_txt_t0_sec:.3f}",
            f"- offset_sec: {offset_sec:.3f}",
            f"- recommended_pass: {abs(offset_sec) >= 3.0 or (best_trigger_gamma.get('1s', {}).get('best_r', 0) >= 0.5)}",
            "",
            "## Best trigger-anchored gamma",
        ]
        for dur_key in ("1s", "4s"):
            cfg = best_trigger_gamma.get(dur_key, {})
            md_lines.append(
                f"- {dur_key}: best_r={cfg.get('best_r')} best_lag={cfg.get('best_lag')} "
                f"bandset={cfg.get('bandset')} feature={cfg.get('feature')}"
            )
        md_lines.append("")
        md_lines.append("## Baseline time.txt gamma (bandset A + logvar)")
        for dur_key in ("1s", "4s"):
            cfg = baseline_gamma.get(dur_key, {})
            md_lines.append(
                f"- {dur_key}: best_r={cfg.get('best_r')} best_lag={cfg.get('best_lag')}"
            )
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md_lines) + "\n")

        print(
            f"[best_gamma] 1s best_r={best_trigger_gamma['1s']['best_r']:.4f} "
            f"best_lag={best_trigger_gamma['1s']['best_lag']} "
            f"cfg={best_trigger_gamma['1s']['bandset']}/{best_trigger_gamma['1s']['feature']}",
            flush=True,
        )
        print(
            f"[best_gamma] 4s best_r={best_trigger_gamma['4s']['best_r']:.4f} "
            f"best_lag={best_trigger_gamma['4s']['best_lag']} "
            f"cfg={best_trigger_gamma['4s']['bandset']}/{best_trigger_gamma['4s']['feature']}",
            flush=True,
        )
        print(
            f"[baseline_gamma] 1s best_r={baseline_gamma['1s']['best_r']:.4f} "
            f"best_lag={baseline_gamma['1s']['best_lag']}",
            flush=True,
        )
        print(
            f"[baseline_gamma] 4s best_r={baseline_gamma['4s']['best_r']:.4f} "
            f"best_lag={baseline_gamma['4s']['best_lag']}",
            flush=True,
        )

    except SystemExit:
        raise
    except Exception as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[audit] error: {type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
