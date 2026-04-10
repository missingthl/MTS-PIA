import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.seed_raw_cnt import build_eeg62_view, load_one_cnt
from datasets.seed_raw_trials import build_trial_index
from tools.alignment_core import (
    AlignmentError,
    compute_raw_de_proxy_series,
    load_official_de_series,
    pearson_r,
    scan_offsets,
)


def _band_smooth_param(mode: str, ema_alpha: float, kalman_q: float, kalman_r: float):
    mode = (mode or "none").strip().lower()
    if mode == "ema":
        return float(ema_alpha)
    if mode == "kalman":
        if kalman_r <= 0:
            raise ValueError("kalman_r must be > 0")
        return float(kalman_q) / float(kalman_r)
    return None


def _stats(values: List[float], thresholds: List[float]) -> dict:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
            "pass_rates": {str(t): 0.0 for t in thresholds},
        }
    pass_rates = {str(t): float(np.mean(vals >= t)) for t in thresholds}
    return {
        "count": int(vals.size),
        "min": float(vals.min()),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(vals.max()),
        "pass_rates": pass_rates,
    }


def _offset_hist(values: List[float], step: float) -> dict:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {}
    counts = Counter()
    for v in vals:
        key = round(float(v) / step) * step
        counts[f"{key:.1f}"] += 1
    return dict(sorted(counts.items(), key=lambda x: float(x[0])))


def _mode_and_coverage(values: List[float], step: float) -> Tuple[Optional[float], float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None, 0.0
    counts = Counter()
    for v in vals:
        key = round(float(v) / step) * step
        counts[key] += 1
    mode_val, mode_count = counts.most_common(1)[0]
    return float(mode_val), float(mode_count) / float(len(vals))


def _group_stats(
    records: List[dict],
    group_key: str,
    r_key: str,
    offset_key: str,
    thresholds: List[float],
    step: float,
) -> dict:
    grouped = defaultdict(list)
    for row in records:
        grouped[str(row[group_key])].append(row)
    out = {}
    for key, rows in grouped.items():
        r_vals = [row.get(r_key) for row in rows]
        offset_vals = [row.get(offset_key) for row in rows]
        out[key] = {
            "count": len(rows),
            "r_stats": _stats(r_vals, thresholds),
            "offset_hist": _offset_hist(offset_vals, step),
        }
    return out


def _build_trial_list(cnt_root: Path, time_unit: str) -> List[dict]:
    trials = []
    cnt_files = sorted(cnt_root.glob("*.cnt"))
    for cnt_path in cnt_files:
        time_txt = cnt_path.parent / "time.txt"
        stim_xlsx = Path("data/SEED/SEED_EEG/SEED_stimulation.xlsx")
        t_list = build_trial_index(str(cnt_path), str(time_txt), str(stim_xlsx), time_unit=time_unit)
        for t in t_list:
            trials.append(
                {
                    "subject": str(t.subject),
                    "session": int(t.session),
                    "trial": int(t.trial),
                    "label": int(t.label),
                    "t_start_s": float(t.t_start_s),
                    "t_end_s": float(t.t_end_s),
                    "cnt_path": str(cnt_path),
                    "trial_meta": t,
                }
            )
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Full alignment audit for de_LDS.")
    parser.add_argument("--cnt-root", required=True)
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--window-sec", type=float, required=True)
    parser.add_argument("--hop-sec", type=float, required=True)
    parser.add_argument("--target-key", type=str, default="de_LDS")
    parser.add_argument("--time-unit", type=str, default="samples@1000")
    parser.add_argument("--fixed-offset", type=float, default=-3.0)
    parser.add_argument("--scan-offset-min", type=float, default=-6.0)
    parser.add_argument("--scan-offset-max", type=float, default=6.0)
    parser.add_argument("--scan-offset-step", type=float, default=0.5)
    parser.add_argument("--bands", type=str, default="delta,gamma")
    parser.add_argument("--smooth-gamma", type=str, default="ema")
    parser.add_argument("--smooth-delta", type=str, default="none")
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--kalman-q", type=float, default=1e-4)
    parser.add_argument("--kalman-r", type=float, default=1.0)
    parser.add_argument("--eps-var", type=float, default=1e-12)
    parser.add_argument("--lag-max", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    cnt_root = Path(args.cnt_root)
    mat_root = args.mat_root
    bands = [b.strip().lower() for b in args.bands.split(",") if b.strip()]
    if not bands:
        raise ValueError("bands list is empty")

    trials = _build_trial_list(cnt_root, args.time_unit)
    total_trials = len(trials)
    if total_trials == 0:
        raise ValueError("no trials discovered")

    offsets = np.arange(
        args.scan_offset_min,
        args.scan_offset_max + 1e-9,
        args.scan_offset_step,
    )

    gamma_param = _band_smooth_param(args.smooth_gamma, args.ema_alpha, args.kalman_q, args.kalman_r)
    delta_param = _band_smooth_param(args.smooth_delta, args.ema_alpha, args.kalman_q, args.kalman_r)
    smooth_params = {"gamma": gamma_param, "delta": delta_param}
    smooth_modes = {"gamma": args.smooth_gamma, "delta": args.smooth_delta}

    records = []
    start_time = time.time()
    processed = 0

    # group by CNT file to reuse raw loading
    by_cnt = defaultdict(list)
    for row in trials:
        by_cnt[row["cnt_path"]].append(row)

    for cnt_path, rows in sorted(by_cnt.items()):
        raw = load_one_cnt(cnt_path, preload=False)
        raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
        sfreq = float(raw62.info["sfreq"])

        for row in rows:
            trial_input = int(row["trial"]) + 1
            record = {
                "subject": row["subject"],
                "session": row["session"],
                "trial": int(row["trial"]),
                "label": row["label"],
                "cnt_path": cnt_path,
                "r_fixed_gamma": None,
                "r_fixed_delta": None,
                "best_r_gamma": None,
                "best_offset_gamma": None,
                "best_lag_gamma": None,
                "best_r_delta": None,
                "best_offset_delta": None,
                "best_lag_delta": None,
                "error_codes": [],
            }

            for band in bands:
                smooth_mode = smooth_modes.get(band, "none")
                smooth_param = smooth_params.get(band)
                try:
                    official_curve = load_official_de_series(
                        mat_root=mat_root,
                        subject=int(row["subject"]),
                        session=int(row["session"]),
                        trial=trial_input,
                        band=band,
                        target_key=args.target_key,
                    )
                except AlignmentError as exc:
                    record["error_codes"].append(f"official_{band}:{exc.code}")
                    continue

                duration_sec = float(len(official_curve)) * float(args.hop_sec)

                # fixed offset
                try:
                    raw_curve = compute_raw_de_proxy_series(
                        cnt_path=cnt_path,
                        trial_input=trial_input,
                        band=band,
                        window_sec=args.window_sec,
                        hop_sec=args.hop_sec,
                        offset_sec=float(args.fixed_offset),
                        smooth_mode=smooth_mode,
                        smooth_param=smooth_param,
                        time_unit=args.time_unit,
                        duration_sec=duration_sec,
                        eps_var=float(args.eps_var),
                        raw62=raw62,
                        sfreq=sfreq,
                        trial_meta=row["trial_meta"],
                    )
                    r_fixed = pearson_r(official_curve, raw_curve)
                    if np.isnan(r_fixed):
                        raise AlignmentError("nan_r", "nan fixed r")
                    record[f"r_fixed_{band}"] = float(r_fixed)
                except AlignmentError as exc:
                    record["error_codes"].append(f"fixed_{band}:{exc.code}")

                # scan offsets
                best_r, best_offset, best_lag, scan_errors = scan_offsets(
                    offsets=offsets,
                    cnt_path=cnt_path,
                    trial_input=trial_input,
                    band=band,
                    window_sec=args.window_sec,
                    hop_sec=args.hop_sec,
                    time_unit=args.time_unit,
                    smooth_mode=smooth_mode,
                    smooth_param=smooth_param,
                    duration_sec=duration_sec,
                    eps_var=float(args.eps_var),
                    official_curve=official_curve,
                    lag_max=int(args.lag_max),
                    raw62=raw62,
                    sfreq=sfreq,
                    trial_meta=row["trial_meta"],
                )
                if best_offset is None or not np.isfinite(best_r):
                    record["error_codes"].append(f"scan_{band}:no_valid")
                else:
                    record[f"best_r_{band}"] = float(best_r)
                    record[f"best_offset_{band}"] = float(best_offset)
                    record[f"best_lag_{band}"] = int(best_lag)
                if scan_errors:
                    record.setdefault("scan_errors", {})[band] = scan_errors

            records.append(record)
            processed += 1
            if args.progress_interval and processed % int(args.progress_interval) == 0:
                elapsed = time.time() - start_time
                avg = elapsed / float(processed)
                remaining = avg * float(total_trials - processed)
                print(
                    f"[progress] {processed}/{total_trials} elapsed={elapsed:.1f}s "
                    f"eta={remaining:.1f}s",
                    flush=True,
                )

        if hasattr(raw, "close"):
            raw.close()

    thresholds = [0.6, 0.7, 0.8]
    summary = {
        "meta": {
            "cnt_root": str(cnt_root),
            "mat_root": mat_root,
            "window_sec": float(args.window_sec),
            "hop_sec": float(args.hop_sec),
            "target_key": args.target_key,
            "time_unit": args.time_unit,
            "fixed_offset": float(args.fixed_offset),
            "scan_offset_min": float(args.scan_offset_min),
            "scan_offset_max": float(args.scan_offset_max),
            "scan_offset_step": float(args.scan_offset_step),
            "bands": bands,
            "smooth_modes": smooth_modes,
            "smooth_params": smooth_params,
            "lag_max": int(args.lag_max),
        },
        "counts": {"total_trials": total_trials},
    }

    for band in bands:
        fixed_vals = [r.get(f"r_fixed_{band}") for r in records]
        best_vals = [r.get(f"best_r_{band}") for r in records]
        best_offsets = [r.get(f"best_offset_{band}") for r in records]
        mode_val, mode_cov = _mode_and_coverage(best_offsets, float(args.scan_offset_step))
        summary[f"fixed_{band}"] = _stats(fixed_vals, thresholds)
        summary[f"best_{band}"] = _stats(best_vals, thresholds)
        summary[f"best_offset_{band}"] = {
            "mode": mode_val,
            "coverage": mode_cov,
            "hist": _offset_hist(best_offsets, float(args.scan_offset_step)),
        }
        summary[f"by_session_{band}"] = _group_stats(
            records,
            "session",
            f"best_r_{band}",
            f"best_offset_{band}",
            thresholds,
            float(args.scan_offset_step),
        )
        summary[f"by_subject_{band}"] = _group_stats(
            records,
            "subject",
            f"best_r_{band}",
            f"best_offset_{band}",
            thresholds,
            float(args.scan_offset_step),
        )

    # failure list by best_r_gamma
    fail_sorted = sorted(
        records,
        key=lambda r: r.get("best_r_gamma") if r.get("best_r_gamma") is not None else -1.0,
    )
    summary["fail_top20_gamma"] = [
        {
            "subject": r["subject"],
            "session": r["session"],
            "trial": r["trial"],
            "best_r_gamma": r.get("best_r_gamma"),
            "best_offset_gamma": r.get("best_offset_gamma"),
            "cnt_path": r["cnt_path"],
        }
        for r in fail_sorted[:20]
    ]

    out_json = Path(args.out_json)
    out_summary = Path(args.out_summary)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2))
    out_summary.write_text(json.dumps(summary, indent=2))

    # build markdown report
    gamma_mode = summary.get("best_offset_gamma", {}).get("mode")
    gamma_cov = summary.get("best_offset_gamma", {}).get("coverage", 0.0)
    gamma_stats = summary.get("best_gamma", {})
    delta_stats = summary.get("best_delta", {})
    delta_p50 = delta_stats.get("p50")
    gamma_p50 = gamma_stats.get("p50")
    delta_warn = ""
    if gamma_p50 is not None and delta_p50 is not None:
        if (gamma_p50 - delta_p50) >= 0.3:
            delta_warn = "delta_p50 << gamma_p50: delta band may be noise source."

    mode_ok = False
    if gamma_mode is not None:
        mode_ok = abs(float(gamma_mode) - (-3.0)) <= 0.5 and gamma_cov >= 0.8

    conclusion_lines = [
        "## Conclusion",
        f"- global_offset_supported: {mode_ok}",
        f"- gamma_p50: {gamma_p50} pass_rate_0.6: {gamma_stats.get('pass_rates', {}).get('0.6')}",
        f"- delta_risk: {delta_warn or 'none'}",
        "",
    ]

    md_lines = [
        "# Alignment Full Audit Report",
        "",
        f"- total_trials: {total_trials}",
        f"- fixed_offset: {args.fixed_offset}",
        f"- gamma_mode: {gamma_mode} coverage={gamma_cov:.3f}",
        f"- mode_supports_global_offset: {mode_ok}",
        "",
        *conclusion_lines,
        "## Gamma (best-offset) stats",
        f"- p50: {gamma_stats.get('p50')} p90: {gamma_stats.get('p90')} p95: {gamma_stats.get('p95')}",
        f"- pass_rate_0.6: {gamma_stats.get('pass_rates', {}).get('0.6')} "
        f"pass_rate_0.7: {gamma_stats.get('pass_rates', {}).get('0.7')} "
        f"pass_rate_0.8: {gamma_stats.get('pass_rates', {}).get('0.8')}",
        "",
        "## Delta (best-offset) stats",
        f"- p50: {delta_stats.get('p50')} p90: {delta_stats.get('p90')} p95: {delta_stats.get('p95')}",
        f"- pass_rate_0.6: {delta_stats.get('pass_rates', {}).get('0.6')}",
        f"- note: {delta_warn or 'n/a'}",
        "",
        "## Top20 failed trials (gamma best_r lowest)",
    ]
    for row in summary["fail_top20_gamma"]:
        md_lines.append(
            f"- sub={row['subject']} sess={row['session']} trial={row['trial']} "
            f"best_r_gamma={row['best_r_gamma']} best_offset={row['best_offset_gamma']} "
            f"path={row['cnt_path']}"
        )
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"[done] records={out_json} summary={out_summary} report={out_md}")


if __name__ == "__main__":
    main()
