from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.seed_raw_cnt import _load_locs_names, build_eeg62_view, load_one_raw
from manifold_raw.features import bandpass, parse_band_spec, window_slices


QUANTILES = [1, 5, 25, 50, 75, 95, 99]
NEAR_ZERO_THRESHOLDS = [1e-10, 1e-9]


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _quantile_dict(values: np.ndarray) -> Dict[str, float]:
    q = np.percentile(values, QUANTILES)
    return {f"p{int(k)}": float(v) for k, v in zip(QUANTILES, q)}


def _format_uv(values: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v * 1e6) for k, v in values.items()}


def _near_zero(std: np.ndarray, names: List[str]) -> List[Dict[str, object]]:
    entries = []
    for thr in NEAR_ZERO_THRESHOLDS:
        idx = [i for i, v in enumerate(std) if v < thr]
        entries.append(
            {
                "threshold": float(thr),
                "count": int(len(idx)),
                "names": [names[i] for i in idx],
            }
        )
    return entries


def _top_bottom(std: np.ndarray, names: List[str], k: int = 10) -> Dict[str, List[Dict[str, object]]]:
    order = np.argsort(std)
    bottom_idx = order[:k]
    top_idx = order[::-1][:k]
    bottom = [
        {
            "name": names[int(i)],
            "std_v": float(std[int(i)]),
            "std_uV": float(std[int(i)] * 1e6),
        }
        for i in bottom_idx
    ]
    top = [
        {
            "name": names[int(i)],
            "std_v": float(std[int(i)]),
            "std_uV": float(std[int(i)] * 1e6),
        }
        for i in top_idx
    ]
    return {"top10": top, "bottom10": bottom}


def _summarize(tag: str, data: np.ndarray, names: List[str]) -> Dict[str, object]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    std = arr.std(axis=1)
    ptp = np.ptp(arr, axis=1)
    std_q = _quantile_dict(std)
    ptp_q = _quantile_dict(ptp)
    top_bottom = _top_bottom(std, names)
    summary = {
        "tag": tag,
        "n_chan": int(arr.shape[0]),
        "n_samples": int(arr.shape[1]),
        "std_v": std_q,
        "std_uV": _format_uv(std_q),
        "ptp_v": ptp_q,
        "ptp_uV": _format_uv(ptp_q),
        "near_zero": _near_zero(std, names),
        "top10_std_channels": top_bottom["top10"],
        "bottom10_std_channels": top_bottom["bottom10"],
    }
    return summary


def _uv_fmt(v: float) -> str:
    return f"{v:.6e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnt",
        type=str,
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/1_1.cnt",
    )
    parser.add_argument("--sec", type=float, default=10.0)
    parser.add_argument(
        "--out",
        type=str,
        default="logs/audit_scale_distributions_1_1.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locs", type=str, default="data/SEED/channel_62_pos.locs")
    parser.add_argument(
        "--bands",
        type=str,
        default="delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    )
    parser.add_argument("--data-format", type=str, default=None)
    args = parser.parse_args()

    cnt_path = _resolve_path(args.cnt)
    locs_path = _resolve_path(args.locs)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_one_raw(str(cnt_path), backend="cnt", preload=False, data_format=args.data_format)
    sfreq = float(raw.info.get("sfreq", 0.0))
    stop = int(round(float(args.sec) * sfreq)) if sfreq > 0 else 0
    if stop <= 0:
        raise ValueError(f"Invalid stop={stop}; check sec={args.sec} sfreq={sfreq}")

    raw66 = raw.get_data(start=0, stop=stop)
    raw66_names = list(raw.ch_names)

    raw62, _meta = build_eeg62_view(raw, locs_path=str(locs_path))
    raw62 = raw62.get_data(start=0, stop=stop)
    raw62_names = _load_locs_names(str(locs_path))

    bands = parse_band_spec(args.bands)
    band_full = {}
    for band in bands:
        band_full[band.name] = bandpass(raw62, sfreq, band)

    win_slices = window_slices(raw62.shape[1], sfreq, 4.0, 4.0)
    if not win_slices:
        raise ValueError("window_slices returned empty list")
    rng = np.random.default_rng(int(args.seed))
    win_idx = int(rng.integers(len(win_slices))) if len(win_slices) > 1 else 0
    w_start, w_end = win_slices[win_idx]

    raw62_win = raw62[:, w_start:w_end]
    band_win = {name: data[:, w_start:w_end] for name, data in band_full.items()}

    report = {
        "cnt_path": str(cnt_path),
        "locs_path": str(locs_path),
        "sec": float(args.sec),
        "sfreq": float(sfreq),
        "stop": int(stop),
        "window_sec": 4.0,
        "hop_sec": 4.0,
        "window_index": int(win_idx),
        "window_start": int(w_start),
        "window_end": int(w_end),
        "bands": [b.name for b in bands],
        "raw66": _summarize("raw66", raw66, raw66_names),
        "raw62": _summarize("raw62", raw62, raw62_names),
        "raw62_win": _summarize("raw62_win", raw62_win, raw62_names),
        "band_full": {name: _summarize(f"band_{name}", data, raw62_names) for name, data in band_full.items()},
        "band_win": {name: _summarize(f"band_{name}_win", data, raw62_names) for name, data in band_win.items()},
    }

    raw66_p50 = report["raw66"]["std_uV"]["p50"]
    raw66_p95 = report["raw66"]["std_uV"]["p95"]
    raw62_p50 = report["raw62"]["std_uV"]["p50"]
    raw62_p95 = report["raw62"]["std_uV"]["p95"]
    raw62_win_p50 = report["raw62_win"]["std_uV"]["p50"]
    raw62_win_p95 = report["raw62_win"]["std_uV"]["p95"]

    print(f"[summary] raw66 std_uV p50={_uv_fmt(raw66_p50)} p95={_uv_fmt(raw66_p95)}")
    print(f"[summary] raw62 std_uV p50={_uv_fmt(raw62_p50)} p95={_uv_fmt(raw62_p95)}")
    for band in bands:
        p50 = report["band_full"][band.name]["std_uV"]["p50"]
        p95 = report["band_full"][band.name]["std_uV"]["p95"]
        print(f"[summary] bandpass_{band.name} std_uV p50={_uv_fmt(p50)} p95={_uv_fmt(p95)}")
    print(f"[summary] raw62_win std_uV p50={_uv_fmt(raw62_win_p50)} p95={_uv_fmt(raw62_win_p95)}")

    band_p95_uv = [report["band_full"][band.name]["std_uV"]["p95"] for band in bands]
    bandpass_collapsed = all(v < 0.1 for v in band_p95_uv)

    ratio_p50 = max(raw62_p50 / max(raw62_win_p50, 1e-12), raw62_win_p50 / max(raw62_p50, 1e-12))
    ratio_p95 = max(raw62_p95 / max(raw62_win_p95, 1e-12), raw62_win_p95 / max(raw62_p95, 1e-12))
    window_mismatch = ratio_p50 > 2.0 or ratio_p95 > 2.0

    long_tail = raw62_p95 >= 10.0 * max(raw62_p50, 1e-12) and raw62_p95 >= 1.0

    if bandpass_collapsed:
        conclusion = "bandpass p95 塌缩 / 需检查 bandpass 输入链路"
    elif window_mismatch:
        conclusion = "window 切片异常 / 需检查 window_slices 或索引轴"
    elif long_tail:
        conclusion = "长尾导致 median 低 / bandpass p95 未塌缩"
    else:
        conclusion = "median 低但未见 bandpass p95 塌缩 / 分布可能偏弱"

    report["conclusion"] = conclusion
    print(f"[summary] conclusion: {conclusion}")
    print(f"[summary] json_report={out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    if hasattr(raw, "close"):
        raw.close()


if __name__ == "__main__":
    main()
