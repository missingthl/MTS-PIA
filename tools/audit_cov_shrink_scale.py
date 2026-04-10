from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from manifold_raw.features import bandpass, cov_shrink, parse_band_spec, window_slices
from manifold_raw.spd_eps import compute_spd_eps


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"p5": math.nan, "p50": math.nan, "p95": math.nan}
    q = np.percentile(values, [5, 50, 95])
    return {"p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def _global_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


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


def _cov_stats(mat: np.ndarray) -> Dict[str, float]:
    eigvals = np.linalg.eigvalsh(mat)
    eig_min = float(np.min(eigvals))
    eig_max = float(np.max(eigvals))
    trace = float(np.trace(mat))
    diag = np.diag(mat)
    diag_mean = float(np.mean(diag))
    cond = eig_max / max(eig_min, 1e-12)
    return {
        "diag_mean": diag_mean,
        "offdiag_mean_abs": _offdiag_mean_abs(mat),
        "diag_energy_ratio": _diag_energy_ratio(mat),
        "eig_min": eig_min,
        "eig_max": eig_max,
        "cond": float(cond),
        "trace": trace,
    }


def _aggregate_stats(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    metrics = rows[0].keys()
    out = {}
    for key in metrics:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = _quantiles(vals)
    return out


def _format_line(label: str, p50: float, p95: float) -> str:
    return f"{label} p50={p50:.6e} p95={p95:.6e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnt",
        type=str,
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/1_1.cnt",
    )
    parser.add_argument("--sec", type=float, default=10.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument(
        "--bands",
        type=str,
        default="delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    )
    parser.add_argument("--cov-method", type=str, default="shrinkage_oas")
    parser.add_argument("--spd-eps", type=float, default=1e-5)
    parser.add_argument(
        "--spd-eps-mode",
        type=str,
        default="relative_trace",
        choices=["absolute", "relative_trace", "relative_diag"],
    )
    parser.add_argument("--spd-eps-alpha", type=float, default=1e-2)
    parser.add_argument("--spd-eps-floor-mult", type=float, default=1e-6)
    parser.add_argument("--spd-eps-ceil-mult", type=float, default=1e-1)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default="logs/audit_cov_shrink_scale_1_1.json",
    )
    parser.add_argument("--locs", type=str, default="data/SEED/channel_62_pos.locs")
    parser.add_argument("--data-format", type=str, default=None)
    args = parser.parse_args()

    cnt_path = _resolve_path(args.cnt)
    locs_path = _resolve_path(args.locs)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_one_raw(str(cnt_path), backend="cnt", preload=False, data_format=args.data_format)
    raw62, _meta = build_eeg62_view(raw, locs_path=str(locs_path))
    sfreq = float(raw62.info.get("sfreq", 0.0))
    stop = int(round(float(args.sec) * sfreq)) if sfreq > 0 else 0
    if stop <= 0:
        raise ValueError(f"Invalid stop={stop}; check sec={args.sec} sfreq={sfreq}")

    seg = raw62.get_data(start=0, stop=stop).astype(np.float64, copy=False)
    n_samples = seg.shape[1]
    win_slices = window_slices(n_samples, sfreq, args.window_sec, args.hop_sec)
    if not win_slices:
        raise ValueError("window_slices returned empty list")

    rng = np.random.default_rng(int(args.seed))
    n_pick = min(int(args.num_windows), len(win_slices))
    pick_idx = rng.choice(len(win_slices), size=n_pick, replace=False)
    pick_idx = sorted(int(i) for i in pick_idx)
    picked_windows = [win_slices[i] for i in pick_idx]

    bands = parse_band_spec(args.bands)
    band_full = {band.name: bandpass(seg, sfreq, band) for band in bands}

    report = {
        "cnt_path": str(cnt_path),
        "locs_path": str(locs_path),
        "sec": float(args.sec),
        "sfreq": float(sfreq),
        "stop": int(stop),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "num_windows": int(n_pick),
        "window_indices": pick_idx,
        "window_slices": [{"start": int(s), "end": int(e)} for s, e in picked_windows],
        "bands": [b.name for b in bands],
        "cov_method": str(args.cov_method),
        "spd_eps": float(args.spd_eps),
        "spd_eps_mode": str(args.spd_eps_mode),
        "spd_eps_alpha": float(args.spd_eps_alpha),
        "spd_eps_floor_mult": float(args.spd_eps_floor_mult),
        "spd_eps_ceil_mult": float(args.spd_eps_ceil_mult),
        "band_reports": {},
    }

    conclusion_inputs = []
    ratio_eps_all = []
    rel_delta_all = []
    eps_used_all = []
    eig_min_before_all = []
    eig_min_after_all = []
    oas_diag_energy_all = []
    oas_diag_mean_all = []
    scm_diag_mean_all = []
    oas_offdiag_all = []
    scm_offdiag_all = []

    for band in bands:
        band_name = band.name
        band_data = band_full[band_name]
        per_window = []
        oas_cov_stats = []
        scm_cov_stats = []
        oas_after_stats = []
        scm_after_stats = []
        ratio_eps = []
        rel_delta = []
        eps_used = []
        oas_vs_scm = []

        for w_start, w_end in picked_windows:
            window = band_data[:, w_start:w_end]
            std = window.std(axis=1)
            var = window.var(axis=1)

            input_stats = {
                "shape": [int(window.shape[0]), int(window.shape[1])],
                "std_uV": {k: v * 1e6 for k, v in _quantiles(std).items()},
                "var_v2": _quantiles(var),
                "global": _global_stats(window),
            }

            cov_oas = cov_shrink(window, method=args.cov_method)
            cov_scm = cov_shrink(window, method="scm")

            stats_oas = _cov_stats(cov_oas)
            stats_scm = _cov_stats(cov_scm)

            eps_oas, base_oas = compute_spd_eps(
                cov_oas,
                mode=args.spd_eps_mode,
                absolute=float(args.spd_eps),
                alpha=float(args.spd_eps_alpha),
                floor_mult=float(args.spd_eps_floor_mult),
                ceil_mult=float(args.spd_eps_ceil_mult),
            )
            eps_scm, _base_scm = compute_spd_eps(
                cov_scm,
                mode=args.spd_eps_mode,
                absolute=float(args.spd_eps),
                alpha=float(args.spd_eps_alpha),
                floor_mult=float(args.spd_eps_floor_mult),
                ceil_mult=float(args.spd_eps_ceil_mult),
            )

            cov_oas_after = 0.5 * (cov_oas + cov_oas.T) + float(eps_oas) * np.eye(cov_oas.shape[0])
            cov_scm_after = 0.5 * (cov_scm + cov_scm.T) + float(eps_scm) * np.eye(cov_scm.shape[0])

            stats_oas_after = _cov_stats(cov_oas_after)
            stats_scm_after = _cov_stats(cov_scm_after)

            diag_mean_oas = stats_oas["diag_mean"]
            diag_mean_scm = stats_scm["diag_mean"]
            offdiag_oas = stats_oas["offdiag_mean_abs"]
            offdiag_scm = stats_scm["offdiag_mean_abs"]
            diag_energy_oas = stats_oas["diag_energy_ratio"]
            diag_energy_scm = stats_scm["diag_energy_ratio"]

            ratio = float(eps_oas) / max(diag_mean_oas, 1e-30)
            diff = cov_oas_after - cov_oas
            rel = float(np.linalg.norm(diff, ord="fro") / max(np.linalg.norm(cov_oas, ord="fro"), 1e-30))

            oas_vs_scm.append(
                {
                    "diag_mean_ratio": diag_mean_oas / max(diag_mean_scm, 1e-30),
                    "offdiag_mean_abs_ratio": offdiag_oas / max(offdiag_scm, 1e-30),
                    "diag_energy_ratio_diff": diag_energy_oas - diag_energy_scm,
                }
            )

            per_window.append(
                {
                    "window": {"start": int(w_start), "end": int(w_end)},
                    "input": input_stats,
                    "cov_oas_before": stats_oas,
                    "cov_scm_before": stats_scm,
                    "cov_oas_after": stats_oas_after,
                    "cov_scm_after": stats_scm_after,
                    "ratio_eps_domination": ratio,
                    "relative_delta": rel,
                    "eps_used": float(eps_oas),
                    "eps_base": float(base_oas),
                }
            )

            oas_cov_stats.append(stats_oas)
            scm_cov_stats.append(stats_scm)
            oas_after_stats.append(stats_oas_after)
            scm_after_stats.append(stats_scm_after)
            ratio_eps.append(ratio)
            rel_delta.append(rel)
            eps_used.append(float(eps_oas))

        input_p50 = np.median([w["input"]["std_uV"]["p50"] for w in per_window])
        input_p95 = np.median([w["input"]["std_uV"]["p95"] for w in per_window])
        conclusion_inputs.append({"band": band_name, "p50": input_p50, "p95": input_p95})

        oas_diag_energy_all.append(np.median([row["diag_energy_ratio"] for row in oas_cov_stats]))
        oas_diag_mean_all.append(np.median([row["diag_mean"] for row in oas_cov_stats]))
        scm_diag_mean_all.append(np.median([row["diag_mean"] for row in scm_cov_stats]))
        oas_offdiag_all.append(np.median([row["offdiag_mean_abs"] for row in oas_cov_stats]))
        scm_offdiag_all.append(np.median([row["offdiag_mean_abs"] for row in scm_cov_stats]))
        ratio_eps_all.append(np.median(ratio_eps))
        rel_delta_all.append(np.median(rel_delta))
        eps_used_all.append(np.median(eps_used))
        eig_min_before_all.append(np.median([row["eig_min"] for row in oas_cov_stats]))
        eig_min_after_all.append(np.median([row["eig_min"] for row in oas_after_stats]))

        report["band_reports"][band_name] = {
            "input_std_uV_quantiles_median": {
                "p50": float(input_p50),
                "p95": float(input_p95),
            },
            "input_std_uV_quantiles_per_window": [w["input"]["std_uV"] for w in per_window],
            "input_var_v2_quantiles_per_window": [w["input"]["var_v2"] for w in per_window],
            "input_global_per_window": [w["input"]["global"] for w in per_window],
            "cov_oas_before_agg": _aggregate_stats(oas_cov_stats),
            "cov_scm_before_agg": _aggregate_stats(scm_cov_stats),
            "cov_oas_after_agg": _aggregate_stats(oas_after_stats),
            "cov_scm_after_agg": _aggregate_stats(scm_after_stats),
            "ratio_eps_domination": _quantiles(np.asarray(ratio_eps, dtype=np.float64)),
            "relative_delta": _quantiles(np.asarray(rel_delta, dtype=np.float64)),
            "eps_used": _quantiles(np.asarray(eps_used, dtype=np.float64)),
            "oas_vs_scm_agg": _aggregate_stats(oas_vs_scm),
            "per_window": per_window,
        }

        print(
            f"[summary] band={band_name} "
            + _format_line("X std_uV", input_p50, input_p95)
        )
        oas_agg = report["band_reports"][band_name]["cov_oas_before_agg"]
        oas_after_agg = report["band_reports"][band_name]["cov_oas_after_agg"]
        scm_agg = report["band_reports"][band_name]["cov_scm_before_agg"]
        print(
            f"[summary] band={band_name} cov_before_oas "
            f"diag_mean={oas_agg['diag_mean']['p50']:.6e} "
            f"offdiag_mean_abs={oas_agg['offdiag_mean_abs']['p50']:.6e} "
            f"diag_energy_ratio={oas_agg['diag_energy_ratio']['p50']:.6e} "
            f"eig_min={oas_agg['eig_min']['p50']:.6e}"
        )
        print(
            f"[summary] band={band_name} cov_before_scm "
            f"diag_mean={scm_agg['diag_mean']['p50']:.6e} "
            f"offdiag_mean_abs={scm_agg['offdiag_mean_abs']['p50']:.6e} "
            f"diag_energy_ratio={scm_agg['diag_energy_ratio']['p50']:.6e}"
        )
        ratio_med = report["band_reports"][band_name]["ratio_eps_domination"]["p50"]
        rel_med = report["band_reports"][band_name]["relative_delta"]["p50"]
        eig_min_after = oas_after_agg["eig_min"]["p50"]
        print(
            f"[summary] band={band_name} ratio_eps_domination_median={ratio_med:.6e} "
            f"relative_delta_median={rel_med:.6e} eig_min_after={eig_min_after:.6e}"
        )

    input_p50_med = float(np.median([v["p50"] for v in conclusion_inputs])) if conclusion_inputs else 0.0
    input_p95_med = float(np.median([v["p95"] for v in conclusion_inputs])) if conclusion_inputs else 0.0
    ratio_eps_med = float(np.median(ratio_eps_all)) if ratio_eps_all else 0.0
    rel_delta_med = float(np.median(rel_delta_all)) if rel_delta_all else 0.0
    eps_med = float(np.median(eps_used_all)) if eps_used_all else 0.0
    eig_min_before_med = float(np.median(eig_min_before_all)) if eig_min_before_all else 0.0
    eig_min_after_med = float(np.median(eig_min_after_all)) if eig_min_after_all else 0.0
    diag_mean_ratio_med = float(
        np.median([o / max(s, 1e-30) for o, s in zip(oas_diag_mean_all, scm_diag_mean_all)])
    ) if oas_diag_mean_all and scm_diag_mean_all else 0.0
    offdiag_ratio_med = float(
        np.median([o / max(s, 1e-30) for o, s in zip(oas_offdiag_all, scm_offdiag_all)])
    ) if oas_offdiag_all and scm_offdiag_all else 0.0
    diag_energy_med = float(np.median(oas_diag_energy_all)) if oas_diag_energy_all else 0.0
    diag_mean_med = float(np.median(oas_diag_mean_all)) if oas_diag_mean_all else 0.0

    eps_pinned = False
    if eps_med > 0:
        ratio_eig_eps = eig_min_after_med / max(eps_med, 1e-30)
        ratio_diag_eps = diag_mean_med / max(eps_med, 1e-30)
        eps_pinned = (0.8 <= ratio_eig_eps <= 1.2) and ratio_diag_eps < 5.0

    if input_p50_med < 1e-2 and input_p95_med < 1e-1:
        conclusion = "输入尺度问题（上游仍在压小，协方差没救）"
    elif args.spd_eps_mode != "absolute":
        ratio_target = float(args.spd_eps_alpha)
        ratio_ok = (ratio_target / 2.0) <= ratio_eps_med <= (ratio_target * 2.0)
        delta_ok = rel_delta_med < 0.05
        eig_ok = eig_min_after_med > 0 and not eps_pinned
        if ratio_ok and delta_ok and eig_ok:
            conclusion = "eps 不主导 + 差异可控（relative eps 验收通过）"
        elif not ratio_ok:
            conclusion = "eps 比例不匹配（ratio_eps_domination 未对齐 alpha）"
        elif not delta_ok:
            conclusion = "relative_delta 过大（eps 扰动过强）"
        elif not eig_ok:
            conclusion = "eig_min_after 被钉死或非正（SPD 修复异常）"
        else:
            conclusion = "relative eps 未完全通过，需要进一步检查"
    elif ratio_eps_med > 1e2:
        conclusion = "eps 绝对值过大（ratio_eps_domination >> 1）"
    elif diag_mean_ratio_med < 0.3 and offdiag_ratio_med < 0.3:
        conclusion = "OAS shrinkage 过强/口径不一致（需调整 shrinkage 或改用 SCM/相对 eps）"
    elif diag_energy_med > 0.9 and diag_mean_med < 1e-12:
        conclusion = "cov_shrink/口径问题（cov_before 近对角且尺度极小）"
    else:
        conclusion = "未见明显 eps 主导或极端 shrinkage，需进一步检查协方差后续步骤"

    report["conclusion"] = conclusion
    report["summary"] = {
        "input_p50_uV_median": input_p50_med,
        "input_p95_uV_median": input_p95_med,
        "ratio_eps_domination_median": ratio_eps_med,
        "relative_delta_median": rel_delta_med,
        "eps_used_median": eps_med,
        "eig_min_before_median": eig_min_before_med,
        "eig_min_after_median": eig_min_after_med,
        "oas_diag_energy_ratio_median": diag_energy_med,
        "oas_diag_mean_median": diag_mean_med,
        "oas_scm_diag_mean_ratio_median": diag_mean_ratio_med,
        "oas_scm_offdiag_ratio_median": offdiag_ratio_med,
    }

    print(f"[summary] ratio_eps_domination_median={ratio_eps_med:.6e}")
    print(f"[summary] relative_delta_median={rel_delta_med:.6e}")
    print(f"[summary] eig_min_before_median={eig_min_before_med:.6e} eig_min_after_median={eig_min_after_med:.6e}")
    print(f"[summary] conclusion: {conclusion}")
    print(f"[summary] json_report={out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    if hasattr(raw, "close"):
        raw.close()
    if hasattr(raw62, "close"):
        raw62.close()


if __name__ == "__main__":
    main()
