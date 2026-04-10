import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from datasets.seed_raw_cnt import build_eeg62_view, load_one_cnt
from datasets.seed_raw_trials import build_trial_index
from tools.alignment_core import (
    AlignmentError,
    apply_smooth,
    best_lag_corr,
    compute_raw_de_proxy_series,
    load_official_de_series,
    pearson_r,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LDS/movingAve alignment audit.")
    parser.add_argument("--cnt", required=True, help="path to CNT file")
    parser.add_argument("--mat-root", required=True, help="ExtractedFeatures_1s or _4s")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument("--band", type=str, default="delta,gamma")
    parser.add_argument("--offset-min", type=float, default=-3.0)
    parser.add_argument("--offset-max", type=float, default=3.0)
    parser.add_argument("--offset-step", type=float, default=0.5)
    parser.add_argument("--lag-max", type=int, default=3)
    parser.add_argument("--eps-var", type=float, default=1e-12)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    raw = load_one_cnt(args.cnt, preload=False)
    sfreq = float(raw.info["sfreq"])
    subject = int(Path(args.cnt).stem.split("_")[0])
    session = int(Path(args.cnt).stem.split("_")[1])

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

    bands = [b.strip().lower() for b in args.band.split(",") if b.strip()]
    band_ranges = {"delta": (1.0, 4.0), "gamma": (31.0, 50.0)}
    for name in bands:
        if name not in band_ranges:
            raise ValueError(f"Unsupported band: {name}")

    official_moving = {}
    official_lds = {}
    for band in bands:
        official_moving[band] = load_official_de_series(
            mat_root=args.mat_root,
            subject=subject,
            session=session,
            trial=trial_input,
            band=band,
            target_key="de_movingAve",
        )
        official_lds[band] = load_official_de_series(
            mat_root=args.mat_root,
            subject=subject,
            session=session,
            trial=trial_input,
            band=band,
            target_key="de_LDS",
        )

    T_off = int(official_lds[bands[0]].shape[0])
    duration_sec = float(T_off * args.hop_sec)
    offsets = np.arange(args.offset_min, args.offset_max + 1e-9, args.offset_step)

    raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
    raw_curves_by_offset: Dict[float, Dict[str, np.ndarray]] = {}
    skip_offsets = []

    for offset_sec in offsets:
        curves = {}
        for band in bands:
            try:
                curve = compute_raw_de_proxy_series(
                    cnt_path=args.cnt,
                    trial_input=trial_input,
                    band=band,
                    window_sec=args.window_sec,
                    hop_sec=args.hop_sec,
                    offset_sec=float(offset_sec),
                    smooth_mode="none",
                    smooth_param=None,
                    time_unit="samples@1000",
                    duration_sec=duration_sec,
                    eps_var=float(args.eps_var),
                    raw62=raw62,
                    sfreq=sfreq,
                )
            except AlignmentError:
                skip_offsets.append(float(offset_sec))
                curves = None
                break
            curves[band] = curve
        if curves is not None:
            raw_curves_by_offset[float(offset_sec)] = curves

    smooth_configs = (
        [{"mode": "none", "param": None}]
        + [{"mode": "ma", "param": k} for k in (3, 5, 7, 9)]
        + [{"mode": "ema", "param": a} for a in (0.05, 0.1, 0.2, 0.3)]
        + [{"mode": "kalman", "param": q} for q in (1e-4, 1e-3, 1e-2, 1e-1)]
    )

    results = []
    skipped_configs = []

    for cfg in smooth_configs:
        cfg_id = f"{cfg['mode']}:{cfg['param']}"
        cfg_res = {"config_id": cfg_id, "mode": cfg["mode"], "param": cfg["param"]}
        cfg_res["targets"] = {"de_movingAve": {}, "de_LDS": {}}
        skip_cfg = False

        for target_name, official in [("de_movingAve", official_moving), ("de_LDS", official_lds)]:
            for band in bands:
                best = {"best_r": float("-inf")}
                official_curve = official[band]
                for offset_sec, curves in raw_curves_by_offset.items():
                    raw_curve = curves[band]
                    smooth_curve = apply_smooth(raw_curve, cfg["mode"], cfg["param"])
                    if not np.isfinite(smooth_curve).all():
                        skip_cfg = True
                        break
                    r0 = pearson_r(official_curve, smooth_curve)
                    best_r, best_lag = best_lag_corr(
                        official_curve, smooth_curve, int(args.lag_max)
                    )
                    if best_r > best["best_r"]:
                        best = {
                            "best_r": float(best_r),
                            "best_lag": int(best_lag),
                            "best_offset": float(offset_sec),
                            "r0": float(r0),
                        }
                if skip_cfg:
                    break
                cfg_res["targets"][target_name][band] = best
            if skip_cfg:
                break
        if skip_cfg:
            skipped_configs.append({"config_id": cfg_id, "reason": "non_finite"})
            continue
        results.append(cfg_res)

    def _rank_by(target: str, band: str) -> List[dict]:
        return sorted(
            results,
            key=lambda x: x["targets"][target][band]["best_r"],
            reverse=True,
        )

    best_moving = _rank_by("de_movingAve", "gamma")[0] if results else None
    best_lds = _rank_by("de_LDS", "gamma")[0] if results else None
    top3_lds = [
        (
            r["config_id"],
            r["targets"]["de_LDS"]["gamma"]["best_r"],
            r["targets"]["de_LDS"]["gamma"]["best_offset"],
            r["targets"]["de_LDS"]["gamma"]["best_lag"],
        )
        for r in _rank_by("de_LDS", "gamma")[:3]
    ]

    payload = {
        "cnt": args.cnt,
        "sfreq": sfreq,
        "subject": subject,
        "session": session,
        "trial_input": trial_input,
        "trial_zero": trial_zero,
        "mat_root": args.mat_root,
        "T_off": T_off,
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "duration_sec": duration_sec,
        "bands": bands,
        "offset_scan": {
            "min": float(args.offset_min),
            "max": float(args.offset_max),
            "step": float(args.offset_step),
            "lag_max": int(args.lag_max),
        },
        "configs_total": len(smooth_configs),
        "configs_valid": len(results),
        "configs_skipped": skipped_configs,
        "skip_offsets": skip_offsets,
        "best_movingAve_gamma": best_moving,
        "best_LDS_gamma": best_lds,
        "top3_LDS_gamma": top3_lds,
        "results": results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# LDS Alignment Audit",
        "",
        f"- mat_root: {args.mat_root}",
        f"- trial: {trial_input}",
        f"- bands: {bands}",
        f"- window_sec: {args.window_sec} hop_sec: {args.hop_sec}",
        f"- configs_total: {len(smooth_configs)} valid: {len(results)}",
        "",
        "## Best gamma",
        f"- target=de_movingAve best={best_moving['config_id'] if best_moving else 'n/a'} "
        f"best_r={best_moving['targets']['de_movingAve']['gamma']['best_r'] if best_moving else 'n/a'}",
        f"- target=de_LDS best={best_lds['config_id'] if best_lds else 'n/a'} "
        f"best_r={best_lds['targets']['de_LDS']['gamma']['best_r'] if best_lds else 'n/a'}",
        "",
        "## Top3 de_LDS gamma configs",
    ]
    for item in top3_lds:
        md_lines.append(f"- {item}")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")

    if best_moving:
        print(
            "[align] target=de_movingAve "
            f"best_r_gamma={best_moving['targets']['de_movingAve']['gamma']['best_r']:.4f} "
            f"best_offset={best_moving['targets']['de_movingAve']['gamma']['best_offset']} "
            f"smooth={best_moving['config_id']}",
            flush=True,
        )
    if best_lds:
        print(
            "[align] target=de_LDS "
            f"best_r_gamma={best_lds['targets']['de_LDS']['gamma']['best_r']:.4f} "
            f"best_offset={best_lds['targets']['de_LDS']['gamma']['best_offset']} "
            f"smooth={best_lds['config_id']}",
            flush=True,
        )
    if top3_lds:
        print(f"[align] top3_de_LDS={top3_lds}", flush=True)


if __name__ == "__main__":
    main()
