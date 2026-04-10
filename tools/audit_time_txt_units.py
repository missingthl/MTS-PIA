import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets.seed_raw_trials import load_seed_time_points


def _score_durations(durations: List[float], min_sec: float, max_sec: float) -> tuple:
    mid = 0.5 * (min_sec + max_sec)
    in_range = sum(min_sec <= d <= max_sec for d in durations)
    penalty = sum(abs(d - mid) / mid for d in durations)
    return (in_range, -penalty)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit time.txt units for SEED trial slicing.")
    parser.add_argument(
        "--time-txt",
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt",
        help="path to time.txt",
    )
    parser.add_argument("--n-trials", type=int, default=3, help="number of trials to inspect")
    parser.add_argument("--min-sec", type=float, default=30.0, help="min reasonable duration (sec)")
    parser.add_argument("--max-sec", type=float, default=600.0, help="max reasonable duration (sec)")
    parser.add_argument(
        "--out",
        default="logs/audit_time_txt_units.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    time_txt = Path(args.time_txt)
    start_pts, end_pts = load_seed_time_points(str(time_txt))
    n = max(1, min(int(args.n_trials), len(start_pts)))

    trials = []
    for i in range(n):
        s = int(start_pts[i])
        e = int(end_pts[i])
        d = e - s
        trials.append({"trial": i + 1, "start": s, "end": e, "delta": d})

    durations = {
        "samples@1000": [t["delta"] / 1000.0 for t in trials],
        "samples@200": [t["delta"] / 200.0 for t in trials],
        "seconds": [float(t["delta"]) for t in trials],
    }

    scores = {
        unit: _score_durations(vals, args.min_sec, args.max_sec)
        for unit, vals in durations.items()
    }
    recommended = max(scores.items(), key=lambda x: x[1])[0]

    print(f"[time_txt] path={time_txt} n_trials={n}", flush=True)
    for t in trials:
        idx = t["trial"]
        print(
            f"[trial{idx}] start={t['start']} end={t['end']} "
            f"sec@1000={durations['samples@1000'][idx-1]:.2f} "
            f"sec@200={durations['samples@200'][idx-1]:.2f} "
            f"sec@seconds={durations['seconds'][idx-1]:.2f}",
            flush=True,
        )
    print(
        "[time_txt] recommended_unit="
        f"{recommended} (range={args.min_sec}-{args.max_sec}s)",
        flush=True,
    )

    payload: Dict[str, object] = {
        "time_txt": str(time_txt),
        "n_trials": n,
        "min_sec": float(args.min_sec),
        "max_sec": float(args.max_sec),
        "trials": trials,
        "durations": durations,
        "scores": {k: {"in_range": v[0], "neg_penalty": v[1]} for k, v in scores.items()},
        "recommended_unit": recommended,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[time_txt] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
