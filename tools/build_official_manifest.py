import argparse
import json
import time
from pathlib import Path

from datasets.seed_official_mat_dataset import build_official_trial_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest for SEED official .mat features.")
    parser.add_argument(
        "--root",
        default="data/SEED/SEED_EEG/ExtractedFeatures_1s",
        help="root directory containing official .mat files",
    )
    parser.add_argument("--feature-base", default="de_LDS", help="feature base name (e.g., de_LDS)")
    parser.add_argument(
        "--label-path",
        default="data/SEED/SEED_EEG/SEED_stimulation.xlsx",
        help="path to SEED_stimulation.xlsx",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output manifest path (default logs/official_mat_manifest_<ts>.json)",
    )
    args = parser.parse_args()

    trials, meta = build_official_trial_index(
        root_dir=args.root,
        feature_base=args.feature_base,
        label_path=args.label_path,
    )
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path("logs") / f"official_mat_manifest_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root_dir": args.root,
        "feature_base": args.feature_base,
        "label_path": args.label_path,
        "created_at": ts,
        "trials": trials,
        "meta": meta,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(
        f"[official_manifest] root={args.root} trials={len(trials)} out={out_path}",
        flush=True,
    )
    if meta.get("missing_trials"):
        print(f"[official_manifest] missing_trials={len(meta['missing_trials'])}", flush=True)


if __name__ == "__main__":
    main()
