import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scipy.io import whosmat
except Exception:
    whosmat = None

from datasets.seed_raw_trials import load_seed_stimulation_labels


IGNORE_KEYS = {"__header__", "__version__", "__globals__"}


def _load_preflight(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _scan_mat(path: Path):
    if whosmat is None:
        raise RuntimeError("scipy.io.whosmat not available; install scipy")
    rows = []
    for name, shape, dtype in whosmat(path):
        if name in IGNORE_KEYS:
            continue
        rows.append({"name": name, "shape": list(shape), "dtype": str(dtype)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Audit label alignment for SEED official .mat features.")
    parser.add_argument("--root", type=str, default=None, help="feature root dir")
    parser.add_argument(
        "--preflight",
        type=str,
        default="logs/baseline_preflight.json",
        help="preflight JSON path",
    )
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument(
        "--out",
        type=str,
        default="logs/official_label_alignment.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--time-txt",
        type=str,
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt",
    )
    parser.add_argument(
        "--stim-xlsx",
        type=str,
        default="data/SEED/SEED_EEG/SEED_stimulation.xlsx",
    )
    args = parser.parse_args()

    preflight = _load_preflight(Path(args.preflight))
    root = Path(args.root) if args.root else None
    if root is None and preflight:
        root = Path(preflight.get("root"))
    if root is None or not root.is_dir():
        print("[align] missing feature root; provide --root or valid preflight", flush=True)
        sys.exit(1)

    recommended = None
    if preflight:
        recommended = preflight.get("recommended_key_base")

    mats = sorted(list(root.glob("*.mat")))
    if not mats:
        mats = sorted(list(root.rglob("*.mat")))
    if not mats:
        print(f"[align] no .mat files found in {root}", flush=True)
        sys.exit(1)

    max_files = min(int(args.max_files), len(mats))
    mats = mats[:max_files]

    labels = load_seed_stimulation_labels(args.stim_xlsx)
    label_len = len(labels)

    ok = 0
    ambiguous = 0
    fail = 0
    reasons = Counter()
    details = []

    for p in mats:
        rows = _scan_mat(p)
        names = [row["name"] for row in rows]
        status = "ambiguous"
        reason = None
        trial_dim = None
        key_base = recommended

        if key_base:
            pattern = re.compile(rf"^{re.escape(key_base)}\d+$")
            matched = [n for n in names if pattern.match(n)]
            if matched:
                if len(matched) == label_len:
                    status = "ok"
                else:
                    status = "ambiguous"
                    reason = f"trial_key_count={len(matched)}"
            else:
                reason = "key_base_not_found"
        else:
            reason = "no_recommended_key_base"

        if status != "ok":
            # attempt to infer trial dimension inside arrays
            for row in rows:
                shape = row["shape"]
                if label_len in shape:
                    trial_dim = shape.index(label_len)
                    status = "ambiguous"
                    reason = reason or "trial_axis_in_array"
                    break

        if status == "ok":
            ok += 1
        elif status == "ambiguous":
            ambiguous += 1
            reasons[reason or "ambiguous"] += 1
        else:
            fail += 1
            reasons[reason or "fail"] += 1

        details.append(
            {
                "file": str(p),
                "status": status,
                "reason": reason,
                "trial_dim": trial_dim,
                "key_base": key_base,
            }
        )

    report = {
        "root": str(root),
        "checked_files": [str(p) for p in mats],
        "label_len": int(label_len),
        "recommended_key_base": recommended,
        "ok_count": ok,
        "ambiguous_count": ambiguous,
        "fail_count": fail,
        "failure_reasons": dict(reasons.most_common(3)),
        "details": details,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(
        f"[align] ok={ok} ambiguous={ambiguous} fail={fail} reasons={dict(reasons.most_common(3))}",
        flush=True,
    )
    print(f"[align] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
