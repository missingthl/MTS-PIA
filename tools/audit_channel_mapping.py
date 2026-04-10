from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _print_missing(missing, limit: int = 20) -> None:
    if not missing:
        print("[audit] missing_list=[]")
        return
    if len(missing) <= limit:
        print(f"[audit] missing_list={missing}")
        return
    head = missing[:limit]
    print(f"[audit] missing_list(first {limit})={head} (total={len(missing)})")


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
        default="logs/audit_channel_mapping_1_1.json",
    )
    parser.add_argument("--locs", type=str, default="data/SEED/channel_62_pos.locs")
    parser.add_argument("--data-format", type=str, default=None)
    args = parser.parse_args()

    cnt_path = _resolve_path(args.cnt)
    out_path = _resolve_path(args.out)
    locs_path = _resolve_path(args.locs)

    raw = load_one_raw(str(cnt_path), backend="cnt", preload=False, data_format=args.data_format)
    raw62, meta = build_eeg62_view(
        raw,
        locs_path=str(locs_path),
        debug_map=True,
        debug_out_path=str(out_path),
        debug_sec=float(args.sec),
    )

    audit = meta.get("mapping_audit") or {}
    matched = int(audit.get("matched_count", 0))
    missing = int(audit.get("missing_count", 0))
    print(f"[audit] matched_count={matched} missing_count={missing}")
    _print_missing(audit.get("missing_list", []))

    near_zero = audit.get("near_zero", [])
    for entry in near_zero:
        threshold = entry.get("threshold")
        count = entry.get("count")
        print(f"[audit] near_zero threshold={threshold:.1e} count={count}")

    top10 = audit.get("top10_std_channels", [])
    if top10:
        print("[audit] top10_std_channels (expected_name, std_uV):")
        for row in top10:
            name = row.get("expected_name")
            std_uv = row.get("std_uV")
            print(f"[audit]  {name}: {std_uv:.3f} uV")
    else:
        print("[audit] top10_std_channels: []")

    print(f"[audit] json_report={out_path}")

    if hasattr(raw, "close"):
        raw.close()
    if hasattr(raw62, "close"):
        raw62.close()


if __name__ == "__main__":
    main()
