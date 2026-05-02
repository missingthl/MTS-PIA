#!/usr/bin/env python3
"""List the external baseline catalog in a human-readable table or CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.external_baseline_manifest import catalog_rows  # noqa: E402


FIELDS = [
    "name",
    "phase",
    "family",
    "source_space",
    "label_mode",
    "budget_matched",
    "implementation_status",
    "code_symbol",
    "code_file",
]


def _row_dict(spec) -> dict[str, str]:
    return {field: str(getattr(spec, field)) for field in FIELDS}


def _print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No baselines matched the requested filters.")
        return
    widths = {
        field: max(len(field), *(len(row[field]) for row in rows))
        for field in FIELDS
    }
    header = "  ".join(field.ljust(widths[field]) for field in FIELDS)
    print(header)
    print("  ".join("-" * widths[field] for field in FIELDS))
    for row in rows:
        print("  ".join(row[field].ljust(widths[field]) for field in FIELDS))


def main() -> None:
    parser = argparse.ArgumentParser(description="List ACT/CSTA external baseline arms.")
    parser.add_argument("--phase", default="", help="Filter by phase, e.g. phase1, phase2, phase3, csta_sampling.")
    parser.add_argument("--family", default="", help="Filter by family, e.g. raw_time, dtw_barycenter, csta_pia.")
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    args = parser.parse_args()

    specs = catalog_rows()
    if args.phase:
        specs = [spec for spec in specs if spec.phase == args.phase]
    if args.family:
        specs = [spec for spec in specs if spec.family == args.family]
    rows = [_row_dict(spec) for spec in specs]

    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
