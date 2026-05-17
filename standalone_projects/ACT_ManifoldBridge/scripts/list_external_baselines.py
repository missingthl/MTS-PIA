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
from utils.external_baseline_groups import group_for_method, known_groups  # noqa: E402
from utils.external_runner_registry import method_visibility  # noqa: E402


FIELDS = [
    "name",
    "paper_group",
    "phase",
    "family",
    "source_space",
    "label_mode",
    "budget_matched",
    "implementation_status",
    "visibility",
    "code_symbol",
    "code_file",
]


def _row_dict(spec) -> dict[str, str]:
    row = {field: str(getattr(spec, field)) for field in FIELDS if field not in {"paper_group", "visibility"}}
    row["paper_group"] = group_for_method(spec.name)
    row["visibility"] = method_visibility(spec.name)
    return row


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
    parser.add_argument(
        "--paper-group",
        default="",
        choices=["", *known_groups()],
        help="Filter by paper-facing group, e.g. temporal_vicinal_heuristic, deep_generative.",
    )
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived probes and optional methods hidden from the default workflow listing.",
    )
    args = parser.parse_args()

    specs = catalog_rows()
    if not args.include_archived:
        specs = [spec for spec in specs if method_visibility(spec.name) != "archived_probe"]
    if args.phase:
        specs = [spec for spec in specs if spec.phase == args.phase]
    if args.family:
        specs = [spec for spec in specs if spec.family == args.family]
    if args.paper_group:
        specs = [spec for spec in specs if group_for_method(spec.name) == args.paper_group]
    rows = [_row_dict(spec) for spec in specs]

    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
