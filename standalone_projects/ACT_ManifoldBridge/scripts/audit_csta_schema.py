#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.csta.schema_audit import audit_result_schema, audit_runner_passthrough_fields
from utils.external_runner_registry import CSTA_RESULT_PASSTHROUGH_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CSTA result schema and runner passthrough fields.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--fail-on-warning", action="store_true", help="Exit non-zero if unexpected schema issues exist.")
    args = parser.parse_args()

    report = {
        "result_schema": audit_result_schema(),
        "runner_passthrough": audit_runner_passthrough_fields(CSTA_RESULT_PASSTHROUGH_FIELDS),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        schema = report["result_schema"]
        passthrough = report["runner_passthrough"]
        print("CSTA schema audit")
        print(f"  field groups: {schema['field_group_count']}")
        print(f"  group field entries: {schema['total_group_field_entries']}")
        print(f"  unique group fields: {schema['unique_group_fields']}")
        print(f"  allowed cross-group duplicates: {schema['allowed_cross_group_duplicates']}")
        print(f"  unexpected cross-group duplicates: {schema['unexpected_cross_group_duplicates']}")
        print(f"  passthrough entries: {passthrough['passthrough_field_entries']}")
        print(f"  passthrough unique fields: {passthrough['passthrough_unique_fields']}")
        print(f"  passthrough duplicates: {passthrough['passthrough_duplicates']}")
        print(f"  missing generation fields in passthrough: {passthrough['missing_generation_fields']}")
        print(f"  ok: {bool(schema['ok'] and passthrough['ok'])}")

    if args.fail_on_warning and not (report["result_schema"]["ok"] and report["runner_passthrough"]["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
