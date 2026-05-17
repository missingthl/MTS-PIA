#!/usr/bin/env python3
"""Read-only workflow readiness checks for ACT/CSTA.

This script is intentionally lightweight and stdlib-only.  It does not launch
experiments and it does not modify locked result roots.  Its job is to make the
current workflow state visible before formal matrix work:

* canonical entrypoints exist;
* locked Phase1/Phase2 references still have expected row counts;
* canonical U5/wDBA Final20 roots are present;
* E1 artifacts exist and expose explicit full/subset/missing coverage.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ACT_ROOT.parents[1]
OUT_ROOT = ACT_ROOT / "results" / "workflow_readiness_v1"


@dataclass(frozen=True)
class Check:
    area: str
    name: str
    status: str
    detail: str
    path: str = ""


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "OK"
    return "WARN" if warn else "FAIL"


def _check_exists(path: Path, *, area: str, name: str, required: bool = True) -> Check:
    exists = path.exists()
    return Check(
        area=area,
        name=name,
        status=_status(exists, warn=not required),
        detail="exists" if exists else "missing",
        path=_rel(path),
    )


def _check_csv_rows(
    path: Path,
    *,
    area: str,
    name: str,
    expected: int | None = None,
    minimum: int | None = None,
    method: str | None = None,
    required: bool = True,
) -> Check:
    rows = _read_rows(path)
    if not path.exists():
        return Check(area, name, _status(False, warn=not required), "missing", _rel(path))

    selected = [r for r in rows if method is None or r.get("method") == method]
    count = len(selected)
    ok = True
    expectations: list[str] = []
    if expected is not None:
        ok = ok and count == expected
        expectations.append(f"expected={expected}")
    if minimum is not None:
        ok = ok and count >= minimum
        expectations.append(f"minimum={minimum}")
    if method is not None:
        expectations.append(f"method={method}")
    suffix = ", ".join(expectations) if expectations else "observed"
    return Check(area, name, _status(ok, warn=not required), f"rows={count}; {suffix}", _rel(path))


def _dataset_coverage(rows: Iterable[dict[str, str]], method: str) -> tuple[int, int, int]:
    method_rows = [r for r in rows if r.get("method") == method]
    datasets = {r.get("dataset", "") for r in method_rows if r.get("dataset", "")}
    seeds = {(r.get("dataset", ""), r.get("seed", "")) for r in method_rows}
    successes = sum(1 for r in method_rows if (r.get("status", "success") or "success") == "success")
    return len(datasets), len(seeds), successes


def _e1_checks() -> list[Check]:
    checks: list[Check] = []
    e1_root = ACT_ROOT / "results" / "e1_main"
    per_seed = e1_root / "per_seed_e1_runs.csv"
    method_registry = e1_root / "e1_method_registry.csv"
    main_table = e1_root / "e1_main_table.csv"
    audit_doc = ACT_ROOT / "docs" / "E1_DATA_AUDIT.md"

    checks.extend(
        [
            _check_exists(per_seed, area="E1", name="per_seed_e1_runs"),
            _check_exists(method_registry, area="E1", name="method_registry"),
            _check_exists(main_table, area="E1", name="main_table"),
            _check_exists(audit_doc, area="E1", name="audit_doc"),
        ]
    )

    rows = _read_rows(per_seed)
    registry_rows = _read_rows(method_registry)
    checks.append(
        Check(
            "E1",
            "atomic_run_count",
            _status(len(rows) >= 1),
            f"rows={len(rows)}",
            _rel(per_seed),
        )
    )
    checks.append(
        Check(
            "E1",
            "method_registry_count",
            _status(len(registry_rows) == 11, warn=True),
            f"methods={len(registry_rows)}; expected=11",
            _rel(method_registry),
        )
    )

    full_expected = {
        "no_aug",
        "raw_aug_jitter",
        "raw_mixup",
        "dba_sameclass",
        "wdba_sameclass",
        "csta_topk_uniform_top5",
    }
    subset_expected = {
        "raw_aug_timewarp",
        "diffusionts_classwise",
        "rgw_sameclass",
        "dgw_sameclass",
    }
    missing_expected = {"timegan_classwise"}

    for method in sorted(full_expected):
        n_datasets, n_pairs, successes = _dataset_coverage(rows, method)
        ok = n_datasets == 20 and n_pairs == 60
        checks.append(
            Check(
                "E1 coverage",
                method,
                _status(ok),
                f"datasets={n_datasets}/20; dataset_seed_pairs={n_pairs}/60; successes={successes}",
                _rel(per_seed),
            )
        )

    for method in sorted(subset_expected):
        n_datasets, n_pairs, successes = _dataset_coverage(rows, method)
        checks.append(
            Check(
                "E1 coverage",
                method,
                "WARN",
                f"subset coverage; datasets={n_datasets}/20; dataset_seed_pairs={n_pairs}/60; successes={successes}",
                _rel(per_seed),
            )
        )

    for method in sorted(missing_expected):
        n_datasets, n_pairs, successes = _dataset_coverage(rows, method)
        checks.append(
            Check(
                "E1 coverage",
                method,
                "WARN",
                f"registered but missing E1 runs; datasets={n_datasets}/20; dataset_seed_pairs={n_pairs}/60; successes={successes}",
                _rel(per_seed),
            )
        )

    return checks


def build_checks() -> list[Check]:
    checks: list[Check] = []

    required_files = [
        ("entrypoint", "run_act_pilot", ACT_ROOT / "run_act_pilot.py"),
        ("entrypoint", "external_runner", ACT_ROOT / "scripts" / "run_external_baselines_phase1.py"),
        ("entrypoint", "e1_builder", ACT_ROOT / "scripts" / "build_e1_main_artifacts.py"),
        ("workflow", "workflow_doc", ACT_ROOT / "docs" / "WORKFLOW.md"),
        ("workflow", "e1_plan", ACT_ROOT / "docs" / "E1_MAIN_TABLE_PLAN.md"),
        ("workflow", "scripts_readme", ACT_ROOT / "scripts" / "README.md"),
    ]
    checks.extend(_check_exists(path, area=area, name=name) for area, name, path in required_files)

    checks.extend(
        [
            _check_csv_rows(
                ACT_ROOT / "results" / "csta_external_baselines_phase1" / "resnet1d_s123" / "per_seed_external.csv",
                area="locked roots",
                name="phase1_row_count",
                expected=231,
            ),
            _check_csv_rows(
                ACT_ROOT / "results" / "csta_external_baselines_phase2" / "resnet1d_s123" / "per_seed_external.csv",
                area="locked roots",
                name="phase2_row_count",
                expected=105,
            ),
            _check_csv_rows(
                ACT_ROOT / "results" / "csta_pia_final20" / "resnet1d_s123" / "per_seed_external.csv",
                area="canonical",
                name="u5_final20",
                expected=60,
                method="csta_topk_uniform_top5",
            ),
            _check_csv_rows(
                ACT_ROOT / "results" / "wdba_final20" / "resnet1d_s123" / "per_seed_external.csv",
                area="canonical",
                name="wdba_final20",
                expected=60,
                method="wdba_sameclass",
            ),
        ]
    )

    checks.extend(_e1_checks())
    return checks


def write_outputs(checks: list[Check]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "workflow_readiness_checks.csv"
    report_path = OUT_ROOT / "workflow_readiness_report.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["area", "name", "status", "detail", "path"])
        writer.writeheader()
        for c in checks:
            writer.writerow(c.__dict__)

    counts = {status: sum(1 for c in checks if c.status == status) for status in ("OK", "WARN", "FAIL")}
    lines = [
        "# Workflow Readiness Report",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Summary",
        "",
        f"- OK: `{counts['OK']}`",
        f"- WARN: `{counts['WARN']}`",
        f"- FAIL: `{counts['FAIL']}`",
        "",
        "Warnings are expected for E1 methods that are intentionally registered but not fully covered yet, such as TimeGAN.",
        "",
        "## Checks",
        "",
        "| Area | Check | Status | Detail | Path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| {c.area} | {c.name} | {c.status} | {c.detail} | `{c.path}` |")
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "- If any locked-root check fails, stop and audit result provenance before running more experiments.",
            "- If only E1 subset warnings appear, continue filling the registered E1 gaps or keep them marked as subset in tables.",
            "- Re-run `scripts/build_e1_main_artifacts.py` after new E1 runs, then re-run this readiness check.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    checks = build_checks()
    write_outputs(checks)

    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    print(f"workflow readiness: {len(checks) - warn_count - fail_count} OK, {warn_count} WARN, {fail_count} FAIL")
    print(f"report: {_rel(OUT_ROOT / 'workflow_readiness_report.md')}")
    print(f"checks: {_rel(OUT_ROOT / 'workflow_readiness_checks.csv')}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
