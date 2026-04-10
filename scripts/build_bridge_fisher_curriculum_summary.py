#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd


DATASET_ORDER = [
    "har",
    "natops",
    "selfregulationscp1",
    "fingermovements",
    "basicmotions",
    "handmovementdirection",
    "uwavegesturelibrary",
    "epilepsy",
    "atrialfibrillation",
    "pendigits",
]


def _collect_csvs(root: Path, suffix: str) -> List[Path]:
    out: List[Path] = []
    for path in sorted(root.glob(f"*{suffix}")):
        if path.is_file():
            out.append(path)
    return out


def _dataset_sort_key(name: str) -> tuple[int, str]:
    key = str(name).strip().lower()
    if key in DATASET_ORDER:
        return (DATASET_ORDER.index(key), key)
    return (999, key)


def main() -> None:
    p = argparse.ArgumentParser(description="Build combined bridge+fisher summary tables.")
    p.add_argument("--out-root", type=str, required=True)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_files = [
        p for p in _collect_csvs(out_root, "_pilot_summary.csv") if "_per_seed" not in p.name
    ]
    target_files = [
        p for p in _collect_csvs(out_root, "_target_health_summary.csv")
    ]
    fidelity_files = [
        p for p in _collect_csvs(out_root, "_fidelity_summary.csv")
    ]

    summary_rows = []
    for path in summary_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["source_file"] = path.name
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "dataset" in summary_df.columns:
        summary_df = summary_df.sort_values(
            by="dataset",
            key=lambda s: s.map(lambda x: _dataset_sort_key(str(x))),
        ).reset_index(drop=True)
    summary_path = out_root / "bridge_fisher_curriculum_all_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    target_rows = []
    for path in target_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        target_rows.append(df)
    target_df = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
    if not target_df.empty and "dataset" in target_df.columns:
        target_df = target_df.sort_values(
            by=["dataset", "target_variant", "seed"],
            key=lambda s: s.map(lambda x: _dataset_sort_key(str(x))) if s.name == "dataset" else s,
        ).reset_index(drop=True)
    target_path = out_root / "bridge_fisher_curriculum_all_target_health.csv"
    target_df.to_csv(target_path, index=False)

    fidelity_rows = []
    for path in fidelity_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        fidelity_rows.append(df)
    fidelity_df = pd.concat(fidelity_rows, ignore_index=True) if fidelity_rows else pd.DataFrame()
    if not fidelity_df.empty and "dataset" in fidelity_df.columns:
        fidelity_df = fidelity_df.sort_values(
            by=["dataset", "target_variant", "seed"],
            key=lambda s: s.map(lambda x: _dataset_sort_key(str(x))) if s.name == "dataset" else s,
        ).reset_index(drop=True)
    fidelity_path = out_root / "bridge_fisher_curriculum_all_fidelity.csv"
    fidelity_df.to_csv(fidelity_path, index=False)

    md_lines = [
        "# Bridge + Fisher Curriculum All-Dataset Summary",
        "",
        f"- out_root: `{out_root}`",
        f"- summary_rows: `{len(summary_df)}`",
        f"- target_rows: `{len(target_df)}`",
        f"- fidelity_rows: `{len(fidelity_df)}`",
        "",
    ]
    if not summary_df.empty:
        md_lines.extend(
            [
                "## Dataset Readout",
                "",
            ]
        )
        for row in summary_df.itertuples(index=False):
            fisher_f1 = getattr(row, "bridge_fisher_curriculum_f1_mean_std", "n/a")
            delta = getattr(row, "delta_fisher_vs_bridge_multiround", "n/a")
            md_lines.append(
                f"- `{row.dataset}`: raw_only={row.raw_only_f1_mean_std}, "
                f"single={row.bridge_single_round_f1_mean_std}, "
                f"multiround={row.bridge_multiround_f1_mean_std}, "
                f"fisher={fisher_f1}, delta_fisher_vs_multiround={delta}, "
                f"label={row.result_label}"
            )
    conclusion_path = out_root / "bridge_fisher_curriculum_all_conclusion.md"
    conclusion_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
