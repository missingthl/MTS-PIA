#!/usr/bin/env python3
"""Index ACT/CSTA experiment matrices from existing result CSVs.

The repository contains canonical roots, locked references, pilot matrices,
recovery shards, smoke outputs, and mechanism probes.  This script creates a
read-only governance index so maintainers can understand what exists without
opening every directory by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUT_ROOT = RESULTS_ROOT / "experiment_matrix_index_v1"


def _tier(path: Path) -> str:
    text = str(path.relative_to(RESULTS_ROOT)).lower()
    if "csta_external_baselines_phase1/resnet1d_s123/per_seed_external.csv" in text:
        return "locked_reference_phase1"
    if "csta_external_baselines_phase2/resnet1d_s123/per_seed_external.csv" in text:
        return "locked_reference_phase2"
    if text.startswith("csta_pia_final20/") or text.startswith("wdba_final20/"):
        return "canonical_final20_component"
    if text.startswith("final20_minimal_baseline_v1/"):
        return "canonical_final20_controls"
    if text.startswith("final20_main_comparison_v1/"):
        return "canonical_final20_report"
    if text.startswith("full_scale_resnet1d_v1/"):
        return "legacy_noncanonical_eta0.5"
    if "smoke" in text or "_smoke" in text:
        return "smoke_or_probe"
    if "_shards" in text:
        return "shard_intermediate"
    if "recovery" in text or "_rec" in text or "rebuilt" in text:
        return "recovery_or_rebuilt"
    if "backbone" in text or "moderntcn" in text or "minirocket" in text or "patchtst" in text or "timesnet" in text or "mptsnet" in text:
        return "backbone_robustness"
    if "pilot" in text or "step3" in text or "selector" in text or "direction" in text or "spg" in text or "latent" in text or "flow" in text or "ag_pia" in text:
        return "mechanism_or_pilot"
    if "external_baselines" in text or "full_scale_external" in text:
        return "external_baseline_matrix"
    return "unclassified"


def _role(path: Path, methods: list[str]) -> str:
    text = str(path.relative_to(RESULTS_ROOT)).lower()
    method_set = set(methods)
    if method_set == {"csta_topk_uniform_top5"}:
        return "u5_only"
    if "no_aug" in method_set and "csta_topk_uniform_top5" in method_set:
        return "u5_vs_no_aug"
    if "wdba_sameclass" in method_set and "csta_topk_uniform_top5" in method_set:
        return "csta_vs_external"
    if method_set and all(m.startswith("csta_") or m in {"random_cov_state", "pca_cov_state"} for m in method_set):
        return "csta_internal_or_controls"
    if "external_baselines" in text:
        return "external_baselines"
    if any(m in method_set for m in {"ag_pia_multihead5", "cs_flow_single_step", "latent_residual_flow", "task_guided_latent_residual_flow", "spg_pia_zhead"}):
        return "nextgen_mechanism_probe"
    return "mixed_or_other"


def _read_matrix(path: Path) -> dict[str, object]:
    rel = path.relative_to(RESULTS_ROOT)
    row = {
        "path": str(rel),
        "tier": _tier(path),
        "read_ok": False,
        "n_rows": 0,
        "n_success": 0,
        "n_failed": 0,
        "n_methods": 0,
        "methods": "",
        "n_datasets": 0,
        "datasets": "",
        "n_seeds": 0,
        "seeds": "",
        "backbones": "",
        "role": "unreadable",
        "mean_aug_f1_by_method": "",
        "warning": "",
    }
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        row["warning"] = str(exc)
        return row

    row["read_ok"] = True
    row["n_rows"] = int(len(df))
    if "status" in df:
        status = df["status"].fillna("success").astype(str)
        row["n_success"] = int(status.eq("success").sum())
        row["n_failed"] = int(status.ne("success").sum())
    methods = sorted(map(str, df["method"].dropna().unique())) if "method" in df else []
    datasets = sorted(map(str, df["dataset"].dropna().unique())) if "dataset" in df else []
    seeds = sorted(map(str, df["seed"].dropna().unique())) if "seed" in df else []
    backbones = sorted(map(str, df["backbone"].dropna().unique())) if "backbone" in df else []
    row["n_methods"] = len(methods)
    row["methods"] = ",".join(methods)
    row["n_datasets"] = len(datasets)
    row["datasets"] = ",".join(datasets)
    row["n_seeds"] = len(seeds)
    row["seeds"] = ",".join(seeds)
    row["backbones"] = ",".join(backbones)
    row["role"] = _role(path, methods)

    if "method" in df and "aug_f1" in df:
        means = df.groupby("method", dropna=False)["aug_f1"].mean().sort_values(ascending=False)
        row["mean_aug_f1_by_method"] = ";".join(f"{k}:{v:.6f}" for k, v in means.items())

    warnings = []
    if row["tier"].startswith("locked_reference") and row["n_failed"]:
        warnings.append("locked reference has failed rows")
    if row["tier"] == "legacy_noncanonical_eta0.5":
        warnings.append("non-canonical eta_safe=0.5 root")
    if "_csta_runs" in str(rel):
        warnings.append("nested csta run; usually not matrix-level evidence")
    if row["n_rows"] == 0:
        warnings.append("empty matrix")
    row["warning"] = "; ".join(warnings)
    return row


def _write_report(index: pd.DataFrame) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tier_counts = (
        index.groupby("tier", dropna=False)
        .agg(n_roots=("path", "count"), n_rows=("n_rows", "sum"), n_success=("n_success", "sum"), n_failed=("n_failed", "sum"))
        .reset_index()
        .sort_values("tier")
    )
    tier_counts.to_csv(OUT_ROOT / "experiment_matrix_tier_counts.csv", index=False)

    lines = [
        "# Experiment Matrix Index",
        "",
        "This index is generated from existing `per_seed_external.csv` files. It",
        "does not launch experiments and does not modify locked roots.",
        "",
        "## Tier Counts",
        "",
        "| tier | roots | rows | success | failed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in tier_counts.iterrows():
        lines.append(
            f"| {row['tier']} | {int(row['n_roots'])} | {int(row['n_rows'])} | "
            f"{int(row['n_success'])} | {int(row['n_failed'])} |"
        )

    important = index[
        index["tier"].isin(
            [
                "canonical_final20_component",
                "canonical_final20_controls",
                "locked_reference_phase1",
                "locked_reference_phase2",
                "backbone_robustness",
                "recovery_or_rebuilt",
                "legacy_noncanonical_eta0.5",
            ]
        )
    ].copy()
    # Keep the report readable: per-dataset backbone files remain available in
    # experiment_matrix_index.csv, while the markdown report focuses on
    # matrix-level or multi-dataset roots.
    important = important[
        (important["n_datasets"] > 1)
        | (~important["tier"].isin(["backbone_robustness", "recovery_or_rebuilt"]))
    ].copy()
    lines.extend(["", "## Important Roots", "", "| tier | role | rows | datasets | methods | path | warning |", "| --- | --- | ---: | ---: | --- | --- | --- |"])
    for _, row in important.sort_values(["tier", "path"]).iterrows():
        methods = row["methods"]
        if len(methods) > 90:
            methods = methods[:87] + "..."
        lines.append(
            f"| {row['tier']} | {row['role']} | {int(row['n_rows'])} | {int(row['n_datasets'])} | "
            f"`{methods}` | `{row['path']}` | {row['warning']} |"
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Use `results/CANONICAL_RESULTS.md` for paper-facing primary results.",
            "- Use `docs/BACKBONE_U5_MATRIX.md` for backbone robustness evidence.",
            "- Treat `legacy_noncanonical_eta0.5` roots as drift-audit references only.",
            "- Treat smoke/probe roots as local validation, not paper evidence.",
            "- Do not write smoke/probe runs into locked Phase1/Phase2 roots.",
        ]
    )
    (OUT_ROOT / "experiment_matrix_index_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    files = sorted(RESULTS_ROOT.rglob("per_seed_external.csv"))
    # Exclude nested CSTA child runs and shards from the default matrix index;
    # they are referenced by parent rows and greatly inflate the listing.
    files = [
        path
        for path in files
        if "_csta_runs" not in path.parts and "_shards" not in path.parts
    ]
    index = pd.DataFrame([_read_matrix(path) for path in files])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    index.to_csv(OUT_ROOT / "experiment_matrix_index.csv", index=False)
    _write_report(index)
    print(f"Wrote {OUT_ROOT.relative_to(PROJECT_ROOT)} with {len(index)} matrix roots")


if __name__ == "__main__":
    main()
