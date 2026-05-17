#!/usr/bin/env python3
"""Build an auditable CSTA-U5 backbone robustness matrix.

This script is intentionally conservative: it does not launch experiments and
does not rewrite any locked reference root.  It only reads existing
``per_seed_external.csv`` files and writes a normalized governance artifact.

The goal is to make backbone robustness evidence discoverable without requiring
future maintainers or agents to reverse-engineer scattered result roots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUT_ROOT = RESULTS_ROOT / "backbone_u5_matrix_v1"
CSTA_METHOD = "csta_topk_uniform_top5"


@dataclass(frozen=True)
class EvidenceSpec:
    backbone: str
    scope: str
    evidence_tier: str
    source_policy: str
    note: str
    files: tuple[Path, ...]


def _existing(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(path for path in paths if path.exists())


def _first_existing(paths: Iterable[Path]) -> tuple[Path, ...]:
    for path in paths:
        if path.exists():
            return (path,)
    return ()


def _glob(pattern: str) -> tuple[Path, ...]:
    return tuple(sorted(RESULTS_ROOT.glob(pattern)))


def _specs() -> list[EvidenceSpec]:
    return [
        EvidenceSpec(
            backbone="resnet1d",
            scope="final20",
            evidence_tier="canonical",
            source_policy="csta_pia_final20_csta_rows_base_f1_as_no_aug",
            note=(
                "Canonical ResNet1D U5 row.  no_aug is read from the CSTA "
                "row's base_f1, matching final20_main_comparison_v1."
            ),
            files=_existing(
                [
                    RESULTS_ROOT
                    / "csta_pia_final20"
                    / "resnet1d_s123"
                    / "per_seed_external.csv"
                ]
            ),
        ),
        EvidenceSpec(
            backbone="resnet1d",
            scope="final20",
            evidence_tier="legacy_noncanonical",
            source_policy="full_scale_resnet1d_v1_eta_safe_0.5",
            note=(
                "Historical full-scale root.  Kept for drift auditing only; "
                "CANONICAL_RESULTS.md marks this root as non-canonical."
            ),
            files=_existing([RESULTS_ROOT / "full_scale_resnet1d_v1" / "per_seed_external.csv"]),
        ),
        EvidenceSpec(
            backbone="moderntcn",
            scope="final20",
            evidence_tier="rebuilt_final20",
            source_policy="rebuilt_per_seed_preferred",
            note="ModernTCN Final20 rebuilt robustness root.",
            files=_first_existing(
                [
                    RESULTS_ROOT
                    / "moderntcn_final20_robustness_v1"
                    / "per_seed_external_REBUILT.csv",
                    RESULTS_ROOT
                    / "moderntcn_final20_robustness_v1"
                    / "per_seed_external.csv",
                ]
            ),
        ),
        EvidenceSpec(
            backbone="minirocket",
            scope="final20",
            evidence_tier="best_available_recovery",
            source_policy="all_minirocket_final20_roots_dedupe_best_aug_f1",
            note=(
                "MiniRocket evidence is assembled from core/batch/recovery roots. "
                "This is useful robustness evidence but should be cited with the "
                "recovery-source caveat."
            ),
            files=_glob("minirocket_final20_*/per_seed_external*.csv"),
        ),
        EvidenceSpec(
            backbone="patchtst",
            scope="final20",
            evidence_tier="best_available_recovery",
            source_policy="recursive_dataset_roots_dedupe_best_aug_f1",
            note=(
                "PatchTST Final20 is stored per dataset plus recovery folders; "
                "deduplication is required for a single matrix."
            ),
            files=_glob("patchtst_final20_v1/**/per_seed_external*.csv"),
        ),
        EvidenceSpec(
            backbone="timesnet",
            scope="final20",
            evidence_tier="best_available_recovery",
            source_policy="recursive_dataset_roots_dedupe_best_aug_f1",
            note=(
                "TimesNet Final20 is stored per dataset plus recovery folders; "
                "deduplication is required for a single matrix."
            ),
            files=_glob("timesnet_final20_v1/**/per_seed_external*.csv"),
        ),
        EvidenceSpec(
            backbone="moderntcn",
            scope="pilot7",
            evidence_tier="pilot_only_u5",
            source_policy="single_u5_root_base_f1_as_no_aug",
            note="Pilot7 U5-only backbone probe; useful but not Final20 evidence.",
            files=_existing(
                [
                    RESULTS_ROOT
                    / "backbone_robustness_moderntcn_v1"
                    / "moderntcn_s123"
                    / "per_seed_external.csv"
                ]
            ),
        ),
        EvidenceSpec(
            backbone="mptsnet",
            scope="pilot7",
            evidence_tier="pilot_only_u5",
            source_policy="single_u5_root_base_f1_as_no_aug",
            note=(
                "MPTSNet currently appears as a Pilot7 U5-only probe.  It should "
                "not be promoted to Final20 robustness without paired evidence."
            ),
            files=_existing(
                [
                    RESULTS_ROOT
                    / "backbone_robustness_mptsnet_v1"
                    / "mptsnet_s123"
                    / "per_seed_external.csv"
                ]
            ),
        ),
    ]


def _read_files(files: tuple[Path, ...]) -> pd.DataFrame:
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - governance script
            frames.append(pd.DataFrame({"_source_file": [str(path)], "_read_error": [str(exc)]}))
            continue
        df = df.copy()
        df["_source_file"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _pair_table(df: pd.DataFrame, spec: EvidenceSpec) -> pd.DataFrame:
    if df.empty or "method" not in df or "dataset" not in df or "seed" not in df:
        return pd.DataFrame()

    df = df.copy()
    if "status" in df:
        df = df[df["status"].fillna("success").astype(str).eq("success")]
    if "aug_f1" not in df:
        return pd.DataFrame()

    # Keep the strongest available duplicate only for explicitly recovery-style
    # sources.  The policy is recorded in the output so this does not masquerade
    # as a pristine single-run root.
    if "dedupe_best_aug_f1" in spec.source_policy:
        df = df.sort_values("aug_f1", ascending=False)
        df = df.drop_duplicates(["dataset", "seed", "method"], keep="first")

    csta = df[df["method"].astype(str).eq(CSTA_METHOD)].copy()
    if csta.empty:
        return pd.DataFrame()

    csta = csta.rename(columns={"aug_f1": "csta_u5_f1"})
    keep = ["dataset", "seed", "csta_u5_f1", "_source_file"]
    if "base_f1" in csta:
        keep.append("base_f1")
    csta = csta[keep].copy()
    csta = csta.rename(columns={"_source_file": "csta_source_file"})

    no_aug = df[df["method"].astype(str).eq("no_aug")].copy()
    if not no_aug.empty:
        no_aug = no_aug.sort_values("aug_f1", ascending=False)
        no_aug = no_aug.drop_duplicates(["dataset", "seed"], keep="first")
        no_aug = no_aug[["dataset", "seed", "aug_f1", "_source_file"]].rename(
            columns={"aug_f1": "no_aug_f1", "_source_file": "no_aug_source_file"}
        )
        paired = csta.merge(no_aug, on=["dataset", "seed"], how="left")
    else:
        paired = csta.copy()
        paired["no_aug_f1"] = np.nan
        paired["no_aug_source_file"] = ""

    if "base_f1" in paired.columns:
        paired["no_aug_f1"] = paired["no_aug_f1"].fillna(paired["base_f1"])
        paired.loc[paired["no_aug_source_file"].eq(""), "no_aug_source_file"] = "base_f1_from_csta_row"

    paired = paired.dropna(subset=["csta_u5_f1", "no_aug_f1"])
    if paired.empty:
        return paired

    paired["delta_vs_no_aug"] = paired["csta_u5_f1"] - paired["no_aug_f1"]
    paired["backbone"] = spec.backbone
    paired["scope"] = spec.scope
    paired["evidence_tier"] = spec.evidence_tier
    paired["source_policy"] = spec.source_policy
    paired["note"] = spec.note
    return paired[
        [
            "backbone",
            "scope",
            "evidence_tier",
            "dataset",
            "seed",
            "csta_u5_f1",
            "no_aug_f1",
            "delta_vs_no_aug",
            "source_policy",
            "csta_source_file",
            "no_aug_source_file",
            "note",
        ]
    ]


def _summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (backbone, scope, tier, policy), g in paired.groupby(
        ["backbone", "scope", "evidence_tier", "source_policy"], dropna=False
    ):
        delta = g["delta_vs_no_aug"]
        win = int((delta > 0.001).sum())
        loss = int((delta < -0.001).sum())
        tie = int(len(delta) - win - loss)
        rows.append(
            {
                "backbone": backbone,
                "scope": scope,
                "evidence_tier": tier,
                "n_datasets": int(g["dataset"].nunique()),
                "n_pairs": int(len(g)),
                "csta_u5_mean": float(g["csta_u5_f1"].mean()),
                "no_aug_mean": float(g["no_aug_f1"].mean()),
                "mean_delta_vs_no_aug": float(delta.mean()),
                "W/T/L": f"{win}/{tie}/{loss}",
                "win_rate": float(win / len(g)) if len(g) else np.nan,
                "source_policy": policy,
                "notes": " | ".join(sorted(set(map(str, g["note"].dropna())))),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "evidence_tier", "backbone"])


def _write_report(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Backbone U5 Matrix Audit",
        "",
        "This is a governance artifact built from existing result CSVs. It does not",
        "launch experiments and does not modify locked roots.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No paired CSTA-U5 rows found.")
    else:
        table_cols = [
            "backbone",
            "scope",
            "evidence_tier",
            "n_datasets",
            "n_pairs",
            "csta_u5_mean",
            "no_aug_mean",
            "mean_delta_vs_no_aug",
            "W/T/L",
            "win_rate",
        ]
        lines.append("| " + " | ".join(table_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(table_cols)) + " |")
        for _, row in summary[table_cols].iterrows():
            cells = []
            for col in table_cols:
                value = row[col]
                if isinstance(value, float):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `canonical` rows are the safest paper-facing entries.",
            "- `rebuilt_final20` rows are usable robustness evidence but should keep the rebuilt-root caveat.",
            "- `best_available_recovery` rows combine scattered dataset/recovery roots and must cite the recovery/deduplication policy.",
            "- `pilot_only_u5` rows are not Final20 robustness claims.",
            "- `legacy_noncanonical` rows are drift-audit references only and should not be used as canonical paper numbers.",
            "",
            "## Source Policies",
            "",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['backbone']}` / `{row['scope']}` / `{row['evidence_tier']}`: "
            f"`{row['source_policy']}`. {row['notes']}"
        )
    lines.extend(
        [
            "",
            "## Files Written",
            "",
            "- `backbone_u5_per_seed.csv`",
            "- `backbone_u5_summary.csv`",
            "- `backbone_u5_matrix_report.md`",
        ]
    )
    (OUT_ROOT / "backbone_u5_matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_pairs = []
    source_rows = []
    for spec in _specs():
        df = _read_files(spec.files)
        source_rows.append(
            {
                "backbone": spec.backbone,
                "scope": spec.scope,
                "evidence_tier": spec.evidence_tier,
                "source_policy": spec.source_policy,
                "n_files": len(spec.files),
                "files": ";".join(str(path.relative_to(PROJECT_ROOT)) for path in spec.files),
                "note": spec.note,
            }
        )
        pairs = _pair_table(df, spec)
        if not pairs.empty:
            all_pairs.append(pairs)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(OUT_ROOT / "backbone_u5_sources.csv", index=False)

    paired = pd.concat(all_pairs, ignore_index=True, sort=False) if all_pairs else pd.DataFrame()
    paired.to_csv(OUT_ROOT / "backbone_u5_per_seed.csv", index=False)
    summary = _summarize(paired) if not paired.empty else pd.DataFrame()
    summary.to_csv(OUT_ROOT / "backbone_u5_summary.csv", index=False)
    _write_report(summary, paired)
    print(f"Wrote {OUT_ROOT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
