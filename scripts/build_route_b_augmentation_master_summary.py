#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


TRAIN_REPLACE_SUMMARY = "augmentation_train_replace_summary.csv"
BEST_ROUND_POOL_SUMMARY = "augmentation_best_round_pool_summary.csv"
FILTERED_POOL_SUMMARY = "augmentation_filtered_pool_summary.csv"
SEED_LIGHT_SUMMARY = "seed_family_bridge_train_replace_summary.csv"
UNIFIED_SMOKE_SUMMARY = "route_b_unified_smoke_expansion_summary.csv"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_mean(text: object) -> float:
    if text is None:
        return float("nan")
    s = str(text).strip()
    if not s:
        return float("nan")
    if "+/-" in s:
        s = s.split("+/-", 1)[0].strip()
    try:
        return float(s)
    except Exception:
        return float("nan")


def _gather_dataset_summaries(root: str, filename: str) -> pd.DataFrame:
    base = Path(root)
    if not base.exists():
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for path in sorted(base.glob(f"*/{filename}")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_stage_outputs(
    *,
    df: pd.DataFrame,
    out_root: str,
    stage_prefix: str,
    title: str,
) -> None:
    _ensure_dir(out_root)
    by_dataset_path = os.path.join(out_root, f"{stage_prefix}_by_dataset.csv")
    summary_path = os.path.join(out_root, f"{stage_prefix}_summary.csv")
    conclusion_path = os.path.join(out_root, f"{stage_prefix}_conclusion.md")
    df.to_csv(by_dataset_path, index=False)
    df.to_csv(summary_path, index=False)
    lines = [f"# {title}", "", f"- dataset_count: `{len(df)}`", ""]
    if not df.empty and "augmentation_label" in df.columns:
        label_counts = df["augmentation_label"].fillna("n/a").value_counts().to_dict()
        lines.append("## Label Count")
        lines.append("")
        for key in ["positive", "flat", "negative", "n/a"]:
            if key in label_counts:
                lines.append(f"- `{key}`: `{int(label_counts[key])}`")
        lines.append("")
    lines.append("## Rows")
    lines.append("")
    for _, row in df.iterrows():
        dataset = row.get("dataset", "")
        best_variant = row.get("best_variant", "")
        delta_vs_raw = row.get("delta_vs_raw", "")
        label = row.get("augmentation_label", "")
        lines.append(
            f"- `{dataset}`: best_variant=`{best_variant}`, delta_vs_raw=`{delta_vs_raw}`, label=`{label}`"
        )
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_master_summary(
    *,
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    filt_df: pd.DataFrame,
    seed_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = [df for df in [train_df, pool_df, filt_df, seed_df] if not df.empty]
    if not frames:
        return pd.DataFrame()

    keys = ["dataset", "dataset_group", "protocol"]
    master = pd.concat(frames, ignore_index=True)
    if "mode" in master.columns:
        master = master.drop(columns=["mode"])
    master = master.drop_duplicates(subset=keys, keep="first").set_index(keys)

    def merge_stage(df: pd.DataFrame, allowed_cols: List[str]) -> None:
        nonlocal master
        if df.empty:
            return
        cur = df.set_index(keys)
        for col in [c for c in allowed_cols if c in cur.columns]:
            if col not in master.columns:
                master[col] = ""
            series = cur[col]
            mask = series.notna()
            if series.dtype == object:
                mask = mask & series.astype(str).ne("")
            if mask.any():
                master.loc[series.index[mask], col] = series[mask]

    merge_stage(
        train_df,
        [
            "seed_count",
            "raw_minirocket_acc",
            "raw_minirocket_macro_f1",
            "bridge_single_train_acc",
            "bridge_single_train_macro_f1",
            "bridge_final_train_acc",
            "bridge_final_train_macro_f1",
            "bridge_best_round_train_acc",
            "bridge_best_round_train_macro_f1",
            "best_rounds",
            "note",
        ],
    )
    merge_stage(
        pool_df,
        [
            "bridge_best_round_pool_acc",
            "bridge_best_round_pool_macro_f1",
            "best_rounds",
            "note",
        ],
    )
    merge_stage(
        filt_df,
        [
            "bridge_filtered_pool_acc",
            "bridge_filtered_pool_macro_f1",
            "best_rounds",
            "note",
        ],
    )
    merge_stage(
        seed_df,
        [
            "seed_count",
            "raw_minirocket_acc",
            "raw_minirocket_macro_f1",
            "bridge_best_round_train_acc",
            "bridge_best_round_train_macro_f1",
            "best_rounds",
            "note",
        ],
    )

    out = master.reset_index()
    if "mode" in out.columns:
        out = out.drop(columns=["mode"])

    best_variants: List[str] = []
    deltas: List[float] = []
    labels: List[str] = []
    notes: List[str] = []
    variant_priority = [
        "bridge_filtered_pool",
        "bridge_best_round_pool",
        "bridge_multiround_best_round",
        "bridge_multiround_final",
        "bridge_single_round",
        "raw",
    ]
    for _, row in out.iterrows():
        variant_cols = {
            "raw": row.get("raw_minirocket_macro_f1", ""),
            "bridge_single_round": row.get("bridge_single_train_macro_f1", ""),
            "bridge_multiround_final": row.get("bridge_final_train_macro_f1", ""),
            "bridge_multiround_best_round": row.get("bridge_best_round_train_macro_f1", ""),
            "bridge_best_round_pool": row.get("bridge_best_round_pool_macro_f1", ""),
            "bridge_filtered_pool": row.get("bridge_filtered_pool_macro_f1", ""),
        }
        scores = {k: _extract_mean(v) for k, v in variant_cols.items()}
        valid_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not valid_scores:
            best_variants.append("")
            deltas.append(float("nan"))
            labels.append("flat")
            notes.append(str(row.get("note", "")))
            continue
        best_variant = max(
            valid_scores.items(),
            key=lambda kv: (
                kv[1],
                -variant_priority.index(kv[0]) if kv[0] in variant_priority else -999,
            ),
        )[0]
        raw_score = scores.get("raw", float("nan"))
        delta_vs_raw = float(valid_scores[best_variant] - raw_score) if np.isfinite(raw_score) else float("nan")
        if delta_vs_raw >= 0.002:
            label = "positive"
        elif delta_vs_raw <= -0.002:
            label = "negative"
        else:
            label = "flat"
        best_variants.append(best_variant)
        deltas.append(delta_vs_raw)
        labels.append(label)
        notes.append(str(row.get("note", "")))

    out["best_variant"] = best_variants
    out["delta_vs_raw"] = deltas
    out["augmentation_label"] = labels
    out["note"] = notes
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build Route B augmentation master summaries.")
    p.add_argument("--unified-root", type=str, default="out/route_b_unified_smoke_expansion_20260322")
    p.add_argument("--train-replace-root", type=str, default="out/route_b_augmentation_train_replace_20260322")
    p.add_argument("--best-round-pool-root", type=str, default="out/route_b_augmentation_best_round_pool_20260322")
    p.add_argument("--filtered-pool-root", type=str, default="out/route_b_augmentation_filtered_pool_20260322")
    p.add_argument("--seed-light-root", type=str, default="out/route_b_augmentation_seed_light_best_round_20260322")
    p.add_argument("--out-root", type=str, default="out/route_b_augmentation_master_20260322")
    args = p.parse_args()

    _ensure_dir(args.out_root)

    unified_summary_src = os.path.join(args.unified_root, "final_coupling_summary.csv")
    unified_conclusion_src = os.path.join(args.unified_root, "route_b_unified_conclusion.md")
    if os.path.exists(unified_summary_src):
        shutil.copyfile(unified_summary_src, os.path.join(args.out_root, UNIFIED_SMOKE_SUMMARY))
    if os.path.exists(unified_conclusion_src):
        shutil.copyfile(
            unified_conclusion_src,
            os.path.join(args.out_root, "route_b_unified_smoke_expansion_conclusion.md"),
        )

    train_df = _gather_dataset_summaries(args.train_replace_root, TRAIN_REPLACE_SUMMARY)
    pool_df = _gather_dataset_summaries(args.best_round_pool_root, BEST_ROUND_POOL_SUMMARY)
    filt_df = _gather_dataset_summaries(args.filtered_pool_root, FILTERED_POOL_SUMMARY)
    seed_df = _gather_dataset_summaries(args.seed_light_root, SEED_LIGHT_SUMMARY)

    if not train_df.empty:
        _write_stage_outputs(
            df=train_df,
            out_root=args.out_root,
            stage_prefix="augmentation_train_replace",
            title="Route B Augmentation Train Replace",
        )
    if not pool_df.empty:
        _write_stage_outputs(
            df=pool_df,
            out_root=args.out_root,
            stage_prefix="augmentation_best_round_pool",
            title="Route B Augmentation Best-Round Neighborhood Pool",
        )
    if not filt_df.empty:
        _write_stage_outputs(
            df=filt_df,
            out_root=args.out_root,
            stage_prefix="augmentation_filtered_pool",
            title="Route B Augmentation Filtered Pool",
        )
    if not seed_df.empty:
        seed_summary_path = os.path.join(args.out_root, SEED_LIGHT_SUMMARY)
        seed_df.to_csv(seed_summary_path, index=False)

    master_df = _build_master_summary(
        train_df=train_df,
        pool_df=pool_df,
        filt_df=filt_df,
        seed_df=seed_df,
    )
    if not master_df.empty:
        master_path = os.path.join(args.out_root, "augmentation_all_datasets_master_summary.csv")
        master_df.to_csv(master_path, index=False)


if __name__ == "__main__":
    main()
