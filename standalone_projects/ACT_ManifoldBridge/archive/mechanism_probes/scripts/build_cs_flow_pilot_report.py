#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


CS_DIAG_FIELDS = [
    "cs_flow_train_mse_mean",
    "cs_flow_train_cosine_mean",
    "cs_flow_pred_target_cosine_mean",
    "cs_flow_target_dist_mean",
    "cs_flow_target_dist_std",
    "cs_flow_velocity_norm_mean",
    "cs_flow_velocity_norm_std",
    "cs_flow_fallback_rate",
    "cs_flow_target_effective_rank",
    "cs_flow_target_pairwise_cosine_mean",
    "cs_flow_velocity_effective_rank",
    "cs_flow_velocity_pairwise_cosine_mean",
    "unique_direction_ratio",
    "generated_direction_pairwise_cosine_mean",
    "effective_aug_multiplier",
    "bridge_success_rate",
    "safe_clip_rate",
    "gamma_used_ratio_mean",
    "transport_error_logeuc_mean",
]


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _fmt(value: object) -> str:
    val = _float(value)
    if math.isnan(val):
        return ""
    return f"{val:.6g}"


def _markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _wins(rows: List[Dict[str, str]], method: str, ref: str) -> tuple[int, int, int]:
    by_key = {(r["dataset"], r["seed"], r["method"]): _float(r.get("aug_f1")) for r in rows}
    keys = sorted({(r["dataset"], r["seed"]) for r in rows if r["method"] == method})
    wins = ties = losses = 0
    for key in keys:
        a = by_key.get((key[0], key[1], method), math.nan)
        b = by_key.get((key[0], key[1], ref), math.nan)
        if math.isnan(a) or math.isnan(b):
            continue
        if abs(a - b) <= 1e-12:
            ties += 1
        elif a > b:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def build_report(out_root: Path) -> None:
    source = out_root / "per_seed_external.csv"
    rows = _read_rows(source)
    by_method: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status", "success") == "success":
            by_method[row["method"]].append(row)

    summary_rows: List[Dict[str, object]] = []
    def _method_mean(method: str) -> float:
        vals = [_float(r["aug_f1"]) for r in by_method.get(method, [])]
        vals = [v for v in vals if not math.isnan(v)]
        return mean(vals) if vals else math.nan

    u5_mean = _method_mean("csta_topk_uniform_top5")
    random_mean = _method_mean("random_cov_state")
    bank_mean = _method_mean("csta_template_random_within_bank")
    for method, method_rows in sorted(by_method.items()):
        f1s = [_float(r["aug_f1"]) for r in method_rows]
        f1s = [v for v in f1s if not math.isnan(v)]
        item: Dict[str, object] = {
            "method": method,
            "role": "debug_probe_not_competing_method" if method == "cs_flow_target_direct" else "method",
            "n_rows": len(method_rows),
            "n_datasets": len({r["dataset"] for r in method_rows}),
            "n_seeds": len({r["seed"] for r in method_rows}),
            "mean_f1": mean(f1s) if f1s else math.nan,
            "std_f1": pstdev(f1s) if len(f1s) > 1 else 0.0,
        }
        item["delta_vs_u5"] = _float(item["mean_f1"]) - u5_mean if not math.isnan(u5_mean) else math.nan
        item["delta_vs_random_cov"] = _float(item["mean_f1"]) - random_mean if not math.isnan(random_mean) else math.nan
        item["delta_vs_bank_random"] = _float(item["mean_f1"]) - bank_mean if not math.isnan(bank_mean) else math.nan
        for ref in ["csta_topk_uniform_top5", "random_cov_state", "csta_template_random_within_bank"]:
            w, t, l = _wins(rows, method, ref)
            item[f"wins_vs_{ref}"] = w
            item[f"ties_vs_{ref}"] = t
            item[f"losses_vs_{ref}"] = l
        for field in CS_DIAG_FIELDS:
            vals = [_float(r.get(field)) for r in method_rows]
            vals = [v for v in vals if not math.isnan(v)]
            item[field] = mean(vals) if vals else math.nan
        summary_rows.append(item)

    summary_path = out_root / "cs_flow_pilot_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    leaderboard = sorted(summary_rows, key=lambda r: _float(r["mean_f1"]), reverse=True)
    leaderboard_rows = [
        [r["method"], r["role"], r["n_rows"], _fmt(r["mean_f1"]), _fmt(r["delta_vs_u5"]), _fmt(r["delta_vs_random_cov"]), _fmt(r["delta_vs_bank_random"])]
        for r in leaderboard
    ]
    cs_rows = [
        [
            r["method"],
            _fmt(r.get("cs_flow_train_mse_mean")),
            _fmt(r.get("cs_flow_train_cosine_mean")),
            _fmt(r.get("cs_flow_pred_target_cosine_mean")),
            _fmt(r.get("cs_flow_fallback_rate")),
            _fmt(r.get("effective_aug_multiplier")),
            _fmt(r.get("generated_direction_pairwise_cosine_mean")),
            _fmt(r.get("bridge_success_rate")),
        ]
        for r in leaderboard
        if str(r["method"]).startswith("cs_flow_")
    ]
    collapse_notes = []
    for r in leaderboard:
        if not str(r["method"]).startswith("cs_flow_"):
            continue
        eff = _float(r.get("effective_aug_multiplier"))
        cos = _float(r.get("generated_direction_pairwise_cosine_mean"))
        if (not math.isnan(eff) and eff < 5.0) or (not math.isnan(cos) and cos > 0.95):
            collapse_notes.append(
                f"- `{r['method']}` shows possible generated-direction collapse before F1 interpretation "
                f"(effective_aug_multiplier={_fmt(eff)}, pairwise_cosine={_fmt(cos)})."
            )
    if not collapse_notes:
        collapse_notes.append("- No CS-Flow generated-direction collapse flag under the v1 heuristic.")

    report = [
        "# CS-Flow Pilot Report",
        "",
        f"Source: `{source}`",
        "",
        "`cs_flow_target_direct` is a debug probe, not a competing paper baseline.",
        "",
        "## Leaderboard",
        "",
        _markdown_table(
            ["method", "role", "n_rows", "mean_f1", "delta_vs_u5", "delta_vs_random_cov", "delta_vs_bank_random"],
            leaderboard_rows,
        ),
        "",
        "## CS-Flow Diagnostics",
        "",
        _markdown_table(
            [
                "method",
                "train_mse",
                "train_cosine",
                "pred_target_cosine",
                "fallback_rate",
                "effective_aug_multiplier",
                "generated_pairwise_cosine",
                "bridge_success_rate",
            ],
            cs_rows,
        ),
        "",
        "## Diversity Collapse Check",
        "",
        "\n".join(collapse_notes),
        "",
        "## Interpretation Guardrails",
        "",
        "- Do not tune CS-Flow hyperparameters inside Phase 1 after this Pilot3.",
        "- If target-direct is strong but learned flow is weak, inspect fitting diagnostics before Pilot7.",
        "- Stop v1 if both target-direct and learned flow are below random covariance control.",
    ]
    (out_root / "cs_flow_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CS-Flow Pilot3 report.")
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    build_report(Path(args.out_root))


if __name__ == "__main__":
    main()
    report.append(_markdown_table(headers, lb_rows))
    report.append("")

    # Diagnostics Table
    report.append("## CS-Flow Diagnostics")
    diag_headers = ["Method", "Train MSE", "Pred-Target Cos", "Effective Rank", "Pairwise Cosine", "Uniq Ratio", "Bridge Succ"]
    diag_rows = []
    for r in summary_rows:
        if "cs_flow" in r["method"]:
            diag_rows.append([
                r["method"],
                _fmt(r["cs_flow_train_mse_mean"]),
                _fmt(r["cs_flow_pred_target_cosine_mean"]),
                _fmt(r["cs_flow_velocity_effective_rank"]),
                _fmt(r["generated_direction_pairwise_cosine_mean"]),
                _fmt(r["unique_direction_ratio"]),
                _fmt(r["bridge_success_rate"])
            ])
    report.append(_markdown_table(diag_headers, diag_rows))
    report.append("")

    cs_res = next((r for r in summary_rows if r["method"] == "cs_flow_single_step"), None)
    if cs_res:
        paired_rows = []
        for ref_name in ["u5", "random_cov", "wdba", "bank_random"]:
            paired_rows.append(
                [
                    ref_name,
                    _fmt(cs_res.get(f"paired_delta_vs_{ref_name}_mean")),
                    f"[{_fmt(cs_res.get(f'paired_delta_vs_{ref_name}_ci_low'))}, {_fmt(cs_res.get(f'paired_delta_vs_{ref_name}_ci_high'))}]",
                    f"{cs_res.get(f'seed_wins_vs_{ref_name}', 0)}/{cs_res.get(f'seed_ties_vs_{ref_name}', 0)}/{cs_res.get(f'seed_losses_vs_{ref_name}', 0)}",
                    f"{cs_res.get(f'dataset_wins_vs_{ref_name}', 0)}/{cs_res.get(f'dataset_ties_vs_{ref_name}', 0)}/{cs_res.get(f'dataset_losses_vs_{ref_name}', 0)}",
                    _fmt(cs_res.get(f"wilcoxon_p_vs_{ref_name}", np.nan)),
                ]
            )
        report.append("## Paired CS-Flow Comparisons")
        report.append(
            _markdown_table(
                ["reference", "mean_delta", "bootstrap_95ci", "seed_W/T/L", "dataset_W/T/L", "wilcoxon_p"],
                paired_rows,
            )
        )
        report.append("")

    # Specialized Interpretation
    report.append("## Critical Mechanism Analysis")
    if cs_res:
        cos = cs_res.get("generated_direction_pairwise_cosine_mean", 0)
        if cos > 0.95:
            report.append("> [!WARNING]")
            report.append("> **Concentrated Dominant Flow Detected**")
            report.append(f"> CS-Flow v1 learns concentrated dominant vicinal flow directions (pairwise_cosine={_fmt(cos)}) rather than high-diversity generation.")
        else:
            report.append("CS-Flow v1 shows healthy directional diversity.")

    report.append("")
    report.append("## Execution Guardrails")
    pilot_pass = cs_res and cs_res["mean_f1"] > ref_f1s["u5"]
    report.append("- Pilot7 stability gate status: " + ("**PASSED**" if pilot_pass else "**PENDING/FAILED**"))
    
    (out_root / "cs_flow_pilot_report.md").write_text("\n".join(report), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    build_report(Path(args.out_root))

if __name__ == "__main__":
    main()
