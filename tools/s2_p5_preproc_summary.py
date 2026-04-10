from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var**0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize S2-P5 SPD preprocessing runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="list of run directories")
    parser.add_argument("--out", required=True, help="output markdown path")
    args = parser.parse_args()

    rows = []
    by_kind: Dict[str, Dict[str, List[float]]] = {}
    audit_rows = []
    for run in args.runs:
        run_dir = Path(run)
        metrics = _read_json(run_dir / "metrics.json")
        cfg = _read_json(run_dir / "run_config.json")
        spd_audit = _read_json(run_dir / "spd_audit.json")
        spd_kind = cfg.get("spd_kind", "cov")
        seed = cfg.get("split_seed")
        rows.append(
            {
                "spd_kind": spd_kind,
                "seed": seed,
                "val_acc": metrics.get("val_acc"),
                "val_macro_f1": metrics.get("val_macro_f1"),
            }
        )
        by_kind.setdefault(spd_kind, {"acc": [], "f1": []})
        by_kind[spd_kind]["acc"].append(metrics.get("val_acc"))
        by_kind[spd_kind]["f1"].append(metrics.get("val_macro_f1"))
        audit_rows.append(
            {
                "spd_kind": spd_kind,
                "seed": seed,
                "train_min_eig": spd_audit["train"]["min_eig"],
                "train_cond": spd_audit["train"]["cond"],
                "train_eps_ratio": spd_audit["train"]["eps_injected_trace_ratio"],
                "val_min_eig": spd_audit["val"]["min_eig"],
                "val_cond": spd_audit["val"]["cond"],
                "val_eps_ratio": spd_audit["val"]["eps_injected_trace_ratio"],
            }
        )

    lines: List[str] = []
    lines.append("| spd_kind | seed | val_acc | val_macro_f1 |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['spd_kind']} | {row['seed']} | {row['val_acc']:.4f} | {row['val_macro_f1']:.4f} |"
        )

    lines.append("")
    lines.append("## Mean ± Std by spd_kind")
    lines.append("| spd_kind | acc_mean | acc_std | f1_mean | f1_std |")
    lines.append("| --- | --- | --- | --- | --- |")
    for kind, vals in by_kind.items():
        acc_mean, acc_std = _mean_std(vals["acc"])
        f1_mean, f1_std = _mean_std(vals["f1"])
        lines.append(
            f"| {kind} | {acc_mean:.4f} | {acc_std:.4f} | {f1_mean:.4f} | {f1_std:.4f} |"
        )

    lines.append("")
    lines.append("## SPD audit (train/val p50/p95/p99)")
    for row in audit_rows:
        lines.append(
            f"- {row['spd_kind']} seed={row['seed']} "
            f"train min_eig={row['train_min_eig']} cond={row['train_cond']} "
            f"eps_ratio={row['train_eps_ratio']} "
            f"val min_eig={row['val_min_eig']} cond={row['val_cond']} "
            f"eps_ratio={row['val_eps_ratio']}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[summary] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
