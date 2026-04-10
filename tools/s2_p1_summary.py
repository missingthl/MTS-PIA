from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize S2 trial ablation runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="list of run directories")
    parser.add_argument("--out", required=True, help="output markdown path")
    args = parser.parse_args()

    rows = []
    spd_audit = None
    tsm_audit = None
    for run in args.runs:
        run_dir = Path(run)
        metrics = _read_json(run_dir / "metrics.json")
        cfg = _read_json(run_dir / "run_config.json")
        rows.append(
            {
                "mode": cfg.get("s2_mode"),
                "classifier": cfg.get("classifier"),
                "val_acc": metrics.get("val_acc"),
                "val_macro_f1": metrics.get("val_macro_f1"),
            }
        )
        if spd_audit is None and (run_dir / "spd_audit.json").exists():
            spd_audit = _read_json(run_dir / "spd_audit.json")
        if tsm_audit is None and (run_dir / "tsm_audit.json").exists():
            tsm_audit = _read_json(run_dir / "tsm_audit.json")

    lines: List[str] = []
    lines.append("| mode | classifier | val_acc | val_macro_f1 |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['classifier']} | {row['val_acc']:.4f} | {row['val_macro_f1']:.4f} |"
        )
    lines.append("")
    if spd_audit:
        lines.append("## SPD audit (p50/p95/p99)")
        for split in ["train", "val"]:
            stats = spd_audit.get(split, {})
            lines.append(
                f"- {split}: min_eig={stats.get('min_eig', {})} "
                f"cond={stats.get('cond', {})} symmetry_error={stats.get('symmetry_error', {})}"
            )
    if tsm_audit:
        lines.append("")
        lines.append("## TSM audit")
        lines.append(f"- tsm_dim={tsm_audit.get('tsm_dim')}")
        lines.append(f"- tsm_norm={tsm_audit.get('tsm_norm')}")
        lines.append(f"- nan_count={tsm_audit.get('nan_count')} inf_count={tsm_audit.get('inf_count')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[summary] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
