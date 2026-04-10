from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize S2-P2 windowed runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="list of run directories")
    parser.add_argument("--out", required=True, help="output markdown path")
    args = parser.parse_args()

    rows = []
    window_audits = []
    for run in args.runs:
        run_dir = Path(run)
        metrics = _read_json(run_dir / "metrics.json")
        cfg = _read_json(run_dir / "run_config.json")
        rows.append(
            {
                "mode": cfg.get("data_mode"),
                "W": cfg.get("window_size"),
                "classifier": cfg.get("classifier"),
                "s2_mode": cfg.get("s2_mode"),
                "val_acc": metrics.get("val_acc"),
                "val_macro_f1": metrics.get("val_macro_f1"),
            }
        )
        wa = run_dir / "window_audit.json"
        if wa.exists():
            audit = _read_json(wa)
            audit["run_dir"] = str(run_dir)
            window_audits.append(audit)

    lines: List[str] = []
    lines.append("| mode | W | classifier | s2_mode | val_acc | val_macro_f1 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['W']} | {row['classifier']} | {row['s2_mode']} | "
            f"{row['val_acc']:.4f} | {row['val_macro_f1']:.4f} |"
        )
    lines.append("")
    if window_audits:
        lines.append("## Window count audit")
        for audit in window_audits:
            lines.append(
                f"- run={audit['run_dir']} W={audit.get('window_size')} step={audit.get('window_step')} "
                f"count_min={audit.get('count_min')} mean={audit.get('count_mean'):.2f} "
                f"max={audit.get('count_max')}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[summary] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
