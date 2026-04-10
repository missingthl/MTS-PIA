import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def _fmt_float(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _risk_flags(row: dict) -> list:
    flags = []
    if int(row.get("nan_inf_count", 0)) > 0:
        flags.append("NaN/Inf")
    vstats = row.get("valid_len_stats") or {}
    try:
        p50 = float(vstats.get("p50", 0))
        vmax = float(vstats.get("max", 0))
        if vmax > 0 and p50 < 0.5 * vmax:
            flags.append("short_len")
    except Exception:
        pass
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    rows = [_load(Path(p)) for p in args.inputs]
    if not rows:
        raise ValueError("no input jsons provided")

    baseline = rows[0]
    base_acc = float(baseline.get("val_acc", 0.0))
    base_f1 = float(baseline.get("val_macro_f1", 0.0))

    lines = []
    lines.append("# baseline_tsm_smooth summary")
    lines.append("")
    lines.append("| name | val_acc | val_macro_f1 | train_acc | nan_inf | valid_len_p50/p95 | n_train | n_val | delta_acc | delta_f1 | flags |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in rows:
        name = Path(row.get("_path", "")).name
        val_acc = float(row.get("val_acc", 0.0))
        val_f1 = float(row.get("val_macro_f1", 0.0))
        train_acc = float(row.get("train_acc", 0.0))
        nan_inf = int(row.get("nan_inf_count", 0))
        vstats = row.get("valid_len_stats") or {}
        v_p50 = vstats.get("p50")
        v_p95 = vstats.get("p95")
        n_train = int(row.get("train_size", 0))
        n_val = int(row.get("val_size", 0))
        delta_acc = val_acc - base_acc
        delta_f1 = val_f1 - base_f1
        flags = ",".join(_risk_flags(row)) or "-"
        lines.append(
            "| {name} | {val_acc} | {val_f1} | {train_acc} | {nan_inf} | {v_p50}/{v_p95} | {n_train} | {n_val} | {delta_acc} | {delta_f1} | {flags} |".format(
                name=name,
                val_acc=_fmt_float(val_acc),
                val_f1=_fmt_float(val_f1),
                train_acc=_fmt_float(train_acc),
                nan_inf=nan_inf,
                v_p50=v_p50 if v_p50 is not None else "n/a",
                v_p95=v_p95 if v_p95 is not None else "n/a",
                n_train=n_train,
                n_val=n_val,
                delta_acc=_fmt_float(delta_acc),
                delta_f1=_fmt_float(delta_f1),
                flags=flags,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[summary] wrote {out_path}")


if __name__ == "__main__":
    main()
