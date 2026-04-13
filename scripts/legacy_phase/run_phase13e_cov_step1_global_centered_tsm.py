import os
import sys
import json
import random
import re
import subprocess
import traceback
from typing import Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.adapters import get_adapter
from runners.manifold_deep_runner import ManifoldDeepRunner


CONFIG = {
    "dataset": "seed1",
    "seeds": [0, 4],
    "bands_mode": "all5_timecat",
    "band_norm_mode": "per_band_global_z",
    "split_mode": "trial_80_20",
    "audit_key": "subject_session_trial",
    "matrix_mode": "cov",
    "subject_centering": False,
    "global_centering": True,
    "use_roi_pooling": False,
    "guided": False,
    "gate": False,
    "spd_eps": 1e-3,
    "epochs": 10,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "rel_path_base_fmt": "phase13e_cov/step1/seed1/seed{}/global_centered_cov_tsm/manifold",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _parse_trial_id(tid: str) -> Tuple[int, int, int]:
    match = re.match(r"^(\d+)_s(\d+)_t(\d+)$", str(tid))
    if not match:
        raise ValueError(f"Unexpected trial_id format: {tid}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _check_order(ids: list) -> bool:
    parsed = [_parse_trial_id(t) for t in ids]
    return parsed == sorted(parsed)


def _preflight_split_check(fold):
    train_ids = [str(t) for t in fold.trial_id_train]
    test_ids = [str(t) for t in fold.trial_id_test]

    train_unique = len(train_ids) == len(set(train_ids))
    test_unique = len(test_ids) == len(set(test_ids))
    train_ordered = _check_order(train_ids)
    test_ordered = _check_order(test_ids)

    split_check = train_unique and test_unique and train_ordered and test_ordered
    return {
        "n_train_trials": len(train_ids),
        "n_test_trials": len(test_ids),
        "train_unique": train_unique,
        "test_unique": test_unique,
        "train_ordered": train_ordered,
        "test_ordered": test_ordered,
        "split_check": split_check,
    }


def _safe_update_json(path: str, payload: dict) -> None:
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(payload)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_seed(seed: int):
    print(f"\n=== Phase 13E-COV Step1 (Seed {seed}) ===")
    set_seed(seed)

    class Args:
        pass

    args = Args()
    args.dataset_name = CONFIG["dataset"]
    args.dataset = CONFIG["dataset"]
    args.seed = seed

    args.bands_mode = CONFIG["bands_mode"]
    args.band_norm_mode = CONFIG["band_norm_mode"]
    args.matrix_mode = CONFIG["matrix_mode"]
    args.subject_centering = CONFIG["subject_centering"]
    args.global_centering = CONFIG["global_centering"]
    args.mvp1_guided_cov = CONFIG["guided"]
    args.use_band_gate = CONFIG["gate"]
    args.use_roi_pooling = CONFIG["use_roi_pooling"]
    args.epochs = CONFIG["epochs"]
    args.batch_size = CONFIG["batch_size"]
    args.spd_eps = CONFIG["spd_eps"]
    args.lr = CONFIG["lr"]
    args.torch_device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None

    rel_path = CONFIG["rel_path_base_fmt"].format(seed)
    out_dir = f"promoted_results/{rel_path}"
    os.makedirs(out_dir, exist_ok=True)
    args.metrics_csv = os.path.join(out_dir, "manifold_window_pred.csv")

    ckpt_dir = os.path.join("experiments/checkpoints", rel_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    args.dcnet_ckpt = None

    adapter = get_adapter(args.dataset)
    folds = adapter.get_manifold_trial_folds()
    fold = folds["fold1"]

    split_audit = _preflight_split_check(fold)
    split_line = (
        f"[Seed {seed}] Split Check: {'PASS' if split_audit['split_check'] else 'FAIL'} "
        f"(train={split_audit['n_train_trials']} test={split_audit['n_test_trials']})"
    )
    print(split_line)
    if not split_audit["split_check"]:
        raise RuntimeError(f"Split determinism check failed: {split_audit}")

    runner = ManifoldDeepRunner(args, num_classes=3)
    fold_name = f"{rel_path}/report"
    runner.fit_predict(fold, fold_name=fold_name)

    # Generate SINGLE_RUN_REPORT.md
    subprocess.check_call([sys.executable, "scripts/analysis/gen_single_run_report.py", "--root", out_dir])

    # Copy trial predictions to standardized name
    pred_src = f"promoted_results/{fold_name}_preds_test_last_trial.csv"
    pred_dst = os.path.join(out_dir, "manifold_trial_pred.csv")
    if os.path.exists(pred_src):
        os.replace(pred_src, pred_dst)

    # Update diagnostics with split audit
    diag_path = os.path.join(out_dir, "report_diagnostics.json")
    _safe_update_json(diag_path, split_audit)

    # Export meta
    metrics_path = os.path.join(out_dir, "report_metrics.json")
    meta = {
        "global_template_source": "train_only",
        "n_train_windows_used": None,
        "matrix_mode": CONFIG["matrix_mode"],
        "global_centering": CONFIG["global_centering"],
        "seed": seed,
        "split_check": split_audit["split_check"],
        "n_train_trials": split_audit["n_train_trials"],
        "n_test_trials": split_audit["n_test_trials"],
    }
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            counts = metrics.get("metadata", {}).get("counts", {})
            meta["n_train_windows_used"] = counts.get("train_windows")
        except Exception:
            pass
    with open(os.path.join(out_dir, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _aggregate_reports():
    seeds = CONFIG["seeds"]
    rows = []
    for seed in seeds:
        base = f"promoted_results/{CONFIG['rel_path_base_fmt'].format(seed)}"
        metrics_path = os.path.join(base, "report_metrics.json")
        diag_path = os.path.join(base, "report_diagnostics.json")

        metrics = {}
        diag = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        if os.path.exists(diag_path):
            with open(diag_path, "r") as f:
                diag = json.load(f)

        trial_acc = None
        win_acc = None
        if metrics:
            if "best" in metrics:
                trial_acc = metrics["best"].get("test_trial_acc")
                win_acc = metrics["best"].get("test_win_acc")
            elif "last" in metrics:
                trial_acc = metrics["last"].get("test_trial_acc")
                win_acc = metrics["last"].get("test_win_acc")

        row = {
            "seed": seed,
            "trial_acc": f"{trial_acc:.4f}" if trial_acc is not None else "N/A",
            "win_acc": f"{win_acc:.4f}" if win_acc is not None else "N/A",
            "matrix_mode": CONFIG["matrix_mode"],
            "global_center": str(CONFIG["global_centering"]),
            "identity_diff": "N/A",
            "cond_p95": "N/A",
            "eff_rank": "N/A",
            "eps_dom": "N/A",
            "low_eigs": "N/A",
            "split_check": "❌ FAIL",
        }

        if diag:
            if "centering_identity_check" in diag:
                row["identity_diff"] = f"{diag['centering_identity_check']:.1e}"
            if "post_eps_cond_p95" in diag:
                row["cond_p95"] = f"{diag['post_eps_cond_p95']:.1f}"
            if "eff_rank" in diag:
                row["eff_rank"] = f"{diag['eff_rank']:.4f}"
            if "eps_dominance" in diag:
                row["eps_dom"] = f"{diag['eps_dominance']:.4f}"
            if "eigs_le_10eps_count" in diag:
                row["low_eigs"] = f"{diag['eigs_le_10eps_count']:.1f}"
            if diag.get("split_check", False):
                row["split_check"] = "✅ PASS"
        rows.append(row)

    # Write summary.csv
    out_dir = "promoted_results/phase13e_cov/step1/seed1"
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(summary_path, index=False)

    # Build EXPERIMENT_REPORT.md
    md = []
    md.append("# Phase 13E-COV Step 1: Global Log-Euclid Centering on Covariance")
    md.append("")
    md.append("## Summary Table")
    md.append("| seed | trial_acc | win_acc | matrix_mode | global_center | identity_diff | cond_p95 | eff_rank | eps_dom | low_eigs | split_check |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        md.append(
            f"| {r['seed']} | {r['trial_acc']} | {r['win_acc']} | {r['matrix_mode']} | {r['global_center']} | "
            f"{r['identity_diff']} | {r['cond_p95']} | {r['eff_rank']} | {r['eps_dom']} | {r['low_eigs']} | {r['split_check']} |"
        )
    md.append("")

    # Comparison to corr baseline
    corr_summary_path = "promoted_results/phase13e/step4/summary.csv"
    if os.path.exists(corr_summary_path):
        try:
            corr_df = pd.read_csv(corr_summary_path)
            md.append("## Comparison to Corr + Global Centered Baseline (Phase 13E Step 4)")
            md.append("| seed | corr_trial_acc | cov_trial_acc | delta |")
            md.append("| --- | --- | --- | --- |")
            for r in rows:
                seed = r["seed"]
                match = corr_df[corr_df["Seed"] == seed]
                if len(match) > 0:
                    corr_acc = match.iloc[0]["Trial Acc"]
                    try:
                        corr_acc_f = float(corr_acc)
                    except Exception:
                        corr_acc_f = None
                    cov_acc_f = float(r["trial_acc"]) if r["trial_acc"] != "N/A" else None
                    if corr_acc_f is not None and cov_acc_f is not None:
                        md.append(f"| {seed} | {corr_acc_f:.4f} | {cov_acc_f:.4f} | {cov_acc_f - corr_acc_f:.4f} |")
            md.append("")
        except Exception:
            md.append("## Comparison to Corr + Global Centered Baseline (Phase 13E Step 4)")
            md.append("Failed to load baseline summary.")

    md.append("## Artifacts")
    for seed in seeds:
        md.append(f"- Seed {seed}: `promoted_results/{CONFIG['rel_path_base_fmt'].format(seed)}/SINGLE_RUN_REPORT.md`")

    report_path = os.path.join(out_dir, "EXPERIMENT_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md))

    print(f"Experiment Report Generated: {report_path}")


def main():
    for seed in CONFIG["seeds"]:
        try:
            run_seed(seed)
        except Exception:
            traceback.print_exc()
            print(f"Seed {seed} Failed.")
            sys.exit(1)
    _aggregate_reports()


if __name__ == "__main__":
    main()
