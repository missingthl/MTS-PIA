#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts"))

from route_b_unified.regression.evaluator import evaluate_regression  # noqa: E402
from route_b_unified.regression.representation import (  # noqa: E402
    RegressionRepresentationConfig,
    build_regression_representation,
)
from route_b_unified.regression.regressor import RegressorConfig, regressor_params_dict  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _format_value(x: float) -> str:
    return f"{float(x):.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-1 Route B z-space regression baseline (IEEEPPG only)")
    p.add_argument("--dataset", type=str, default="ieeeppg")
    p.add_argument("--out-root", type=str, default="out/regression/route_b_zspace_regression_baseline_20260328")
    p.add_argument("--spd-eps", type=float, default=1e-4)
    p.add_argument("--ieeeppg-root", type=str, default="data/regression/aeon")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--elasticnet-alpha", type=float, default=0.01)
    p.add_argument("--elasticnet-l1-ratio", type=float, default=0.5)
    p.add_argument("--elasticnet-max-iter", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0, help="Model seed only; data split remains official fixed split.")
    args = p.parse_args()

    dataset = str(args.dataset).strip().lower()
    if dataset != "ieeeppg":
        raise ValueError("Phase-1 regression baseline only supports IEEEPPG.")

    _ensure_dir(args.out_root)

    rep_state = build_regression_representation(
        RegressionRepresentationConfig(
            dataset=dataset,
            spd_eps=float(args.spd_eps),
            ieeeppg_root=str(args.ieeeppg_root),
        )
    )

    reg_cfgs = [
        RegressorConfig(regressor_type="ridge", alpha=float(args.ridge_alpha), seed=int(args.seed)),
        RegressorConfig(
            regressor_type="elasticnet",
            alpha=float(args.elasticnet_alpha),
            l1_ratio=float(args.elasticnet_l1_ratio),
            max_iter=int(args.elasticnet_max_iter),
            seed=int(args.seed),
        ),
    ]

    config_rows: List[Dict[str, object]] = []
    per_seed_rows: List[Dict[str, object]] = []
    results: Dict[str, Dict[str, object]] = {}

    representation_settings = {
        "geometry": str(rep_state.meta["geometry"]),
        "spd_eps": float(rep_state.meta["spd_eps"]),
        "z_dim": int(rep_state.meta["z_dim"]),
        "channels": int(rep_state.meta["channels"]),
        "length": int(rep_state.meta["length"]),
    }

    for reg_cfg in reg_cfgs:
        result = evaluate_regression(rep_state, reg_cfg)
        reg_name = str(reg_cfg.regressor_type).strip().lower()
        results[reg_name] = {
            "rmse": float(result.rmse),
            "mae": float(result.mae),
            "r2": float(result.r2),
            "params": dict(result.params),
        }
        config_rows.append(
            {
                "dataset": dataset,
                "split_mode": str(rep_state.split_meta["protocol_type"]),
                "representation_settings": json.dumps(representation_settings, ensure_ascii=False, sort_keys=True),
                "regressor_type": reg_name,
                "regressor_params": json.dumps(regressor_params_dict(reg_cfg), ensure_ascii=False, sort_keys=True),
            }
        )
        per_seed_rows.append(
            {
                "dataset": dataset,
                "seed": int(args.seed),
                "seed_note": "official_fixed_split_no_data_resplit",
                "regressor": reg_name,
                "rmse": float(result.rmse),
                "mae": float(result.mae),
                "r2": float(result.r2),
            }
        )

    best_regressor = max(results.items(), key=lambda kv: float(kv[1]["r2"]))[0]
    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "ridge_rmse": float(results["ridge"]["rmse"]),
                "ridge_mae": float(results["ridge"]["mae"]),
                "ridge_r2": float(results["ridge"]["r2"]),
                "elasticnet_rmse": float(results["elasticnet"]["rmse"]),
                "elasticnet_mae": float(results["elasticnet"]["mae"]),
                "elasticnet_r2": float(results["elasticnet"]["r2"]),
                "best_regressor": str(best_regressor),
            }
        ]
    )

    pd.DataFrame(config_rows).to_csv(os.path.join(args.out_root, "zspace_regression_config_table.csv"), index=False)
    pd.DataFrame(per_seed_rows).to_csv(os.path.join(args.out_root, "zspace_regression_per_seed.csv"), index=False)
    dataset_summary.to_csv(os.path.join(args.out_root, "zspace_regression_dataset_summary.csv"), index=False)

    info = {
        "dataset": dataset,
        "n_train": int(rep_state.meta["n_train"]),
        "n_test": int(rep_state.meta["n_test"]),
        "channels": int(rep_state.meta["channels"]),
        "length": int(rep_state.meta["length"]),
        "z_dim": int(rep_state.meta["z_dim"]),
        "spd_eps": float(rep_state.meta["spd_eps"]),
        "y_train_mean": float(rep_state.meta["y_train_mean"]),
        "y_train_std": float(rep_state.meta["y_train_std"]),
        "y_train_min": float(rep_state.meta["y_train_min"]),
        "y_train_max": float(rep_state.meta["y_train_max"]),
        "y_test_mean": float(rep_state.meta["y_test_mean"]),
        "y_test_std": float(rep_state.meta["y_test_std"]),
        "y_test_min": float(rep_state.meta["y_test_min"]),
        "y_test_max": float(rep_state.meta["y_test_max"]),
        "split_mode": str(rep_state.split_meta["protocol_type"]),
        "split_note": str(rep_state.split_meta["protocol_note"]),
    }
    _write_json(os.path.join(args.out_root, "zspace_regression_dataset_info.json"), info)

    ridge_r2 = float(results["ridge"]["r2"])
    if ridge_r2 > 0.0:
        decision = "worth_phase2_pia_augmentation_in_zspace"
        predictive = "yes"
    else:
        decision = "phase2_not_yet_justified"
        predictive = "no"

    conclusion_lines = [
        "# Z-Space Regression Baseline Conclusion",
        "",
        f"- dataset: `{dataset}`",
        f"- split: `{rep_state.split_meta['protocol_type']}`",
        f"- n_train / n_test: `{int(rep_state.meta['n_train'])} / {int(rep_state.meta['n_test'])}`",
        f"- channels x length: `{int(rep_state.meta['channels'])} x {int(rep_state.meta['length'])}`",
        f"- z_dim: `{int(rep_state.meta['z_dim'])}`",
        "",
        "## Results",
        "",
        f"- Ridge: `RMSE={_format_value(results['ridge']['rmse'])}`, `MAE={_format_value(results['ridge']['mae'])}`, `R2={_format_value(results['ridge']['r2'])}`",
        f"- ElasticNet: `RMSE={_format_value(results['elasticnet']['rmse'])}`, `MAE={_format_value(results['elasticnet']['mae'])}`, `R2={_format_value(results['elasticnet']['r2'])}`",
        f"- best_regressor: `{best_regressor}`",
        "",
        "## Judgement",
        "",
        f"- IEEEPPG 上，当前 z-space 表征是否有预测力：`{predictive}`",
        f"- Ridge 是否已足够形成干净 baseline：`{'yes' if ridge_r2 > 0.0 else 'not_yet'}`",
        f"- 是否值得进入第二阶段（PIA augmentation in z-space）：`{decision}`",
        "- 当前第一阶段未做数据重切或 seed sweep；`seed` 仅表示回归器随机状态，数据 split 使用 aeon 官方 train/test。",
        "",
        "## Reused Modules",
        "",
        "- `datasets/trial_dataset_factory.py`",
        "- `route_b_unified` 几何表示主逻辑（covariance -> log-Euclidean -> z-space）",
        "- `scripts/run_phase14r_step6b1_rev2.py` 中的 `logm_spd / vec_utri`",
        "- 未复用 `bridge.py`、`raw MiniROCKET evaluator`、`augmentation_admission.py`",
        "",
    ]
    with open(os.path.join(args.out_root, "zspace_regression_conclusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines))

    print(f"[zspace-regression] wrote {os.path.join(args.out_root, 'zspace_regression_config_table.csv')}", flush=True)
    print(f"[zspace-regression] wrote {os.path.join(args.out_root, 'zspace_regression_per_seed.csv')}", flush=True)
    print(f"[zspace-regression] wrote {os.path.join(args.out_root, 'zspace_regression_dataset_summary.csv')}", flush=True)
    print(f"[zspace-regression] wrote {os.path.join(args.out_root, 'zspace_regression_conclusion.md')}", flush=True)


if __name__ == "__main__":
    main()
