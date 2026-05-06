from __future__ import annotations

import os

import numpy as np

from core.csta.pipelines import (
    _run_act_pipeline,
    _run_act_rc4_multiz_fused_pipeline,
    _run_act_zpia_template_pool_pipeline,
)
from core.csta.result_rows import (
    build_failure_result_row,
    build_success_result_row,
    merge_candidate_audit_summary,
)
from core.csta.state import build_trial_records as _build_trial_records
from utils.datasets import load_trials_for_dataset, make_trial_split


def run_experiment(dataset_name, args):
    print(f"\n>>>> Dataset: {dataset_name} | Model: {args.model} <<<<")
    try:
        all_trials = load_trials_for_dataset(dataset_name)
    except Exception as exc:
        print(f"Failed to load {dataset_name}: {exc}")
        return [build_failure_result_row(dataset_name=dataset_name, seed=-1, args=args, fail_reason=str(exc))]

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    patience = args.patience

    if args.host_config != "none":
        if args.host_config == "resnet1d_default":
            epochs, lr, batch_size, patience = 30, 1e-3, 64, 10
        elif args.host_config == "patchtst_default":
            epochs, lr, batch_size, patience = 100, 5e-4, 64, 15
        elif args.host_config == "timesnet_default":
            epochs, lr, batch_size, patience = 100, 5e-4, 32, 15

    results = []
    seeds = [int(seed) for seed in args.seeds.split(",")]
    for seed in seeds:
        print(f"Seed {seed}...")
        try:
            train_trials, test_trials, val_trials = make_trial_split(all_trials, seed=seed, val_ratio=args.val_ratio)
            train_recs, mean_log = _build_trial_records(train_trials)
            test_recs, _ = _build_trial_records(test_trials)
            val_recs, _ = _build_trial_records(val_trials)

            X_train_raw = np.stack([record.x_raw for record in train_recs])
            y_train = np.asarray([record.y for record in train_recs], dtype=np.int64)
            X_test_raw = np.stack([record.x_raw for record in test_recs])
            y_test = np.asarray([record.y for record in test_recs], dtype=np.int64)

            X_val_raw, y_val = None, None
            if val_recs:
                X_val_raw = np.stack([record.x_raw for record in val_recs])
                y_val = np.asarray([record.y for record in val_recs], dtype=np.int64)

            X_train_z = np.stack([record.z for record in train_recs])
            if args.algo == "zpia_top1_pool":
                pipeline_out = _run_act_zpia_template_pool_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                    algo_label="zpia_top1_pool",
                    top1_only=True,
                )
            elif args.algo == "zpia_multidir_pool":
                pipeline_out = _run_act_zpia_template_pool_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                    algo_label="zpia_multidir_pool",
                    top1_only=False,
                )
            elif args.algo == "rc4_multiz_fused":
                pipeline_out = _run_act_rc4_multiz_fused_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            elif args.algo == "pia":
                pipeline_out = _run_act_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            else:
                # Default to base ACT pipeline for lraes/zpia
                pipeline_out = _run_act_pipeline(
                    args=args,
                    seed=seed,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    X_train_z=X_train_z,
                    train_recs=train_recs,
                    mean_log=mean_log,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    patience=patience,
                )
            summary = build_success_result_row(
                dataset_name=dataset_name,
                seed=seed,
                args=args,
                pipeline_out=pipeline_out,
                y_train=y_train,
            )
            print(
                f"Base: {summary['base_f1']:.4f} | "
                f"ACT: {summary['act_f1']:.4f} | "
                f"Gain: {summary['gain']:.4f} ({summary['f1_gain_pct']:.1f}%)"
            )
            audit_rows = list(pipeline_out.get("audit_rows", []))
            summary = merge_candidate_audit_summary(
                summary=summary,
                audit_rows=audit_rows,
                args=args,
                dataset_name=dataset_name,
                seed=int(seed),
                eta_safe=pipeline_out.get("eta_safe", None),
            )
            results.append(summary)

            if args.save_viz_samples:
                viz_dir = os.path.join(args.out_root, "viz_data")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{dataset_name}_s{seed}_viz.npz")
                np.savez(
                    save_path,
                    Z_orig=pipeline_out["viz_payload"].get("Z_orig", X_train_z),
                    y_orig=y_train,
                    Z_aug=pipeline_out["viz_payload"]["Z_aug"],
                    y_aug=pipeline_out["viz_payload"]["y_aug"],
                    X_orig_raw=X_train_raw[:20],
                    X_aug_raw=pipeline_out["viz_payload"]["X_aug_raw"],
                    mean_log=mean_log,
                )
                print(f"Visualization samples saved to {save_path}")

        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"Error in {dataset_name} Seed {seed}: {exc}")
            results.append(
                build_failure_result_row(dataset_name=dataset_name, seed=seed, args=args, fail_reason=str(exc))
            )
    return results
