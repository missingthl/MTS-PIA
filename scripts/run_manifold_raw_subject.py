from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runners.manifold_raw_v1 import ManifoldRawV1Runner


def _load_manifest(path: str) -> List[dict]:
    manifest_path = None
    if "*" in path or "?" in path:
        matches = sorted(glob.glob(path))
        if matches:
            manifest_path = matches[-1]
    elif os.path.isfile(path):
        manifest_path = path

    if not manifest_path:
        raise FileNotFoundError(f"manifest not found: {path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "trials" not in data:
            raise ValueError(f"manifest dict missing 'trials': {manifest_path}")
        return data["trials"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported manifest format: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="seed1")
    parser.add_argument("--raw-manifest", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument(
        "--seed-raw-root",
        type=str,
        default="data/SEED/SEED_EEG/SEED_RAW_EEG",
    )
    parser.add_argument(
        "--seed-raw-backend",
        type=str,
        default="fif",
        choices=["cnt", "fif"],
    )
    parser.add_argument("--raw-window-sec", type=float, default=4.0)
    parser.add_argument("--raw-window-hop-sec", type=float, default=4.0)
    parser.add_argument("--raw-resample-fs", type=float, default=0.0)
    parser.add_argument(
        "--raw-bands",
        type=str,
        default="delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    )
    parser.add_argument(
        "--raw-time-unit",
        type=str,
        default="",
        help="time unit for time.txt (samples@1000, samples@200, seconds)",
    )
    parser.add_argument(
        "--raw-trial-offset-sec",
        type=float,
        default=-3.0,
        help="global trial offset in seconds (applied to start/end)",
    )
    parser.add_argument("--raw-cov", type=str, default="shrinkage_oas")
    parser.add_argument("--raw-logmap-eps", type=float, default=1e-6)
    parser.add_argument(
        "--raw-seq-save-format",
        type=str,
        default="vec_utri",
        choices=["vec_utri", "cov_spd"],
    )
    parser.add_argument("--spd-eps", type=float, default=1e-5)
    parser.add_argument(
        "--spd-eps-mode",
        type=str,
        default="relative_trace",
        choices=["absolute", "relative_trace", "relative_diag"],
    )
    parser.add_argument("--spd-eps-alpha", type=float, default=1e-2)
    parser.add_argument("--spd-eps-floor-mult", type=float, default=1e-6)
    parser.add_argument("--spd-eps-ceil-mult", type=float, default=1e-1)
    parser.add_argument("--clf", type=str, default="ridge")
    parser.add_argument(
        "--trial-protocol",
        type=str,
        default="session_holdout",
        choices=["session_holdout", "loso_subject"],
    )
    parser.add_argument(
        "--raw-save-trial",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
    )
    parser.add_argument("--raw-mem-debug", type=int, default=0)
    parser.add_argument("--raw-mem-interval", type=int, default=0)
    parser.add_argument("--raw-filter-chunk", type=int, default=0)
    parser.add_argument("--raw-resample-chunk", type=int, default=0)
    parser.add_argument("--raw-cnt-subprocess", type=int, default=0)
    parser.add_argument("--out-prefix", type=str, default=None)
    args = parser.parse_args()

    if args.dataset != "seed1":
        raise ValueError("run_manifold_raw_subject supports seed1 only")

    rows = _load_manifest(args.raw_manifest)
    rows = [r for r in rows if str(r["subject"]) == str(args.subject)]
    if not rows:
        raise ValueError(f"No trials found for subject {args.subject}")
    if args.seed_raw_backend != "fif":
        raise ValueError(
            "manifold_raw_v1_frozen: CNT backend is not allowed. "
            "Use offline conversion to FIF and set --seed-raw-backend fif."
        )

    if args.out_prefix:
        out_prefix = f"{args.out_prefix}_sub{args.subject}"
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_prefix = os.path.join("logs", f"manifold_raw_v1_subproc_{args.subject}_{ts}")

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    if args.raw_cnt_subprocess:
        # Run one CNT per subprocess, collect features, then evaluate.
        cnt_paths = set()
        for r in rows:
            if args.seed_raw_backend == "fif" and r.get("out_path"):
                cnt_paths.add(r["out_path"])
            else:
                cnt_paths.add(r.get("source_cnt_path") or r.get("cnt_path"))
        cnt_paths = sorted(cnt_paths)
        features = []
        labels = []
        meta_trials = []
        for cnt_path in cnt_paths:
            cnt_prefix = f"{out_prefix}_cnt{os.path.basename(cnt_path).split('.')[0]}"
            cmd = [
                sys.executable,
                os.path.join(ROOT, "scripts", "run_manifold_raw_cnt.py"),
                "--dataset",
                "seed1",
                "--raw-manifest",
                args.raw_manifest,
                "--cnt-path",
                cnt_path,
            "--seed-raw-root",
            args.seed_raw_root,
            "--seed-raw-backend",
            args.seed_raw_backend,
                "--raw-window-sec",
                str(args.raw_window_sec),
                "--raw-window-hop-sec",
                str(args.raw_window_hop_sec),
                "--raw-resample-fs",
                str(args.raw_resample_fs),
                "--raw-resample-chunk",
                str(args.raw_resample_chunk),
                "--raw-bands",
                args.raw_bands,
                "--raw-time-unit",
                args.raw_time_unit,
                "--raw-trial-offset-sec",
                str(args.raw_trial_offset_sec),
                "--raw-cov",
                args.raw_cov,
                "--raw-logmap-eps",
                str(args.raw_logmap_eps),
                "--raw-seq-save-format",
                args.raw_seq_save_format,
                "--spd-eps",
                str(args.spd_eps),
                "--spd-eps-mode",
                args.spd_eps_mode,
                "--spd-eps-alpha",
                str(args.spd_eps_alpha),
                "--spd-eps-floor-mult",
                str(args.spd_eps_floor_mult),
                "--spd-eps-ceil-mult",
                str(args.spd_eps_ceil_mult),
                "--raw-filter-chunk",
                str(args.raw_filter_chunk),
                "--clf",
                args.clf,
                "--trial-protocol",
                args.trial_protocol,
                "--raw-save-trial",
                "yes",
                "--raw-mem-debug",
                str(args.raw_mem_debug),
                "--raw-mem-interval",
                str(args.raw_mem_interval),
                "--out-prefix",
                cnt_prefix,
            ]
            log_path = f"{cnt_prefix}.log"
            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.run(cmd, stdout=log_file, stderr=log_file)
            if proc.returncode != 0:
                raise RuntimeError(f"CNT worker failed: {cnt_path} (see {log_path})")

            x_path = f"{cnt_prefix}_X_trial.npy"
            y_path = f"{cnt_prefix}_y_trial.npy"
            meta_path = f"{cnt_prefix}_meta.json"
            if not (os.path.isfile(x_path) and os.path.isfile(y_path) and os.path.isfile(meta_path)):
                raise FileNotFoundError(f"Missing CNT artifacts for {cnt_path}")

            import numpy as np

            X_cnt = np.load(x_path)
            y_cnt = np.load(y_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_cnt = json.load(f)
            features.append(X_cnt)
            labels.append(y_cnt)
            meta_trials.extend(meta_cnt.get("trials", []))

            if args.raw_save_trial == "no":
                for p in [
                    x_path,
                    y_path,
                    meta_path,
                    f"{cnt_prefix}_report.json",
                    f"{cnt_prefix}_report.csv",
                    f"{cnt_prefix}_manifest.json",
                ]:
                    if os.path.isfile(p):
                        os.remove(p)

        import numpy as np
        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.svm import LinearSVC

        X_trials = np.concatenate(features, axis=0)
        y_trials = np.concatenate(labels, axis=0)

        tmp_runner = ManifoldRawV1Runner(
            raw_manifest=None,
            seed_raw_root=args.seed_raw_root,
            trial_protocol=args.trial_protocol,
        )
        splits = tmp_runner._build_splits(meta_trials)

        split_rows = []
        acc_list = []
        f1_list = []
        for split in splits:
            train_idx = np.asarray(split["train_idx"], dtype=np.int64)
            test_idx = np.asarray(split["test_idx"], dtype=np.int64)
            X_tr, y_tr = X_trials[train_idx], y_trials[train_idx]
            X_te, y_te = X_trials[test_idx], y_trials[test_idx]

            if args.clf.lower() == "ridge":
                clf = RidgeClassifier()
            elif args.clf.lower() == "svm_linear":
                clf = LinearSVC()
            else:
                raise ValueError(f"Unknown clf: {args.clf}")

            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            acc = float(accuracy_score(y_te, pred))
            f1 = float(f1_score(y_te, pred, average="macro"))
            acc_list.append(acc)
            f1_list.append(f1)
            split_rows.append(
                {
                    "name": split["name"],
                    "subject": split["subject"],
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "acc": acc,
                    "macro_f1": f1,
                }
            )

        acc_mean = float(np.mean(acc_list)) if acc_list else 0.0
        acc_std = float(np.std(acc_list)) if acc_list else 0.0
        f1_mean = float(np.mean(f1_list)) if f1_list else 0.0
        f1_std = float(np.std(f1_list)) if f1_list else 0.0

        report = {
            "protocol": args.trial_protocol,
            "window_sec": args.raw_window_sec,
            "hop_sec": args.raw_window_hop_sec,
            "resample_fs": args.raw_resample_fs,
            "cov": args.raw_cov,
            "logmap_eps": args.raw_logmap_eps,
            "clf": args.clf,
            "raw_backend": args.seed_raw_backend,
            "n_splits": len(split_rows),
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "splits": split_rows,
        }

        report_json = f"{out_prefix}_report.json"
        report_csv = f"{out_prefix}_report.csv"
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(report_csv, "w", encoding="utf-8") as f:
            f.write("name,subject,n_train,n_test,acc,macro_f1\n")
            for row in split_rows:
                f.write(
                    f"{row['name']},{row['subject']},{row['n_train']},"
                    f"{row['n_test']},{row['acc']:.6f},{row['macro_f1']:.6f}\n"
                )
    else:
        manifest_path = f"{out_prefix}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    runner = ManifoldRawV1Runner(
        raw_manifest=manifest_path,
        seed_raw_root=args.seed_raw_root,
        raw_window_sec=args.raw_window_sec,
        raw_window_hop_sec=args.raw_window_hop_sec,
        raw_resample_fs=args.raw_resample_fs,
        raw_bands=args.raw_bands,
        raw_time_unit=args.raw_time_unit or None,
        raw_trial_offset_sec=args.raw_trial_offset_sec,
        raw_cov=args.raw_cov,
        raw_logmap_eps=args.raw_logmap_eps,
        raw_seq_save_format=args.raw_seq_save_format,
        spd_eps=args.spd_eps,
        spd_eps_mode=args.spd_eps_mode,
        spd_eps_alpha=args.spd_eps_alpha,
        spd_eps_floor_mult=args.spd_eps_floor_mult,
        spd_eps_ceil_mult=args.spd_eps_ceil_mult,
        clf=args.clf,
        trial_protocol=args.trial_protocol,
        out_prefix=out_prefix,
        raw_chunk_by="none",
        raw_max_subjects=0,
        raw_subject_list=None,
        raw_mem_debug=args.raw_mem_debug,
        raw_mem_interval=args.raw_mem_interval,
        raw_save_trial=args.raw_save_trial,
        raw_filter_chunk=args.raw_filter_chunk,
        raw_resample_chunk=args.raw_resample_chunk,
        raw_backend=args.seed_raw_backend,
    )
    runner.run()


if __name__ == "__main__":
    main()
