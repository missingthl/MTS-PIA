from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_tangent_audit import (  # noqa: E402
    build_alignment_rows,
    estimate_local_tangent_spaces,
    summarize_candidate_audit,
)
from core.pia import build_pca_direction_bank, build_zpia_direction_bank  # noqa: E402
from run_act_pilot import _build_trial_records  # noqa: E402
from utils.datasets import load_trials_for_dataset, make_trial_split  # noqa: E402


DEFAULT_DATASETS = (
    "atrialfibrillation,ering,handmovementdirection,handwriting,"
    "japanesevowels,natops,racketsports"
)
DEFAULT_METHODS = "csta_topk_uniform_top5,csta_top1_current,random_cov_state,pca_cov_state"
DEFAULT_SELECTOR_ROOT = PROJECT_ROOT / "results" / "csta_selector_ablation_v1" / "resnet1d_s123"


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_tangent_dim(value: str) -> int | str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    return int(text)


def _load_train_states(dataset: str, seed: int, *, val_ratio: float):
    all_trials = load_trials_for_dataset(dataset)
    train_trials, _test_trials, _val_trials = make_trial_split(all_trials, seed=seed, val_ratio=val_ratio)
    train_records, _mean_log = _build_trial_records(train_trials)
    if not train_records:
        raise RuntimeError(f"No train records loaded for {dataset} seed={seed}")
    Z = np.stack([r.z for r in train_records], axis=0).astype(np.float64)
    y = np.asarray([r.y for r in train_records], dtype=np.int64)
    return train_records, Z, y


def _candidate_path_from_summary(summary_path: Path, dataset: str, seed: int, method: str) -> Path | None:
    if not summary_path.is_file():
        return None
    df = pd.read_csv(summary_path)
    required = {"dataset", "seed", "method", "candidate_audit_path"}
    if not required.issubset(df.columns):
        return None
    hit = df[
        (df["dataset"].astype(str) == str(dataset))
        & (df["seed"].astype(int) == int(seed))
        & (df["method"].astype(str) == str(method))
    ]
    if hit.empty:
        return None
    raw = hit.iloc[0].get("candidate_audit_path", "")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.is_file() else None


def _candidate_path_by_glob(selector_root: Path, dataset: str, seed: int, method: str) -> Path | None:
    byarm = selector_root.parent / f"{selector_root.name}_byarm"
    patterns = [
        byarm / method / "_csta_runs" / method / dataset / f"s{seed}" / "candidate_audit" / f"{dataset}_s{seed}_{method}_candidate_audit.csv.gz",
        selector_root / "_csta_runs" / method / dataset / f"s{seed}" / "candidate_audit" / f"{dataset}_s{seed}_{method}_candidate_audit.csv.gz",
    ]
    for path in patterns:
        if path.is_file():
            return path
    return None


def _load_actual_candidate_audit(selector_root: Path, dataset: str, seed: int, method: str) -> tuple[pd.DataFrame | None, str]:
    if method not in {"csta_topk_uniform_top5", "csta_top1_current"}:
        return None, ""
    summary_path = selector_root / "per_seed_selector_with_refs.csv"
    path = _candidate_path_from_summary(summary_path, dataset, seed, method)
    if path is None:
        path = _candidate_path_by_glob(selector_root, dataset, seed, method)
    if path is None:
        return None, ""
    df = pd.read_csv(path)
    return df, str(path)


def run_one(dataset: str, seed: int, args) -> pd.DataFrame:
    train_records, Z, y = _load_train_states(dataset, seed, val_ratio=float(args.val_ratio))
    tangent_dim = _parse_tangent_dim(args.tangent_dim)
    tangent = estimate_local_tangent_spaces(
        Z,
        y,
        k_neighbors=int(args.k_neighbors),
        tangent_dim=tangent_dim,
        class_conditioned=True,
        distance="euclidean",
        min_neighbors=int(args.min_neighbors),
        explained_var_threshold=float(args.explained_var_threshold),
        max_tangent_dim=int(args.max_tangent_dim),
    )

    zpia_bank, zpia_meta = build_zpia_direction_bank(
        Z,
        k_dir=int(args.k_dir),
        seed=int(seed),
        telm2_n_iters=int(args.telm2_n_iters),
        telm2_c_repr=float(args.telm2_c_repr),
        telm2_activation=str(args.telm2_activation),
        telm2_bias_update_mode=str(args.telm2_bias_update_mode),
    )
    pca_bank, pca_meta = build_pca_direction_bank(Z, k_dir=int(args.k_dir), seed=int(seed))

    rows = []
    actual_paths: dict[str, str] = {}
    for method in _parse_csv(args.methods):
        if method not in {"csta_topk_uniform_top5", "csta_top1_current", "random_cov_state", "pca_cov_state"}:
            raise ValueError(f"Unsupported local tangent audit method: {method}")
        actual_rows = None
        actual_path = ""
        audit_source = "policy_replay"
        if method in {"csta_topk_uniform_top5", "csta_top1_current"}:
            actual_rows, actual_path = _load_actual_candidate_audit(Path(args.selector_root), dataset, seed, method)
            if actual_rows is not None and not actual_rows.empty:
                audit_source = "actual_candidate_audit"
                actual_paths[method] = actual_path
        rows.extend(
            build_alignment_rows(
                dataset=dataset,
                seed=seed,
                method=method,
                Z=Z,
                y=y,
                tangent=tangent,
                direction_bank=zpia_bank,
                pca_bank=pca_bank,
                multiplier=int(args.multiplier),
                k_dir=int(args.k_dir),
                actual_candidate_rows=actual_rows,
                audit_source=audit_source,
            )
        )
    df = pd.DataFrame(rows)
    df["backbone"] = str(args.backbone)
    df["k_neighbors"] = int(args.k_neighbors)
    df["tangent_dim_policy"] = str(args.tangent_dim)
    df["explained_var_threshold"] = float(args.explained_var_threshold)
    df["max_tangent_dim"] = int(args.max_tangent_dim)
    df["train_split_only"] = True
    df["zpia_bank_source"] = str(zpia_meta.get("bank_source", "zpia_telm2"))
    df["pca_bank_source"] = str(pca_meta.get("bank_source", "pca"))
    df["n_train"] = int(len(train_records))
    if "actual_candidate_audit_path" not in df.columns:
        df["actual_candidate_audit_path"] = ""
    for method, actual_path in actual_paths.items():
        df.loc[df["method"] == method, "actual_candidate_audit_path"] = actual_path
    return df


def write_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "local_tangent_candidate_audit.csv.gz"
    df.to_csv(candidate_path, index=False, compression="gzip")

    summary = summarize_candidate_audit(df)
    for col in ["k_neighbors", "tangent_dim_policy", "explained_var_threshold", "max_tangent_dim"]:
        if col in df.columns and not summary.empty:
            summary[col] = df[col].iloc[0]
    summary_path = out_dir / "local_tangent_run_summary.csv"
    summary.to_csv(summary_path, index=False)

    json_rows = json.loads(summary.to_json(orient="records"))
    (out_dir / "local_tangent_run_summary.json").write_text(
        json.dumps(json_rows, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc local tangent alignment audit for CSTA/PIA.")
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--backbone", type=str, default="resnet1d")
    parser.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--min-neighbors", type=int, default=3)
    parser.add_argument("--tangent-dim", type=str, default="auto")
    parser.add_argument("--explained-var-threshold", type=float, default=0.90)
    parser.add_argument("--max-tangent-dim", type=int, default=10)
    parser.add_argument("--k-dir", type=int, default=10)
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--telm2-c-repr", type=float, default=10.0)
    parser.add_argument("--telm2-n-iters", type=int, default=50)
    parser.add_argument("--telm2-activation", type=str, default="sine", choices=["sine", "sigmoid", "none"])
    parser.add_argument("--telm2-bias-update-mode", type=str, default="none", choices=["none", "mean", "ema"])
    parser.add_argument("--device", type=str, default="cpu", help="Accepted for interface consistency; audit is CPU/numpy.")
    parser.add_argument("--selector-root", type=Path, default=DEFAULT_SELECTOR_ROOT)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "local_tangent_audit_v1" / "resnet1d_s123",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = _parse_csv(args.datasets)
    seeds = _parse_seeds(args.seeds)
    methods = _parse_csv(args.methods)
    print(
        json.dumps(
            {
                "datasets": datasets,
                "seeds": seeds,
                "methods": methods,
                "out_root": str(args.out_root),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    if args.dry_run:
        for dataset in datasets:
            for seed in seeds:
                print(args.out_root / dataset / f"s{seed}")
        return

    for dataset in datasets:
        for seed in seeds:
            df = run_one(dataset, seed, args)
            out_dir = Path(args.out_root) / dataset / f"s{seed}"
            write_outputs(df, out_dir)
            print(f"[local-tangent] wrote {out_dir} rows={len(df)}")


if __name__ == "__main__":
    main()
