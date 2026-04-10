import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.manifold_streaming_riemann import RiemannianUtils
from manifold_raw.features import vec_utri


def _resolve_paths(paths, base_dir: Path):
    resolved = []
    for p in paths:
        p = str(p)
        cand = base_dir / p
        if cand.exists():
            resolved.append(str(cand))
        elif (base_dir.parent / p).exists():
            resolved.append(str(base_dir.parent / p))
        else:
            resolved.append(p)
    return resolved


def _load_manifest(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        file_paths = data.get("file_paths") or data.get("paths") or data.get("files")
        if file_paths is None:
            raise ValueError("manifest missing file_paths")
        trials = data.get("trials")
        labels = None
        if "labels" in data:
            labels = [int(v) for v in data["labels"]]
        elif isinstance(trials, list):
            labels = [int(t["label"]) for t in trials]
        base_dir = path.parent
        return _resolve_paths(file_paths, base_dir), trials, labels
    if isinstance(data, list):
        if not data:
            raise ValueError("manifest list is empty")
        if isinstance(data[0], dict):
            if "file_path" in data[0]:
                file_paths = [item["file_path"] for item in data]
            elif "path" in data[0]:
                file_paths = [item["path"] for item in data]
            else:
                raise ValueError("manifest list entries missing file_path/path")
            labels = None
            if "label" in data[0]:
                labels = [int(item["label"]) for item in data]
            base_dir = path.parent
            return _resolve_paths(file_paths, base_dir), data, labels
    raise ValueError("unsupported manifest format")


def _split_groups(trials, split_by: str, split_seed: int, val_ratio: float):
    if trials is None:
        raise ValueError("manifest missing trials for group split")
    groups = {}
    for idx, row in enumerate(trials):
        if split_by == "subject":
            key = str(row.get("subject"))
        else:
            key = row.get("trial_id")
            if not key:
                subj = row.get("subject")
                sess = row.get("session")
                tr = row.get("trial")
                key = f"{subj}_s{sess}_t{tr}"
        groups.setdefault(key, []).append(idx)
    if not groups:
        raise ValueError("no groups resolved for split")
    rng = np.random.default_rng(int(split_seed))
    group_list = list(groups.keys())
    rng.shuffle(group_list)
    val_size = int(len(group_list) * val_ratio)
    if val_ratio > 0.0 and val_size == 0 and len(group_list) > 1:
        val_size = 1
    if val_size >= len(group_list):
        val_size = len(group_list) - 1 if len(group_list) > 1 else 0
    val_groups = group_list[:val_size]
    train_groups = group_list[val_size:]
    train_idx = [i for g in train_groups for i in groups[g]]
    val_idx = [i for g in val_groups for i in groups[g]]
    return train_idx, val_idx, val_groups


def _load_ref_mean(ref_mean_path: str) -> torch.Tensor:
    if not ref_mean_path:
        raise ValueError("ref_mean_path is required for tsm_* modes")
    ref_mean = torch.load(ref_mean_path, map_location="cpu")
    if isinstance(ref_mean, dict):
        if "ref_mean" in ref_mean:
            ref_mean = ref_mean["ref_mean"]
        elif "mean" in ref_mean:
            ref_mean = ref_mean["mean"]
        else:
            raise ValueError("ref_mean dict missing ref_mean/mean")
    ref_mean = torch.as_tensor(ref_mean, dtype=torch.float32)
    if tuple(ref_mean.shape) != (5, 62, 62):
        raise ValueError(f"unexpected ref_mean shape: {tuple(ref_mean.shape)}")
    return ref_mean


def _normalize_cov(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"expected 4D array, got {arr.shape}")
    if arr.shape[1] == 62 and arr.shape[2] == 62:
        arr = np.transpose(arr, (0, 3, 1, 2))
    if arr.shape[1:] != (5, 62, 62):
        raise ValueError(f"unexpected array shape after transpose: {arr.shape}")
    return arr


def _feat_diag_all(arr: np.ndarray, eps_diag: float) -> np.ndarray:
    diag = np.diagonal(arr, axis1=2, axis2=3)
    diag = np.maximum(diag, eps_diag)
    feat_t = np.log(diag)
    feat = feat_t.mean(axis=0)
    return feat.reshape(-1)


def _feat_diag_gamma(arr: np.ndarray, gamma_index: int, eps_diag: float) -> np.ndarray:
    diag = np.diagonal(arr, axis1=2, axis2=3)
    diag = np.maximum(diag, eps_diag)
    feat_t = np.log(diag[:, gamma_index, :])
    return feat_t.mean(axis=0).reshape(-1)


def _feat_tsm_gamma(arr: np.ndarray, ref_mean: torch.Tensor, gamma_index: int) -> np.ndarray:
    cov = arr[:, gamma_index, :, :]
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    cov_t = torch.as_tensor(cov, dtype=torch.float32)
    ref_gamma = ref_mean[gamma_index]
    with torch.no_grad():
        tsm = RiemannianUtils.tangent_space_mapping(cov_t, ref_gamma)
    tsm_np = tsm.cpu().numpy()
    idx = np.triu_indices(tsm_np.shape[-1])
    vec = tsm_np[:, idx[0], idx[1]]
    return vec.mean(axis=0)


def _feat_tsm_all(arr: np.ndarray, ref_mean: torch.Tensor) -> np.ndarray:
    feats = []
    for b in range(arr.shape[1]):
        cov = arr[:, b, :, :]
        cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
        cov_t = torch.as_tensor(cov, dtype=torch.float32)
        with torch.no_grad():
            tsm = RiemannianUtils.tangent_space_mapping(cov_t, ref_mean[b])
        tsm_np = tsm.cpu().numpy()
        idx = np.triu_indices(tsm_np.shape[-1])
        vec = tsm_np[:, idx[0], idx[1]]
        feats.append(vec.mean(axis=0))
    return np.concatenate(feats, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--ref-mean-path", type=str, default="")
    parser.add_argument("--split-by", type=str, default="subject")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--gamma-index", type=int, default=4)
    parser.add_argument("--eps-diag", type=float, default=1e-12)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    file_paths, trials, labels = _load_manifest(manifest_path)
    if trials is None or labels is None:
        raise ValueError("manifest must include trials with labels for group split")
    if len(file_paths) != len(labels) or len(trials) != len(labels):
        raise ValueError("manifest length mismatch")
    if args.gamma_index < 0 or args.gamma_index > 4:
        raise ValueError("gamma_index must be in [0,4]")

    ref_mean = None
    if args.mode.startswith("tsm"):
        ref_mean = _load_ref_mean(args.ref_mean_path)

    train_idx, val_idx, val_groups = _split_groups(
        trials, args.split_by, args.split_seed, args.val_ratio
    )

    features = []
    subjects = []
    for idx, path in enumerate(file_paths):
        arr = _normalize_cov(np.load(path))
        if args.mode == "diag_all":
            feat = _feat_diag_all(arr, args.eps_diag)
        elif args.mode == "diag_gamma":
            feat = _feat_diag_gamma(arr, args.gamma_index, args.eps_diag)
        elif args.mode == "tsm_gamma":
            feat = _feat_tsm_gamma(arr, ref_mean, args.gamma_index)
        elif args.mode == "tsm_all":
            feat = _feat_tsm_all(arr, ref_mean)
        else:
            raise ValueError(f"unsupported mode: {args.mode}")
        if not np.all(np.isfinite(feat)):
            raise ValueError(f"non-finite feature at idx={idx} path={path}")
        features.append(feat.astype(np.float32, copy=False))
        subjects.append(str(trials[idx].get("subject")))

    X = np.stack(features, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs"),
    )
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)

    train_acc = float(accuracy_score(y_train, pred_train))
    train_f1 = float(f1_score(y_train, pred_train, average="macro"))
    val_acc = float(accuracy_score(y_val, pred_val))
    val_f1 = float(f1_score(y_val, pred_val, average="macro"))
    val_f1_per_class = f1_score(y_val, pred_val, average=None).tolist()
    cm = confusion_matrix(y_val, pred_val).tolist()

    val_subjects = sorted({subjects[i] for i in val_idx})
    val_subject_acc = None
    if val_subjects:
        subj_accs = []
        for subj in val_subjects:
            positions = [pos for pos, idx in enumerate(val_idx) if subjects[idx] == subj]
            if positions:
                subj_accs.append(float(accuracy_score(y_val[positions], pred_val[positions])))
        if subj_accs:
            val_subject_acc = float(np.mean(subj_accs))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "manifest_path": str(manifest_path),
        "ref_mean_path": args.ref_mean_path if args.ref_mean_path else None,
        "mode": args.mode,
        "sample_unit": "trial-level",
        "pooling_method": "mean_over_time (mask-aware by trial length)",
        "eps_diag": float(args.eps_diag),
        "gamma_index": int(args.gamma_index),
        "split_by": args.split_by,
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "val_subjects_count": len(val_subjects),
        "val_subjects": val_subjects,
    }
    metrics = {
        "train_acc": train_acc,
        "train_macro_f1": train_f1,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "val_per_class_f1": val_f1_per_class,
        "confusion_matrix_val": cm,
        "val_subject_acc": val_subject_acc,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(
        f"[ablation] mode={args.mode} train_acc={train_acc:.4f} "
        f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
    )
    print(f"[ablation] out_dir={out_dir}")


if __name__ == "__main__":
    main()
