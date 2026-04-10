import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.manifold_streaming_riemann import ManifoldStreamingDataset


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
        return list(file_paths), trials, labels
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
            return list(file_paths), None, labels
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
    return train_idx, val_idx


def _load_ref_mean(ref_mean_path: str) -> torch.Tensor:
    if not ref_mean_path:
        raise ValueError("ref_mean_path is required for TSM baseline")
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


def _feature_vector(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4 or arr.shape[1:4] != (5, 62, 62):
        raise ValueError(f"unexpected array shape {arr.shape}")
    t_len, bands, ch, _ = arr.shape
    feats = []
    for b in range(bands):
        mats = arr[:, b]
        diag = np.diagonal(mats, axis1=1, axis2=2)
        diag_mean_t = diag.mean(axis=1)

        abs_mats = np.abs(mats)
        abs_sum = abs_mats.sum(axis=(1, 2))
        abs_diag = np.abs(diag).sum(axis=1)
        offdiag_mean_abs_t = (abs_sum - abs_diag) / float(ch * ch - ch)

        fro_norm_t = np.linalg.norm(mats.reshape(t_len, -1), axis=1)

        feats.extend(
            [
                float(diag_mean_t.mean()),
                float(diag_mean_t.std()),
                float(offdiag_mean_abs_t.mean()),
                float(offdiag_mean_abs_t.std()),
                float(fro_norm_t.mean()),
                float(fro_norm_t.std()),
            ]
        )

        mean_mat = mats.mean(axis=0)
        mean_mat = 0.5 * (mean_mat + mean_mat.T)
        eigvals = np.linalg.eigvalsh(mean_mat)
        topk = eigvals[-8:][::-1]
        feats.extend([float(x) for x in topk])

    feats = np.asarray(feats, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def main():
    parser = argparse.ArgumentParser(description="TSM baseline with optional smoothing.")
    parser.add_argument(
        "--manifest-path",
        default="logs/seed1_tsm_cov_spd_full_rel_seq_manifest.json",
        help="manifest path",
    )
    parser.add_argument(
        "--ref-mean-path",
        default="logs/seed1_tsm_cov_spd_full_rel_ref_mean.pt",
        help="reference mean path for TSM",
    )
    parser.add_argument(
        "--split-by",
        default="subject",
        choices=["subject", "trial"],
        help="split by subject or trial",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument(
        "--tsm-smooth-mode",
        default="none",
        choices=["none", "ema", "kalman"],
        help="optional TSM temporal smoothing mode",
    )
    parser.add_argument("--tsm-ema-alpha", type=float, default=0.05)
    parser.add_argument("--tsm-kalman-qr", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        default="logs/baseline_tsm_smooth_subject.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    file_paths, trials, labels = _load_manifest(manifest_path)
    if labels is None:
        raise ValueError("manifest missing labels")
    total = len(labels)
    indices = list(range(total))
    if args.max_trials and args.max_trials > 0 and args.max_trials < total:
        rng = np.random.default_rng(int(args.split_seed))
        indices = rng.choice(indices, size=int(args.max_trials), replace=False).tolist()
    if trials is not None and indices != list(range(total)):
        trials = [trials[i] for i in indices]
    if file_paths is not None and indices != list(range(total)):
        file_paths = [file_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    train_idx, val_idx = _split_groups(
        trials, args.split_by, args.split_seed, args.val_ratio
    )

    ref_mean = _load_ref_mean(args.ref_mean_path)
    dataset = ManifoldStreamingDataset(
        str(manifest_path),
        reference_mean=ref_mean,
        tsm_smooth_mode=args.tsm_smooth_mode,
        tsm_ema_alpha=args.tsm_ema_alpha,
        tsm_kalman_qr=args.tsm_kalman_qr,
    )

    features = []
    lengths = []
    nan_inf_count = 0
    for idx in indices:
        proj_tensor, _label = dataset[idx]
        arr = proj_tensor.numpy()
        lengths.append(int(arr.shape[0]))
        feats = _feature_vector(arr)
        if not np.isfinite(feats).all():
            nan_inf_count += 1
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feats)

    X = np.stack(features, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
        ),
    )
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)

    train_acc = float(accuracy_score(y_train, train_pred))
    train_f1 = float(f1_score(y_train, train_pred, average="macro"))
    val_acc = float(accuracy_score(y_val, val_pred))
    val_f1 = float(f1_score(y_val, val_pred, average="macro"))

    length_arr = np.asarray(lengths, dtype=np.int64)
    report = {
        "manifest_path": str(manifest_path),
        "ref_mean_path": str(args.ref_mean_path),
        "split_by": args.split_by,
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "max_trials": int(args.max_trials),
        "tsm_smooth_mode": args.tsm_smooth_mode,
        "tsm_ema_alpha": float(args.tsm_ema_alpha),
        "tsm_kalman_qr": float(args.tsm_kalman_qr),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "train_acc": train_acc,
        "train_macro_f1": train_f1,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "valid_len_stats": {
            "min": int(length_arr.min()) if length_arr.size else 0,
            "p50": int(np.percentile(length_arr, 50)) if length_arr.size else 0,
            "p95": int(np.percentile(length_arr, 95)) if length_arr.size else 0,
            "max": int(length_arr.max()) if length_arr.size else 0,
        },
        "nan_inf_count": int(nan_inf_count),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(
        f"[baseline] train_acc={train_acc:.4f} train_macro_f1={train_f1:.4f} "
        f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}",
        flush=True,
    )
    print(f"[baseline] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
