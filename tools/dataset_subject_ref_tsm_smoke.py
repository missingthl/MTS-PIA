import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.manifold_streaming_riemann import RiemannianUtils


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
    raise ValueError("manifest must be dict with file_paths/trials")


def _resolve_path(path: str, manifest_path: Path) -> Path:
    p = Path(path)
    if p.is_absolute() and p.is_file():
        return p
    if p.is_file():
        return p
    candidate = manifest_path.parent / p
    if candidate.is_file():
        return candidate
    return p


def _load_cov(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"expected [T,5,62,62], got {arr.shape}")
    if arr.shape[1:] == (5, 62, 62):
        return arr
    if arr.shape[-1] == 5:
        return arr.transpose(0, 3, 1, 2)
    raise ValueError(f"unexpected array shape {arr.shape}")


def _split_by_subject(trials, split_seed: int, val_ratio: float):
    groups = {}
    for idx, row in enumerate(trials):
        key = str(row.get("subject"))
        groups.setdefault(key, []).append(idx)
    if not groups:
        raise ValueError("no subject groups resolved")
    rng = np.random.default_rng(int(split_seed))
    subjects = list(groups.keys())
    rng.shuffle(subjects)
    val_size = int(len(subjects) * val_ratio)
    if val_ratio > 0.0 and val_size == 0 and len(subjects) > 1:
        val_size = 1
    if val_size >= len(subjects):
        val_size = len(subjects) - 1 if len(subjects) > 1 else 0
    val_subjects = subjects[:val_size]
    train_subjects = subjects[val_size:]
    train_idx = [i for s in train_subjects for i in groups[s]]
    val_idx = [i for s in val_subjects for i in groups[s]]
    return train_idx, val_idx


def _offdiag_mean_abs(mats: np.ndarray) -> np.ndarray:
    diag = np.diagonal(mats, axis1=1, axis2=2)
    abs_sum = np.abs(mats).sum(axis=(1, 2))
    abs_diag = np.abs(diag).sum(axis=1)
    return (abs_sum - abs_diag) / float(mats.shape[1] * mats.shape[2] - mats.shape[1])


def _features_tsm_stats(arr: np.ndarray) -> np.ndarray:
    feats = []
    for b in range(arr.shape[1]):
        mats = arr[:, b]
        diag = np.diagonal(mats, axis1=1, axis2=2)
        diag_mean_t = diag.mean(axis=1)
        offdiag_mean_abs_t = _offdiag_mean_abs(mats)
        fro_norm_t = np.linalg.norm(mats.reshape(mats.shape[0], -1), axis=1)
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
    return np.asarray(feats, dtype=np.float32)


def _run_classifier(X_train, y_train, X_val, y_val):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs"),
    )
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    train_acc = float(accuracy_score(y_train, train_pred))
    train_f1 = float(f1_score(y_train, train_pred, average="macro"))
    val_acc = float(accuracy_score(y_val, val_pred))
    val_f1 = float(f1_score(y_val, val_pred, average="macro"))
    return {
        "train_acc": train_acc,
        "train_macro_f1": train_f1,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
    }


def _compute_features(
    *,
    file_paths,
    trials,
    manifest_path: Path,
    ref_means: dict,
    max_trials_per_subject: int,
    max_windows_per_trial: int,
    seed: int,
):
    rng = np.random.default_rng(int(seed))
    by_subject = {}
    for idx, row in enumerate(trials):
        by_subject.setdefault(str(row.get("subject")), []).append(idx)

    indices = []
    for subj, idxs in by_subject.items():
        if max_trials_per_subject > 0 and len(idxs) > max_trials_per_subject:
            indices.extend(rng.choice(idxs, size=max_trials_per_subject, replace=False).tolist())
        else:
            indices.extend(idxs)
    indices = sorted(indices)

    X = []
    y = []
    for idx in indices:
        row = trials[idx]
        cov_path = _resolve_path(file_paths[idx], manifest_path)
        cov_seq = _load_cov(cov_path)
        if max_windows_per_trial > 0 and cov_seq.shape[0] > max_windows_per_trial:
            win_idx = rng.choice(
                cov_seq.shape[0],
                size=int(max_windows_per_trial),
                replace=False,
            )
            win_idx = np.sort(win_idx)
            cov_seq = cov_seq[win_idx]
        subj = str(row.get("subject"))
        ref = ref_means[subj]
        bands_out = []
        for b in range(5):
            covs = torch.from_numpy(np.asarray(cov_seq[:, b], dtype=np.float32))
            tsm = RiemannianUtils.tangent_space_mapping(covs, ref[b])
            bands_out.append(tsm.cpu().numpy())
        tsm_seq = np.stack(bands_out, axis=1)
        X.append(_features_tsm_stats(tsm_seq))
        y.append(int(row["label"]))
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64), indices


def main():
    parser = argparse.ArgumentParser(description="Subjectwise ref-mean TSM baseline.")
    parser.add_argument(
        "--manifest_path",
        default="logs/seed1_tsm_cov_spd_full_rel_seq_manifest.json",
        help="manifest path",
    )
    parser.add_argument(
        "--ref_mean_global",
        default="logs/seed1_tsm_cov_spd_full_rel_ref_mean.pt",
        help="global ref mean path",
    )
    parser.add_argument(
        "--ref_mean_subjectwise",
        default="logs/ref_mean_subjectwise.pt",
        help="subjectwise ref mean dict",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_trials_per_subject", type=int, default=0)
    parser.add_argument("--max_windows_per_trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="logs/subjectwise_tsm_baseline.json",
        help="output report path",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    file_paths, trials, labels = _load_manifest(manifest_path)
    if trials is None:
        raise ValueError("manifest missing trials")
    if labels is None:
        raise ValueError("manifest missing labels")

    ref_global = torch.load(args.ref_mean_global, map_location="cpu")
    if isinstance(ref_global, dict):
        if "ref_mean" in ref_global:
            ref_global = ref_global["ref_mean"]
        elif "mean" in ref_global:
            ref_global = ref_global["mean"]
    ref_global = torch.as_tensor(ref_global, dtype=torch.float32)
    if tuple(ref_global.shape) != (5, 62, 62):
        raise ValueError(f"global ref_mean shape mismatch: {tuple(ref_global.shape)}")

    ref_subject = torch.load(args.ref_mean_subjectwise, map_location="cpu")
    if not isinstance(ref_subject, dict):
        raise ValueError("subjectwise ref mean must be a dict")

    global_dict = {str(row.get("subject")): ref_global for row in trials}
    subject_dict = {str(k): torch.as_tensor(v, dtype=torch.float32) for k, v in ref_subject.items()}

    X_global, y_global, idx_global = _compute_features(
        file_paths=file_paths,
        trials=trials,
        manifest_path=manifest_path,
        ref_means=global_dict,
        max_trials_per_subject=int(args.max_trials_per_subject),
        max_windows_per_trial=int(args.max_windows_per_trial),
        seed=int(args.seed),
    )
    X_subject, y_subject, idx_subject = _compute_features(
        file_paths=file_paths,
        trials=trials,
        manifest_path=manifest_path,
        ref_means=subject_dict,
        max_trials_per_subject=int(args.max_trials_per_subject),
        max_windows_per_trial=int(args.max_windows_per_trial),
        seed=int(args.seed),
    )

    train_idx, val_idx = _split_by_subject(trials, args.split_seed, args.val_ratio)
    train_idx = [i for i in train_idx if i in idx_global]
    val_idx = [i for i in val_idx if i in idx_global]
    idx_to_pos = {idx: pos for pos, idx in enumerate(idx_global)}
    train_pos = [idx_to_pos[i] for i in train_idx]
    val_pos = [idx_to_pos[i] for i in val_idx]

    metrics_global = _run_classifier(
        X_global[train_pos], y_global[train_pos], X_global[val_pos], y_global[val_pos]
    )
    metrics_subject = _run_classifier(
        X_subject[train_pos], y_subject[train_pos], X_subject[val_pos], y_subject[val_pos]
    )

    delta = metrics_subject["val_acc"] - metrics_global["val_acc"]
    report = {
        "manifest_path": str(manifest_path),
        "ref_mean_global": str(args.ref_mean_global),
        "ref_mean_subjectwise": str(args.ref_mean_subjectwise),
        "split_by": "subject",
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "max_trials_per_subject": int(args.max_trials_per_subject),
        "max_windows_per_trial": int(args.max_windows_per_trial),
        "global_metrics": metrics_global,
        "subjectwise_metrics": metrics_subject,
        "delta_val_acc": float(delta),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(
        f"[subjectwise] global_val_acc={metrics_global['val_acc']:.4f} "
        f"global_val_macro_f1={metrics_global['val_macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"[subjectwise] subject_val_acc={metrics_subject['val_acc']:.4f} "
        f"subject_val_macro_f1={metrics_subject['val_macro_f1']:.4f}",
        flush=True,
    )
    print(f"[subjectwise] delta_val_acc={delta:.4f}", flush=True)
    print(f"[subjectwise] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
