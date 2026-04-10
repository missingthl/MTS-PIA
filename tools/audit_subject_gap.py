import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _split_by_index(total: int, split_seed: int, val_ratio: float):
    rng = np.random.default_rng(int(split_seed))
    indices = np.arange(total)
    rng.shuffle(indices)
    val_size = int(total * val_ratio)
    if val_ratio > 0.0 and val_size == 0 and total > 1:
        val_size = 1
    if val_size >= total:
        val_size = total - 1 if total > 1 else 0
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    return train_idx, val_idx


def _offdiag_mean_abs(mats: np.ndarray) -> np.ndarray:
    diag = np.diagonal(mats, axis1=1, axis2=2)
    abs_sum = np.abs(mats).sum(axis=(1, 2))
    abs_diag = np.abs(diag).sum(axis=1)
    return (abs_sum - abs_diag) / float(mats.shape[1] * mats.shape[2] - mats.shape[1])


def _features_tsm_stats(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"expected [T,5,62,62], got {arr.shape}")
    if arr.shape[1:] == (5, 62, 62):
        data = arr
    elif arr.shape[-1] == 5:
        data = arr.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"unexpected array shape {arr.shape}")
    feats = []
    for b in range(data.shape[1]):
        mats = data[:, b]
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


def _features_tsm_flat_small(arr: np.ndarray, topk: int = 8) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"expected [T,5,62,62], got {arr.shape}")
    if arr.shape[1:] == (5, 62, 62):
        data = arr
    elif arr.shape[-1] == 5:
        data = arr.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"unexpected array shape {arr.shape}")
    feats = []
    for b in range(data.shape[1]):
        mats = data[:, b]
        mean_mat = mats.mean(axis=0)
        mean_mat = 0.5 * (mean_mat + mean_mat.T)
        eigvals = np.linalg.eigvalsh(mean_mat)
        eigvals = eigvals[::-1][:topk]
        diag_mean = float(np.mean(np.diag(mean_mat)))
        offdiag_mean_abs = float(np.mean(np.abs(mean_mat - np.diag(np.diag(mean_mat)))))
        feats.extend([diag_mean, offdiag_mean_abs])
        feats.extend([float(x) for x in eigvals])
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


def main():
    parser = argparse.ArgumentParser(description="Audit emotion vs subject separability.")
    parser.add_argument(
        "--manifest_path",
        default="logs/seed1_tsm_cov_spd_full_rel_seq_manifest.json",
        help="TSM manifest path",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--feature_mode",
        default="tsm_stats",
        choices=["tsm_stats", "tsm_flat_small"],
        help="feature mode",
    )
    parser.add_argument("--max_trials_per_subject", type=int, default=50)
    parser.add_argument(
        "--out",
        default="logs/audit_subject_gap.json",
        help="output JSON report",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    file_paths, trials, labels = _load_manifest(manifest_path)
    if trials is None:
        raise ValueError("manifest missing trials for subject split")
    if labels is None:
        raise ValueError("manifest missing labels")

    rng = random.Random(int(args.split_seed))
    by_subject = {}
    for idx, row in enumerate(trials):
        by_subject.setdefault(str(row.get("subject")), []).append(idx)

    indices = []
    max_trials = int(args.max_trials_per_subject)
    for subj, idxs in by_subject.items():
        if max_trials > 0 and len(idxs) > max_trials:
            indices.extend(rng.sample(idxs, max_trials))
        else:
            indices.extend(idxs)
    indices = sorted(indices)

    file_paths = [file_paths[i] for i in indices]
    trials = [trials[i] for i in indices]
    labels = [labels[i] for i in indices]

    features = []
    for path in file_paths:
        npy_path = _resolve_path(path, manifest_path)
        arr = np.load(npy_path, mmap_mode="r")
        if args.feature_mode == "tsm_stats":
            feat = _features_tsm_stats(arr)
        else:
            feat = _features_tsm_flat_small(arr)
        features.append(feat)
    X = np.stack(features, axis=0)
    y_emotion = np.asarray(labels, dtype=np.int64)

    subj_labels = [str(row.get("subject")) for row in trials]
    subj_to_id = {s: i for i, s in enumerate(sorted(set(subj_labels)))}
    y_subject = np.asarray([subj_to_id[s] for s in subj_labels], dtype=np.int64)

    emo_train_idx, emo_val_idx = _split_by_subject(trials, args.split_seed, args.val_ratio)
    X_emo_train, X_emo_val = X[emo_train_idx], X[emo_val_idx]
    y_emo_train, y_emo_val = y_emotion[emo_train_idx], y_emotion[emo_val_idx]
    emotion_metrics = _run_classifier(X_emo_train, y_emo_train, X_emo_val, y_emo_val)

    subj_train_idx, subj_val_idx = _split_by_index(len(trials), args.split_seed, args.val_ratio)
    X_sub_train, X_sub_val = X[subj_train_idx], X[subj_val_idx]
    y_sub_train, y_sub_val = y_subject[subj_train_idx], y_subject[subj_val_idx]
    subject_metrics = _run_classifier(X_sub_train, y_sub_train, X_sub_val, y_sub_val)

    gap_ratio = subject_metrics["val_acc"] / max(emotion_metrics["val_acc"], 1e-12)
    report = {
        "manifest_path": str(manifest_path),
        "feature_mode": args.feature_mode,
        "emotion_split_by": "subject",
        "subject_split_by": "index",
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "max_trials_per_subject": int(args.max_trials_per_subject),
        "emotion_metrics": emotion_metrics,
        "subject_metrics": subject_metrics,
        "gap_ratio": float(gap_ratio),
        "subject_count": int(len(subj_to_id)),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(
        f"[gap] emotion_val_acc={emotion_metrics['val_acc']:.4f} "
        f"emotion_val_macro_f1={emotion_metrics['val_macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"[gap] subject_val_acc={subject_metrics['val_acc']:.4f} "
        f"subject_val_macro_f1={subject_metrics['val_macro_f1']:.4f}",
        flush=True,
    )
    print(f"[gap] gap_ratio={gap_ratio:.4f}", flush=True)
    print(f"[gap] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
