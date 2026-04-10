import argparse
import json
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


def _feature_logvar(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"expected [T,5,62,62], got {arr.shape}")
    if arr.shape[1:] == (5, 62, 62):
        data = arr
    elif arr.shape[-1] == 5:
        data = arr.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"unexpected array shape {arr.shape}")

    eps = 1e-12
    feats = []
    for b in range(data.shape[1]):
        mats = data[:, b]
        diag = np.diagonal(mats, axis1=1, axis2=2)
        diag = np.clip(diag, eps, None)
        log_diag = np.log(diag)
        log_diag_mean_t = log_diag.mean(axis=1)

        trace = np.trace(mats, axis1=1, axis2=2) / float(mats.shape[1])
        trace = np.clip(trace, eps, None)
        log_trace_t = np.log(trace)

        feats.extend(
            [
                float(log_diag_mean_t.mean()),
                float(log_diag_mean_t.std()),
                float(log_trace_t.mean()),
                float(log_trace_t.std()),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Baseline using log-variance features.")
    parser.add_argument(
        "--manifest_path",
        default="logs/seed1_tsm_cov_spd_full_rel_seq_manifest.json",
        help="manifest path",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--out",
        default="logs/baseline_de_logvar_subject.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    file_paths, trials, labels = _load_manifest(manifest_path)
    if trials is None:
        raise ValueError("manifest missing trials for subject split")
    if labels is None:
        raise ValueError("manifest missing labels")

    features = []
    for path in file_paths:
        npy_path = _resolve_path(path, manifest_path)
        arr = np.load(npy_path, mmap_mode="r")
        features.append(_feature_logvar(arr))

    X = np.stack(features, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    train_idx, val_idx = _split_by_subject(trials, args.split_seed, args.val_ratio)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

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
    per_class_f1 = f1_score(y_val, val_pred, average=None).tolist()

    report = {
        "manifest_path": str(manifest_path),
        "split_by": "subject",
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "train_acc": train_acc,
        "train_macro_f1": train_f1,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "val_per_class_f1": per_class_f1,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(
        f"[baseline] train_acc={train_acc:.4f} train_macro_f1={train_f1:.4f} "
        f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}",
        flush=True,
    )
    print(f"[baseline] val_per_class_f1={per_class_f1}", flush=True)
    print(f"[baseline] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
