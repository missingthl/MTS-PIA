import argparse
import json
import random
from pathlib import Path

import numpy as np


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


def _resolve_trial_ids(trials, total):
    trial_ids = [None] * total
    if not isinstance(trials, list) or len(trials) != total:
        return trial_ids
    for i, t in enumerate(trials):
        trial_id = t.get("trial_id") or t.get("trial_uid")
        if trial_id is None:
            subj = t.get("subject") or t.get("subject_id") or t.get("subject_idx")
            trial_idx = t.get("trial") or t.get("trial_idx")
            session = t.get("session")
            if trial_idx is not None and subj is not None and session is not None:
                trial_id = f"{subj}_s{session}_t{trial_idx}"
            elif trial_idx is not None and subj is not None:
                trial_id = f"{subj}_t{trial_idx}"
            elif trial_idx is not None:
                trial_id = str(trial_idx)
        if trial_id is not None:
            trial_ids[i] = str(trial_id)
    return trial_ids


def _stats(values):
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(values)
    return {
        "min": int(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit manifest label consistency")
    parser.add_argument(
        "--manifest",
        default="logs/seed1_tsm_cov_spd_full_rel_seq_manifest.json",
        help="Path to manifest JSON",
    )
    parser.add_argument(
        "--out",
        default="logs/audit_label_consistency_full.json",
        help="Output JSON path",
    )
    parser.add_argument("--per-class", type=int, default=20, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    file_paths, trials, labels = _load_manifest(manifest_path)
    if labels is None:
        raise ValueError("manifest missing labels")
    total = len(file_paths)
    if total != len(labels):
        raise ValueError(f"file_paths/labels length mismatch: {total} vs {len(labels)}")

    trial_ids = _resolve_trial_ids(trials, total)
    filepath_label = {}
    conflict_filepaths = {}
    duplicate_filepaths = 0
    for path, label in zip(file_paths, labels):
        if path in filepath_label:
            duplicate_filepaths += 1
            if filepath_label[path] != label:
                conflict_filepaths.setdefault(path, set()).update(
                    [filepath_label[path], label]
                )
        else:
            filepath_label[path] = label

    trial_label = {}
    conflict_trials = {}
    for trial_id, label in zip(trial_ids, labels):
        if trial_id is None:
            continue
        if trial_id in trial_label and trial_label[trial_id] != label:
            conflict_trials.setdefault(trial_id, set()).update(
                [trial_label[trial_id], label]
            )
        else:
            trial_label[trial_id] = label

    label_hist = {}
    for label in labels:
        label_hist[int(label)] = label_hist.get(int(label), 0) + 1

    rng = random.Random(int(args.seed))
    by_class = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    per_class_T = {cls: [] for cls in by_class}
    invalid_shapes = []
    missing_files = []
    sample_counts = {}
    for cls, indices in by_class.items():
        sample_n = min(int(args.per_class), len(indices))
        sample_counts[cls] = sample_n
        sampled = rng.sample(indices, sample_n) if sample_n else []
        for idx in sampled:
            path = Path(file_paths[idx])
            if not path.exists():
                missing_files.append(str(path))
                continue
            try:
                arr = np.load(path, mmap_mode="r")
            except Exception as exc:
                invalid_shapes.append(
                    {"path": str(path), "error": f"load_error:{type(exc).__name__}"}
                )
                continue
            if arr.ndim != 4 or tuple(arr.shape[1:]) != (5, 62, 62):
                invalid_shapes.append(
                    {"path": str(path), "shape": list(arr.shape)}
                )
                continue
            per_class_T[cls].append(int(arr.shape[0]))

    per_class_T_stats = {str(cls): _stats(values) for cls, values in per_class_T.items()}

    summary = {
        "manifest_path": str(manifest_path),
        "total": total,
        "duplicate_filepaths": int(duplicate_filepaths),
        "conflict_filepath_count": int(len(conflict_filepaths)),
        "conflict_filepaths": {
            path: sorted(list(labels)) for path, labels in conflict_filepaths.items()
        },
        "conflict_trial_count": int(len(conflict_trials)),
        "conflict_trials": {
            trial_id: sorted(list(labels)) for trial_id, labels in conflict_trials.items()
        },
        "label_hist": label_hist,
        "sample_counts": sample_counts,
        "per_class_T_stats": per_class_T_stats,
        "invalid_shapes": invalid_shapes,
        "missing_files": missing_files,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"[audit] conflict_filepath_count={len(conflict_filepaths)}", flush=True)
    print(f"[audit] conflict_trial_count={len(conflict_trials)}", flush=True)
    print(f"[audit] label_hist={label_hist}", flush=True)
    print(f"[audit] per_class_T_stats={per_class_T_stats}", flush=True)
    print(f"[audit] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
