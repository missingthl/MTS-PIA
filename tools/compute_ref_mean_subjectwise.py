import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
        return list(file_paths), trials
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


def main():
    parser = argparse.ArgumentParser(description="Compute subjectwise reference means.")
    parser.add_argument(
        "--manifest_path",
        default="logs/seed1_tsm_cov_spd_full_rel_seq_manifest.json",
        help="manifest path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_trials_per_subject", type=int, default=50)
    parser.add_argument("--max_windows_per_trial", type=int, default=20)
    parser.add_argument(
        "--out",
        default="logs/ref_mean_subjectwise.pt",
        help="output ref mean dict",
    )
    parser.add_argument(
        "--meta-out",
        default="logs/ref_mean_subjectwise.meta.json",
        help="output meta json",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    file_paths, trials = _load_manifest(manifest_path)
    if trials is None:
        raise ValueError("manifest missing trials")

    rng = np.random.default_rng(int(args.seed))
    by_subject = {}
    for idx, row in enumerate(trials):
        by_subject.setdefault(str(row.get("subject")), []).append(idx)

    ref_means = {}
    meta = {
        "manifest_path": str(manifest_path),
        "seed": int(args.seed),
        "max_trials_per_subject": int(args.max_trials_per_subject),
        "max_windows_per_trial": int(args.max_windows_per_trial),
        "subjects": {},
    }

    for subject, idxs in sorted(by_subject.items()):
        idxs = list(idxs)
        total_trials = len(idxs)
        if args.max_trials_per_subject > 0 and total_trials > args.max_trials_per_subject:
            idxs = rng.choice(idxs, size=int(args.max_trials_per_subject), replace=False).tolist()
        total_windows = 0
        used_windows = 0
        covs_by_band = {b: [] for b in range(5)}
        for idx in idxs:
            cov_path = _resolve_path(file_paths[idx], manifest_path)
            cov_seq = _load_cov(cov_path)
            total_windows += int(cov_seq.shape[0])
            if args.max_windows_per_trial > 0 and cov_seq.shape[0] > args.max_windows_per_trial:
                win_idx = rng.choice(
                    cov_seq.shape[0],
                    size=int(args.max_windows_per_trial),
                    replace=False,
                )
                win_idx = np.sort(win_idx)
            else:
                win_idx = np.arange(cov_seq.shape[0])
            used_windows += int(len(win_idx))
            for b in range(5):
                covs_by_band[b].append(np.asarray(cov_seq[win_idx, b], dtype=np.float64))

        subject_means = []
        for b in range(5):
            mats = np.concatenate(covs_by_band[b], axis=0)
            covs_t = torch.from_numpy(mats).to(dtype=torch.float64)
            mean = RiemannianUtils.cal_riemannian_mean(covs_t)
            subject_means.append(mean.to(dtype=torch.float32).cpu())
        ref_means[subject] = torch.stack(subject_means, dim=0)
        meta["subjects"][subject] = {
            "trials_total": int(total_trials),
            "trials_used": int(len(idxs)),
            "windows_total": int(total_windows),
            "windows_used": int(used_windows),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref_means, out_path)
    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[ref] subjects={len(ref_means)} out={out_path}", flush=True)
    print(f"[ref] meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
