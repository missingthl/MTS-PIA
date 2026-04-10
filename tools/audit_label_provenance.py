import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.seed_raw_trials import build_trial_index


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


def _load_fif_mapping(path: Path):
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    mapping = {}
    if isinstance(data, list):
        for row in data:
            out_path = row.get("out_path")
            cnt_path = row.get("cnt_path")
            if out_path and cnt_path:
                mapping[str(out_path)] = str(cnt_path)
    return mapping


def _resolve_cnt_path(source_path: str, mapping: dict, raw_root: str | None):
    if source_path.endswith(".cnt"):
        return source_path
    if source_path in mapping:
        return mapping[source_path]
    if source_path.endswith(".fif"):
        base = Path(source_path).name
        if base.endswith("_eeg62_raw.fif"):
            cnt_name = base.replace("_eeg62_raw.fif", ".cnt")
        else:
            cnt_name = base.replace(".fif", ".cnt")
        if raw_root:
            candidate = Path(raw_root) / cnt_name
            if candidate.is_file():
                return str(candidate)
    return None


def main():
    parser = argparse.ArgumentParser(description="Audit manifest label provenance against SEED labels.")
    parser.add_argument(
        "--manifest-path",
        default="logs/seed1_tsm_cov_spd_full_rel_seq_manifest.json",
        help="manifest path (tsm seq manifest)",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=50,
        help="number of samples to audit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--time-txt",
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt",
        help="time.txt for trial boundaries",
    )
    parser.add_argument(
        "--stim-xlsx",
        default="data/SEED/SEED_EEG/SEED_stimulation.xlsx",
        help="SEED_stimulation.xlsx path",
    )
    parser.add_argument(
        "--fif-manifest",
        default="logs/seed_raw_fif_manifest.json",
        help="mapping file for CNT->FIF conversion",
    )
    parser.add_argument(
        "--raw-root",
        default="data/SEED/SEED_EEG/SEED_RAW_EEG",
        help="CNT root for fallback resolution",
    )
    parser.add_argument(
        "--out",
        default="logs/audit_label_provenance_full.json",
        help="output JSON report path",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    file_paths, trials, labels = _load_manifest(manifest_path)
    if trials is None:
        raise ValueError("manifest missing trials; cannot resolve subject/session/trial keys")
    if labels is None:
        raise ValueError("manifest missing labels")
    if len(trials) != len(file_paths):
        raise ValueError("file_paths and trials length mismatch")

    mapping = _load_fif_mapping(Path(args.fif_manifest))
    rng = random.Random(int(args.seed))
    total = len(trials)
    sample_n = min(int(args.sample_n), total)
    indices = rng.sample(range(total), sample_n)

    mismatches = []
    for idx in indices:
        row = trials[idx]
        manifest_label = int(row["label"])
        subject = row.get("subject")
        session = row.get("session")
        trial = row.get("trial")
        source_path = row.get("source_cnt_path") or row.get("cnt_path") or file_paths[idx]
        cnt_path = _resolve_cnt_path(str(source_path), mapping, args.raw_root)
        if cnt_path is None:
            mismatches.append(
                {
                    "path": str(source_path),
                    "key": f"{subject}_s{session}_t{trial}",
                    "manifest_label": manifest_label,
                    "source_label": None,
                    "error": "cnt_path_not_found",
                }
            )
            continue
        trial_list = build_trial_index(cnt_path, args.time_txt, args.stim_xlsx)
        trial_idx = int(trial)
        if trial_idx < 0 or trial_idx >= len(trial_list):
            mismatches.append(
                {
                    "path": str(source_path),
                    "key": f"{subject}_s{session}_t{trial}",
                    "manifest_label": manifest_label,
                    "source_label": None,
                    "error": "trial_index_out_of_range",
                }
            )
            continue
        source_label = int(trial_list[trial_idx].label)
        if source_label != manifest_label:
            mismatches.append(
                {
                    "path": str(source_path),
                    "key": f"{subject}_s{session}_t{trial}",
                    "manifest_label": manifest_label,
                    "source_label": source_label,
                }
            )

    report = {
        "manifest_path": str(manifest_path),
        "time_txt": str(args.time_txt),
        "stim_xlsx": str(args.stim_xlsx),
        "sample_n": int(sample_n),
        "mismatch_count": int(len(mismatches)),
        "mismatch_examples": mismatches[:10],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"[audit] mismatch_count={len(mismatches)} / {sample_n}", flush=True)
    if mismatches:
        print("[audit] mismatch_examples:", flush=True)
        for row in mismatches[:10]:
            print(row, flush=True)
    print(f"[audit] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
