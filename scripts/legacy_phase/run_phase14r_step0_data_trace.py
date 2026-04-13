import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from datasets.seed_raw_trials import build_trial_index, load_seed_stimulation_labels, load_seed_time_points
from manifold_raw.features import bandpass, parse_band_spec, window_slices


CONFIG = {
    "seed": 0,
    "raw_root": "data/SEED/SEED_EEG/SEED_RAW_EEG",
    "raw_backend": "cnt",
    "time_unit": None,
    "trial_offset_sec": 0.0,
    "window_sec": 4.0,
    "hop_sec": 4.0,
    "bands": "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    "norm_mode": "per_band_global_z",
    "out_dir": "promoted_results/phase14r/step0_data_trace/seed1/seed0",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def _parse_cnt_name(cnt_path: str) -> Tuple[str, int]:
    base = os.path.splitext(os.path.basename(cnt_path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid CNT filename: {cnt_path}")
    return parts[0], int(parts[1])


def _sorted_raw_files(raw_root: str, ext: str) -> List[str]:
    files = [str(p) for p in Path(raw_root).iterdir() if p.suffix.lower() == ext]
    return sorted(files, key=lambda p: _parse_cnt_name(p))


def _trial_id(trial) -> str:
    return f"{trial.subject}_s{trial.session}_t{trial.trial}"


def _deterministic_split(trials: List[dict], seed: int) -> Tuple[List[dict], List[dict]]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(trials))
    n_train = int(0.8 * len(trials))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return [trials[i] for i in train_idx], [trials[i] for i in test_idx]


def _audit_split(train_ids: List[str], test_ids: List[str]) -> dict:
    train_set = set(train_ids)
    test_set = set(test_ids)
    inter = sorted(list(train_set.intersection(test_set)))
    return {
        "n_train_trials": len(train_ids),
        "n_test_trials": len(test_ids),
        "intersection_size": len(inter),
        "violations": inter[:5],
    }


def _band_norm_global(x: np.ndarray) -> np.ndarray:
    mean = float(x.mean())
    std = float(x.std()) + 1e-6
    return (x - mean) / std


def _window_stats(counts: List[int]) -> dict:
    if not counts:
        return {}
    arr = np.asarray(counts)
    return {
        "min": int(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": int(np.max(arr)),
    }


class RawWindowDatasetTrace(Dataset):
    def __init__(
        self,
        trial_rows: List[dict],
        raw_backend: str,
        window_sec: float,
        hop_sec: float,
        bands: str,
        trial_offset_sec: float,
        norm_mode: str,
    ):
        self.trial_rows = trial_rows
        self.raw_backend = raw_backend
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.trial_offset_sec = trial_offset_sec
        self.bands = parse_band_spec(bands)
        self.norm_mode = norm_mode
        self.windows = []
        self.trial_ids = []
        self.labels = []
        self.trial_meta = []
        self.trial_window_counts = []
        raw_cache = {}
        for idx, row in enumerate(self.trial_rows):
            cnt_path = row["cnt_path"]
            if cnt_path not in raw_cache:
                raw = load_one_raw(cnt_path, backend=self.raw_backend, preload=False)
                raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
                raw_cache[cnt_path] = raw62
            raw62 = raw_cache[cnt_path]
            fs = float(raw62.info["sfreq"])
            t = row["trial_obj"]
            start_idx = int(round((t.t_start_s + self.trial_offset_sec) * fs))
            end_idx = int(round((t.t_end_s + self.trial_offset_sec) * fs))
            n_samples = max(0, end_idx - start_idx)
            slices = window_slices(n_samples, fs, self.window_sec, self.hop_sec)
            self.trial_ids.append(row["trial_id"])
            self.labels.append(int(row["label"]))
            self.trial_meta.append(
                {
                    "cnt_path": cnt_path,
                    "trial_obj": t,
                    "fs": fs,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )
            self.trial_window_counts.append(len(slices))
            for w_idx, (s, e) in enumerate(slices):
                self.windows.append((idx, w_idx, s, e))

        self.raw_cache = {}
        self.last_trial_idx = None
        self.last_band_full = None

    def __len__(self) -> int:
        return len(self.windows)

    def _load_trial_band_full(self, trial_idx: int) -> dict:
        if self.last_trial_idx == trial_idx and self.last_band_full is not None:
            return self.last_band_full
        meta = self.trial_meta[trial_idx]
        cnt_path = meta["cnt_path"]
        if cnt_path not in self.raw_cache:
            raw = load_one_raw(cnt_path, backend=self.raw_backend, preload=False)
            raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
            self.raw_cache[cnt_path] = raw62
        raw62 = self.raw_cache[cnt_path]
        start_idx = meta["start_idx"]
        end_idx = meta["end_idx"]
        seg = raw62.get_data(start=start_idx, stop=end_idx).astype(np.float32, copy=False)
        fs = meta["fs"]
        band_full = {b.name: bandpass(seg, fs, b) for b in self.bands}
        self.last_trial_idx = trial_idx
        self.last_band_full = band_full
        return band_full

    def __getitem__(self, idx: int):
        trial_idx, window_idx, start, end = self.windows[idx]
        band_full = self._load_trial_band_full(trial_idx)
        band_windows = []
        for b in self.bands:
            bw = band_full[b.name][:, start:end]
            if self.norm_mode == "per_band_global_z":
                bw = _band_norm_global(bw)
            band_windows.append(bw)
        x_cat = np.concatenate(band_windows, axis=1)
        y = int(self.trial_rows[trial_idx]["label"])
        tid = self.trial_rows[trial_idx]["trial_id"]
        meta = self.trial_meta[trial_idx]
        return (
            torch.tensor(x_cat, dtype=torch.float32),
            y,
            tid,
            int(window_idx),
            int(meta["start_idx"] + start),
            int(meta["start_idx"] + end),
        )


def main() -> None:
    cfg = CONFIG
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    ext = ".fif" if cfg["raw_backend"] == "fif" else ".cnt"
    raw_root = cfg["raw_root"]
    raw_files = _sorted_raw_files(raw_root, ext)
    if not raw_files:
        raise FileNotFoundError(f"No {ext} files under {raw_root}")

    time_txt = os.path.join(raw_root, "time.txt")
    if not os.path.isfile(time_txt):
        time_txt = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
    stim_xlsx = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")

    start_pts, end_pts = load_seed_time_points(time_txt)
    labels = load_seed_stimulation_labels(stim_xlsx)

    trials_all = []
    fs_cache = {}
    errors = []
    for cnt_path in raw_files:
        try:
            trials = build_trial_index(cnt_path, time_txt, stim_xlsx, time_unit=cfg["time_unit"])
        except Exception as exc:
            errors.append(f"build_trial_index failed for {cnt_path}: {exc}")
            continue
        if cnt_path not in fs_cache:
            raw = load_one_raw(cnt_path, backend=cfg["raw_backend"], preload=False)
            raw62, _ = build_eeg62_view(raw, locs_path="data/SEED/channel_62_pos.locs")
            fs_cache[cnt_path] = float(raw62.info["sfreq"])
        for t in trials:
            trials_all.append(
                {
                    "trial_id": _trial_id(t),
                    "label": int(t.label),
                    "trial_obj": t,
                    "cnt_path": cnt_path,
                }
            )

    if errors:
        write_text(os.path.join(out_dir, "DATA_TRACE_REPORT.md"), "\n".join(["FAIL", *errors]))
        raise RuntimeError("Errors encountered while building trial index")

    trials_all = sorted(trials_all, key=lambda r: r["trial_id"])
    trial_ids = [r["trial_id"] for r in trials_all]
    if len(trial_ids) != len(set(trial_ids)):
        raise RuntimeError("duplicate trial_id_str detected")

    train_trials, test_trials = _deterministic_split(trials_all, cfg["seed"])
    train_ids = [r["trial_id"] for r in train_trials]
    test_ids = [r["trial_id"] for r in test_trials]
    audit = _audit_split(train_ids, test_ids)
    if audit["intersection_size"] != 0:
        raise RuntimeError("split leakage detected")

    train_counts = []
    test_counts = []
    for row in train_trials:
        t = row["trial_obj"]
        fs = fs_cache[row["cnt_path"]]
        start_idx = int(round((t.t_start_s + cfg["trial_offset_sec"]) * fs))
        end_idx = int(round((t.t_end_s + cfg["trial_offset_sec"]) * fs))
        n_samples = max(0, end_idx - start_idx)
        slices = window_slices(n_samples, fs, cfg["window_sec"], cfg["hop_sec"])
        train_counts.append(len(slices))
    for row in test_trials:
        t = row["trial_obj"]
        fs = fs_cache[row["cnt_path"]]
        start_idx = int(round((t.t_start_s + cfg["trial_offset_sec"]) * fs))
        end_idx = int(round((t.t_end_s + cfg["trial_offset_sec"]) * fs))
        n_samples = max(0, end_idx - start_idx)
        slices = window_slices(n_samples, fs, cfg["window_sec"], cfg["hop_sec"])
        test_counts.append(len(slices))

    train_ds = RawWindowDatasetTrace(
        train_trials,
        raw_backend=cfg["raw_backend"],
        window_sec=cfg["window_sec"],
        hop_sec=cfg["hop_sec"],
        bands=cfg["bands"],
        trial_offset_sec=cfg["trial_offset_sec"],
        norm_mode=cfg["norm_mode"],
    )
    test_ds = RawWindowDatasetTrace(
        test_trials,
        raw_backend=cfg["raw_backend"],
        window_sec=cfg["window_sec"],
        hop_sec=cfg["hop_sec"],
        bands=cfg["bands"],
        trial_offset_sec=cfg["trial_offset_sec"],
        norm_mode=cfg["norm_mode"],
    )

    import pandas as pd

    pd.DataFrame(
        {
            "trial_id_str": train_ds.trial_ids,
            "label": train_ds.labels,
            "n_windows": train_ds.trial_window_counts,
        }
    ).to_csv(os.path.join(out_dir, "trial_window_counts_train.csv"), index=False)
    pd.DataFrame(
        {
            "trial_id_str": test_ds.trial_ids,
            "label": test_ds.labels,
            "n_windows": test_ds.trial_window_counts,
        }
    ).to_csv(os.path.join(out_dir, "trial_window_counts_test.csv"), index=False)

    sample_indices = [0, len(train_ds) // 2, max(0, len(train_ds) - 1)]
    sample_dump = {"split": "train", "indices": []}
    for idx in sample_indices:
        x, y, tid, widx, abs_start, abs_end = train_ds[idx]
        sample_dump["indices"].append(
            {
                "index": int(idx),
                "x_shape": list(x.shape),
                "y": int(y),
                "trial_id_str": tid,
                "window_idx": int(widx),
                "abs_sample_range": [int(abs_start), int(abs_end)],
            }
        )
    write_json(os.path.join(out_dir, "sample_dump.json"), sample_dump)

    raw_probe_path = raw_files[0]
    raw_obj = load_one_raw(raw_probe_path, backend=cfg["raw_backend"], preload=False)
    info = raw_obj.info
    label_fields = [k for k in info.keys() if "label" in str(k).lower()]
    raw_header_probe = {
        "raw_path": raw_probe_path,
        "backend": cfg["raw_backend"],
        "sfreq": float(info["sfreq"]),
        "n_times": int(raw_obj.n_times),
        "n_channels": int(info["nchan"]),
        "info_keys": sorted(list(info.keys())),
        "label_fields_found": label_fields,
        "subject_info": info.get("subject_info"),
        "description": info.get("description"),
        "experimenter": info.get("experimenter"),
        "annotations_len": int(len(raw_obj.annotations)) if hasattr(raw_obj, "annotations") else None,
        "has_annotations": bool(len(raw_obj.annotations)) if hasattr(raw_obj, "annotations") else None,
    }
    write_json(os.path.join(out_dir, "raw_header_probe.json"), raw_header_probe)

    counts_summary = {
        "raw_root": raw_root,
        "raw_suffix": ext,
        "n_raw_files": len(raw_files),
        "time_txt_path": time_txt,
        "stim_xlsx_path": stim_xlsx,
        "label_column": "Label",
        "n_trials_total": len(trials_all),
        "n_trials_train": len(train_trials),
        "n_trials_test": len(test_trials),
        "label_count_from_stim": len(labels),
        "trial_boundary_count": len(start_pts),
        "split_audit": audit,
        "n_windows_total_train": int(np.sum(train_counts)),
        "n_windows_total_test": int(np.sum(test_counts)),
        "window_stats_train": _window_stats(train_counts),
        "window_stats_test": _window_stats(test_counts),
        "trial_id_template": "{subject}_s{session}_t{trial}",
    }
    write_json(os.path.join(out_dir, "counts_summary.json"), counts_summary)

    report_lines = []
    report_lines.append("# Phase 14R Step0 Data Path Forensics")
    report_lines.append("")
    report_lines.append("## Source of Truth")
    report_lines.append(f"- raw_root: {raw_root}")
    report_lines.append(f"- raw_suffix: {ext}")
    report_lines.append(f"- n_raw_files: {len(raw_files)}")
    report_lines.append(f"- time_txt_path: {time_txt}")
    report_lines.append(f"- stim_xlsx_path: {stim_xlsx}")
    report_lines.append("- label_loader: datasets/seed_raw_trials.py::load_seed_stimulation_labels")
    report_lines.append("- label_column: Label")
    report_lines.append("- trial_id_str template: {subject}_s{session}_t{trial}")
    report_lines.append("")
    report_lines.append("## Trial Layer")
    report_lines.append(f"- n_trials_total: {len(trials_all)}")
    report_lines.append(f"- n_trials_train: {len(train_trials)}")
    report_lines.append(f"- n_trials_test: {len(test_trials)}")
    report_lines.append(f"- label_count_from_stim: {len(labels)}")
    report_lines.append(f"- time_points_count: {len(start_pts)}")
    report_lines.append("")
    report_lines.append("## Window Layer")
    report_lines.append(f"- n_windows_total_train: {int(np.sum(train_counts))}")
    report_lines.append(f"- n_windows_total_test: {int(np.sum(test_counts))}")
    report_lines.append(f"- window_stats_train: {counts_summary['window_stats_train']}")
    report_lines.append(f"- window_stats_test: {counts_summary['window_stats_test']}")
    report_lines.append("")
    report_lines.append("## Split & Alignment")
    report_lines.append(f"- split_mode: trial_80_20 seed={cfg['seed']}")
    report_lines.append(f"- train_test_intersection_size: {audit['intersection_size']}")
    report_lines.append("")
    report_lines.append("## Raw Header Probe")
    report_lines.append(f"- raw_path: {raw_probe_path}")
    report_lines.append(f"- label_fields_found: {label_fields}")
    report_lines.append("")
    report_lines.append("## Counts Table")
    report_lines.append("")
    report_lines.append("| layer | count |")
    report_lines.append("| --- | --- |")
    report_lines.append(f"| raw_files | {len(raw_files)} |")
    report_lines.append(f"| trials_total | {len(trials_all)} |")
    report_lines.append(f"| trials_train | {len(train_trials)} |")
    report_lines.append(f"| trials_test | {len(test_trials)} |")
    report_lines.append(f"| windows_train | {int(np.sum(train_counts))} |")
    report_lines.append(f"| windows_test | {int(np.sum(test_counts))} |")
    write_text(os.path.join(out_dir, "DATA_TRACE_REPORT.md"), "\n".join(report_lines))


if __name__ == "__main__":
    main()
