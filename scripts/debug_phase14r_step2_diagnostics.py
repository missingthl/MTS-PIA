import os
import sys
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from datasets.seed_raw_trials import build_trial_index
from manifold_raw.features import bandpass, parse_band_spec, window_slices
from models.raw_manifold_net import RawCovTSMNet


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
    "spd_eps": 1e-3,
    "logmap_eps": 1e-6,
    "batch_size": 8,
    "debug_batches": 5,
    "max_train_trials": 60,
    "max_test_trials": 30,
    "out_dir": "promoted_results/phase14r/step2c/seed1/seed0/collapse_forensics",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_error(path: str, msg: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(msg)


def _parse_cnt_name(cnt_path: str) -> Tuple[int, int]:
    base = os.path.splitext(os.path.basename(cnt_path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid CNT filename: {cnt_path}")
    return int(parts[0]), int(parts[1])


def _sorted_cnt_files(raw_root: str, ext: str) -> List[str]:
    files = [str(p) for p in Path(raw_root).iterdir() if p.suffix.lower() == ext]
    return sorted(files, key=lambda p: _parse_cnt_name(p))


def _trial_id(t) -> str:
    return f"{t.subject}_s{t.session}_t{t.trial}"


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


class RawCovWindowDataset(Dataset):
    def __init__(
        self,
        trial_rows: List[dict],
        raw_backend: str,
        window_sec: float,
        hop_sec: float,
        bands: str,
        trial_offset_sec: float,
    ):
        self.trial_rows = trial_rows
        self.raw_backend = raw_backend
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.trial_offset_sec = trial_offset_sec
        self.bands = parse_band_spec(bands)
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
            if n_samples <= 0:
                continue
            slices = window_slices(n_samples, fs, self.window_sec, self.hop_sec)
            self.trial_ids.append(row["trial_id"])
            self.labels.append(int(row["label"]))
            self.trial_meta.append({"cnt_path": cnt_path, "trial_obj": t, "fs": fs, "start_idx": start_idx, "end_idx": end_idx})
            self.trial_window_counts.append(len(slices))
            for w_idx, (s, e) in enumerate(slices):
                self.windows.append((idx, w_idx, s, e))

        self.raw_cache = {}
        self.last_trial_idx = None
        self.last_band_full = None

    def __len__(self):
        return len(self.windows)

    def _load_trial_band_full(self, trial_idx: int):
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

    def __getitem__(self, idx):
        trial_idx, window_idx, start, end = self.windows[idx]
        band_full = self._load_trial_band_full(trial_idx)
        band_windows = []
        for b in self.bands:
            bw = band_full[b.name][:, start:end]
            bw = _band_norm_global(bw)
            band_windows.append(bw)
        x_cat = np.concatenate(band_windows, axis=1)
        y = int(self.trial_rows[trial_idx]["label"])
        tid = self.trial_rows[trial_idx]["trial_id"]
        return torch.tensor(x_cat, dtype=torch.float32), y, tid, int(window_idx)


def _aggregate_debug(debug_list: List[dict]) -> dict:
    if not debug_list:
        return {}

    agg = {}
    scalar_keys = [
        ("cond_pre_eps_p95", "cond_pre_eps_p95"),
        ("cond_post_eps_p95", "cond_post_eps_p95"),
    ]
    for out_key, k in scalar_keys:
        vals = [d.get(k) for d in debug_list if k in d]
        if vals:
            agg[out_key] = float(np.mean(vals))

    for tag in ["eig_pre_eps", "eig_post_eps"]:
        vals = [d.get(tag, {}) for d in debug_list]
        if vals:
            agg[tag] = {k: float(np.mean([v.get(k, 0.0) for v in vals])) for k in vals[0].keys()}

    for tag in ["cov_pre_eps", "cov_post_eps"]:
        vals = [d.get(tag, {}) for d in debug_list]
        if vals:
            agg[tag] = {k: float(np.mean([v.get(k, 0.0) for v in vals])) for k in vals[0].keys()}

    for tag in ["nan_inf_x_in", "nan_inf_x_conv", "nan_inf_cov_pre", "nan_inf_cov_post", "nan_inf_eig_pre", "nan_inf_eig_post"]:
        flags = [d.get(tag, {}) for d in debug_list]
        if flags:
            agg[tag] = {k: bool(any(f.get(k, False) for f in flags)) for k in flags[0].keys()}

    return agg


def _hard_asserts(debug: dict, eps: float):
    eig_post = debug.get("eig_post_eps", {})
    if eig_post:
        min_eig = eig_post.get("min", 0.0)
        if min_eig < eps * 0.99:
            raise RuntimeError(f"eig_post_eps min {min_eig} < eps*0.99")

    cov_post = debug.get("cov_post_eps", {})
    if cov_post:
        symm = cov_post.get("symmetry_error_mean", 0.0)
        if symm > 1e-5:
            raise RuntimeError(f"symmetry error too high: {symm}")

    for k in ["nan_inf_cov_post", "nan_inf_eig_post"]:
        flags = debug.get(k, {})
        if flags.get("has_nan") or flags.get("has_inf"):
            raise RuntimeError(f"NaN/Inf detected in {k}: {flags}")


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm(2).item()) ** 2
    return float(math.sqrt(total))


def _select_trials(trials: List[dict], max_trials: int | None) -> List[dict]:
    trials_sorted = sorted(trials, key=lambda r: r["trial_id"])
    if max_trials is None:
        return trials_sorted
    if len(trials_sorted) < max_trials:
        raise RuntimeError(f"Not enough trials for selection: have {len(trials_sorted)}, need {max_trials}")
    return trials_sorted[:max_trials]


def _coverage_assertions(tag: str, trial_ids: List[str], window_counts: List[int], min_windows: int) -> None:
    n_selected = len(trial_ids)
    n_with_windows = sum(1 for c in window_counts if c > 0)
    if n_selected != n_with_windows:
        missing = [tid for tid, c in zip(trial_ids, window_counts) if c == 0][:5]
        raise RuntimeError(f"{tag} trials missing windows: selected={n_selected} with_windows={n_with_windows} sample={missing}")
    min_count = min(window_counts) if window_counts else 0
    if min_count < min_windows:
        low = [tid for tid, c in zip(trial_ids, window_counts) if c < min_windows][:5]
        raise RuntimeError(f"{tag} trials with too few windows: min={min_count} sample={low}")


def _class_counts(trials: List[dict]) -> Dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for row in trials:
        lbl = int(row["label"])
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _select_micro_trials(trials: List[dict]) -> Dict[str, List[str]]:
    trials_sorted = sorted(trials, key=lambda r: r["trial_id"])
    by_class = {0: [], 1: [], 2: []}
    for row in trials_sorted:
        by_class[int(row["label"])].append(row["trial_id"])
    sel = {c: by_class[c][:3] for c in [0, 1, 2]}
    all_sel = sel[0] + sel[1] + sel[2]
    return {
        "class0": sel[0],
        "class1": sel[1],
        "class2": sel[2],
        "all": all_sel,
    }


def main():
    cfg = CONFIG
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    ext = ".fif" if cfg["raw_backend"] == "fif" else ".cnt"
    try:
        cnt_files = _sorted_cnt_files(cfg["raw_root"], ext)
        if not cnt_files:
            raise FileNotFoundError(f"No {ext} files under {cfg['raw_root']}")
        time_txt = os.path.join(cfg["raw_root"], "time.txt")
        stim_xlsx = os.path.join("data", "SEED", "SEED_EEG", "SEED_stimulation.xlsx")
        if not os.path.isfile(time_txt):
            time_txt = os.path.join("data", "SEED", "SEED_EEG", "time.txt")
        if not os.path.isfile(time_txt):
            raise FileNotFoundError("time.txt not found")

        trials_all = []
        for cnt_path in cnt_files:
            trials = build_trial_index(cnt_path, time_txt, stim_xlsx, time_unit=cfg["time_unit"])
            for t in trials:
                trials_all.append(
                    {
                        "trial_id": _trial_id(t),
                        "label": int(t.label),
                        "trial_obj": t,
                        "cnt_path": cnt_path,
                    }
                )
    except Exception as e:
        write_error(os.path.join(out_dir, "ERROR.md"), f"ERROR: {e}")
        raise

    trials_all = sorted(trials_all, key=lambda r: r["trial_id"])
    if len(trials_all) != len({t["trial_id"] for t in trials_all}):
        write_error(os.path.join(out_dir, "ERROR.md"), "ERROR: duplicate trial_id_str")
        raise RuntimeError("duplicate trial_id_str")

    train_trials, test_trials = _deterministic_split(trials_all, cfg["seed"])
    train_trials = _select_trials(train_trials, cfg["max_train_trials"])
    test_trials = _select_trials(test_trials, cfg["max_test_trials"])
    audit = _audit_split([t["trial_id"] for t in train_trials], [t["trial_id"] for t in test_trials])
    if audit["intersection_size"] != 0:
        write_error(os.path.join(out_dir, "ERROR.md"), f"ERROR: split leakage {audit}")
        raise RuntimeError("split leakage detected")
    write_json(os.path.join(out_dir, "alignment_audit.json"), audit)

    class_counts = _class_counts(train_trials)
    if any(class_counts.get(c, 0) < 3 for c in [0, 1, 2]):
        micro_selection = _select_micro_trials(train_trials)
        write_json(os.path.join(out_dir, "micro_selection.json"), micro_selection)
        report_lines = [
            "# Micro Overfit Report",
            "",
            "status = FAIL_FAST_LABEL_COVERAGE",
            f"n_trials_class = {class_counts}",
            "explanation = micro-overfit aborted; insufficient per-class trials in selected train subset",
        ]
        with open(os.path.join(out_dir, "micro_overfit_report.md"), "w") as f:
            f.write("\n".join(report_lines))
        raise RuntimeError("FAIL_FAST_LABEL_COVERAGE")

    micro_selection = _select_micro_trials(train_trials)
    write_json(os.path.join(out_dir, "micro_selection.json"), micro_selection)
    micro_ids = set(micro_selection["all"])
    test_ids = set([t["trial_id"] for t in test_trials])
    if micro_ids.intersection(test_ids):
        raise RuntimeError("Leakage detected: micro set intersects test split")

    train_ds = RawCovWindowDataset(
        train_trials,
        raw_backend=cfg["raw_backend"],
        window_sec=cfg["window_sec"],
        hop_sec=cfg["hop_sec"],
        bands=cfg["bands"],
        trial_offset_sec=cfg["trial_offset_sec"],
    )
    test_ds = RawCovWindowDataset(
        test_trials,
        raw_backend=cfg["raw_backend"],
        window_sec=cfg["window_sec"],
        hop_sec=cfg["hop_sec"],
        bands=cfg["bands"],
        trial_offset_sec=cfg["trial_offset_sec"],
    )

    _coverage_assertions("train", train_ds.trial_ids, train_ds.trial_window_counts, min_windows=2)
    _coverage_assertions("test", test_ds.trial_ids, test_ds.trial_window_counts, min_windows=2)

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

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug_outputs = {}
    for mode in ["none", "instancenorm"]:
        model = RawCovTSMNet(
            spd_eps=cfg["spd_eps"],
            logmap_eps=cfg["logmap_eps"],
            raw_norm_mode=mode,
        ).to(device)
        model.eval()
        batch_debug = []
        with torch.no_grad():
            for i, (xb, yb, tid, widx) in enumerate(train_loader):
                if i >= cfg["debug_batches"]:
                    break
                xb = xb.to(device)
                logits, dbg = model(xb, debug=True)
                batch_debug.append(dbg)
        agg = _aggregate_debug(batch_debug)
        debug_outputs[mode] = {"aggregate": agg, "batches": batch_debug}
        write_json(os.path.join(out_dir, f"diag_{mode}.json"), debug_outputs[mode])
        _hard_asserts(agg, cfg["spd_eps"])

    backward = {}
    for mode in ["none", "instancenorm"]:
        model = RawCovTSMNet(
            spd_eps=cfg["spd_eps"],
            logmap_eps=cfg["logmap_eps"],
            raw_norm_mode=mode,
        ).to(device)
        model.train()
        xb, yb, _, _ = next(iter(train_loader))
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        loss.backward()
        grad_conv = _grad_norm(model.conv.parameters())
        grad_tsm = _grad_norm(model.tsm_proj.parameters())
        grad_head = _grad_norm(model.head.parameters())
        backward[mode] = {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm_conv": grad_conv,
            "grad_norm_tsm": grad_tsm,
            "grad_norm_head": grad_head,
            "nan_inf_grad": {
                "conv": bool(not np.isfinite(grad_conv)),
                "tsm": bool(not np.isfinite(grad_tsm)),
                "head": bool(not np.isfinite(grad_head)),
            },
        }
    write_json(os.path.join(out_dir, "backward_check.json"), backward)

    smoke_rows = []
    for mode in ["none", "instancenorm"]:
        model = RawCovTSMNet(
            spd_eps=cfg["spd_eps"],
            logmap_eps=cfg["logmap_eps"],
            raw_norm_mode=mode,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        loss_start = None
        loss_end = None
        for i, (xb, yb, _, _) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            if loss_start is None:
                loss_start = float(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            loss_end = float(loss.detach().cpu().item())

        model.eval()
        all_probs = []
        all_labels = []
        all_trial_ids = []
        with torch.no_grad():
            for xb, yb, tid, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(yb.numpy().tolist())
                all_trial_ids.extend(list(tid))
        probs_all = np.vstack(all_probs)
        preds = np.argmax(probs_all, axis=1)
        win_acc = float(np.mean(preds == np.asarray(all_labels)))

        trial_map: Dict[str, dict] = {}
        for tid, y, p in zip(all_trial_ids, all_labels, probs_all):
            entry = trial_map.setdefault(tid, {"y": int(y), "p_sum": np.zeros(3), "n": 0})
            entry["p_sum"] += p
            entry["n"] += 1
        trial_acc = float(
            np.mean([int(np.argmax(v["p_sum"] / v["n"])) == v["y"] for v in trial_map.values()])
        )

        smoke_rows.append(
            {
                "variant": mode,
                "loss_start": loss_start,
                "loss_end": loss_end,
                "loss_drop": (loss_start - loss_end) if loss_start is not None and loss_end is not None else None,
                "val_trial_acc": trial_acc,
                "val_win_acc": win_acc,
                "win_pred_hist": np.bincount(preds, minlength=3).tolist(),
                "trial_pred_hist": np.bincount(
                    [int(np.argmax(v["p_sum"] / v["n"])) for v in trial_map.values()],
                    minlength=3,
                ).tolist(),
            }
        )

    pd.DataFrame(smoke_rows).to_csv(os.path.join(out_dir, "learnability_smoke_coverage_safe.csv"), index=False)

    summary_lines = []
    summary_lines.append("# Phase14R Step2c Coverage-Safe Summary")
    summary_lines.append("")
    summary_lines.append("## Coverage")
    summary_lines.append(f"- train_trials_selected: {len(train_ds.trial_ids)}")
    summary_lines.append(f"- test_trials_selected: {len(test_ds.trial_ids)}")
    summary_lines.append(f"- min_windows_train: {min(train_ds.trial_window_counts) if train_ds.trial_window_counts else 0}")
    summary_lines.append(f"- min_windows_test: {min(test_ds.trial_window_counts) if test_ds.trial_window_counts else 0}")
    summary_lines.append("")
    for mode in ["none", "instancenorm"]:
        agg = debug_outputs[mode]["aggregate"]
        summary_lines.append(f"## Variant: {mode}")
        summary_lines.append(f"- cond_post_eps_p95: {agg.get('cond_post_eps_p95')}")
        summary_lines.append(f"- eig_post_eps min: {agg.get('eig_post_eps', {}).get('min')}")
        summary_lines.append(f"- nan_inf_cov_post: {agg.get('nan_inf_cov_post')}")
        for row in smoke_rows:
            if row["variant"] == mode:
                summary_lines.append(f"- win_acc: {row['val_win_acc']}")
                summary_lines.append(f"- trial_acc: {row['val_trial_acc']}")
                summary_lines.append(f"- win_pred_hist: {row['win_pred_hist']}")
                summary_lines.append(f"- trial_pred_hist: {row['trial_pred_hist']}")
                break
        summary_lines.append("")
    summary_lines.append("## Reproduction")
    summary_lines.append("/home/thl/miniconda3/envs/pia/bin/python scripts/debug_phase14r_step2_diagnostics.py")

    with open(os.path.join(out_dir, "coverage_safe_summary.md"), "w") as f:
        f.write("\n".join(summary_lines))

    micro_trials = [t for t in train_trials if t["trial_id"] in micro_ids]
    micro_ds = RawCovWindowDataset(
        micro_trials,
        raw_backend=cfg["raw_backend"],
        window_sec=cfg["window_sec"],
        hop_sec=cfg["hop_sec"],
        bands=cfg["bands"],
        trial_offset_sec=cfg["trial_offset_sec"],
    )
    _coverage_assertions("micro", micro_ds.trial_ids, micro_ds.trial_window_counts, min_windows=2)
    micro_loader = DataLoader(micro_ds, batch_size=cfg["batch_size"], shuffle=False)

    micro_results = {}
    for mode in ["none", "instancenorm"]:
        model = RawCovTSMNet(
            spd_eps=cfg["spd_eps"],
            logmap_eps=cfg["logmap_eps"],
            raw_norm_mode=mode,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        loss_start = None
        loss_min = None
        loss_end = None
        iterations = 0
        while iterations < 200:
            for xb, yb, _, _ in micro_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = torch.nn.functional.cross_entropy(logits, yb)
                if loss_start is None:
                    loss_start = float(loss.detach().cpu().item())
                loss_min = float(loss.detach().cpu().item()) if loss_min is None else min(loss_min, float(loss.detach().cpu().item()))
                loss.backward()
                optimizer.step()
                loss_end = float(loss.detach().cpu().item())
                iterations += 1
                if iterations >= 200:
                    break

        model.eval()
        all_probs = []
        all_labels = []
        all_trial_ids = []
        with torch.no_grad():
            for xb, yb, tid, _ in micro_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(yb.numpy().tolist())
                all_trial_ids.extend(list(tid))
        probs_all = np.vstack(all_probs)
        preds = np.argmax(probs_all, axis=1)
        win_acc = float(np.mean(preds == np.asarray(all_labels)))
        trial_map: Dict[str, dict] = {}
        for tid, y, p in zip(all_trial_ids, all_labels, probs_all):
            entry = trial_map.setdefault(tid, {"y": int(y), "p_sum": np.zeros(3), "n": 0})
            entry["p_sum"] += p
            entry["n"] += 1
        trial_preds = [int(np.argmax(v["p_sum"] / v["n"])) for v in trial_map.values()]
        trial_acc = float(np.mean([p == v["y"] for p, v in zip(trial_preds, trial_map.values())]))

        micro_results[mode] = {
            "iterations": iterations,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_min": loss_min,
            "micro_win_acc": win_acc,
            "micro_trial_acc": trial_acc,
            "micro_pred_hist": {
                "win": np.bincount(preds, minlength=3).tolist(),
                "trial": np.bincount(trial_preds, minlength=3).tolist(),
            },
        }

    write_json(os.path.join(out_dir, "micro_overfit.json"), micro_results)
    report_lines = [
        "# Micro Overfit Report",
        "",
        "status = OK",
        f"class_counts_train = {class_counts}",
        f"micro_trials = {micro_selection['all']}",
    ]
    for mode in ["none", "instancenorm"]:
        res = micro_results.get(mode, {})
        report_lines.append(f"## Variant: {mode}")
        report_lines.append(f"- micro_win_acc: {res.get('micro_win_acc')}")
        report_lines.append(f"- micro_trial_acc: {res.get('micro_trial_acc')}")
        report_lines.append(f"- loss_start: {res.get('loss_start')}")
        report_lines.append(f"- loss_end: {res.get('loss_end')}")
        report_lines.append(f"- loss_min: {res.get('loss_min')}")
        report_lines.append(f"- micro_pred_hist: {res.get('micro_pred_hist')}")
        report_lines.append("")
    with open(os.path.join(out_dir, "micro_overfit_report.md"), "w") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
