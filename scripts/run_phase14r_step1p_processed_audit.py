
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_processed_trials import SeedProcessedTrialDataset
from datasets.seed_processed_windows import SeedProcessedWindowDatasetRefined
from datasets.seed_raw_trials import load_seed_time_points

CONFIG = {
    "seed": 0,
    "processed_root": "data/SEED/SEED_EEG/Preprocessed_EEG",
    "stim_xlsx": "data/SEED/SEED_EEG/SEED_stimulation.xlsx",
    "time_txt": "data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt",
    "window_sec": 4.0,
    "hop_sec": 1.0,
    "out_root": "promoted_results/phase14r/step1p/seed1/seed0/processed_audit"
}

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    # Convert numpy types
    def default(o):
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)): return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)): return float(o)
        elif isinstance(o, (np.ndarray,)): return o.tolist()
        raise TypeError
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)

def _deterministic_split(items, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(items))
    n_train = int(0.8 * len(items))
    train = [items[i] for i in indices[:n_train]]
    test = [items[i] for i in indices[n_train:]]
    return train, test

def main():
    cfg = CONFIG
    out_dir = cfg["out_root"]
    ensure_dir(out_dir)
    
    print("Layer A: File/Trial Audit...")
    ds_trials = SeedProcessedTrialDataset(cfg["processed_root"], cfg["stim_xlsx"])
    
    trials_all = []
    for t in ds_trials:
        trials_all.append(t)
    
    trials_all.sort(key=lambda x: x["trial_id_str"])
    
    n_trials = len(trials_all)
    print(f"Total Trials: {n_trials}")
    if n_trials != 675:
        raise RuntimeError(f"Expected 675 trials, got {n_trials}")
        
    lbl_counts = Counter([t["label"] for t in trials_all])
    print(f"Labels: {lbl_counts}")
    if len(lbl_counts) != 3:
        raise RuntimeError(f"Missing labels. Found: {list(lbl_counts.keys())}")
        
    # Layer A3: Alignment Check (Duration)
    print("Layer A3: Duration Alignment Check...")
    # Load raw boundaries from time.txt (assuming samples@1000)
    # The `load_seed_time_points` returns start/end lists.
    # We iterate them 15 times for each subject/session.
    # Note: `load_seed_time_points` returns ONE list of 15 trials.
    # SEED convention: same 15 time points for ALL sessions?
    # Yes, typically time.txt is global or per-session (but file is single).
    # We assume global template applied to all.
    
    s_pts, e_pts = load_seed_time_points(cfg["time_txt"])
    raw_Dur_sec_template = [(e - s)/1000.0 for s, e in zip(s_pts, e_pts)]
    
    alignment_logs = []
    max_diff = 0.0
    
    for t in trials_all:
        t_idx_1based = t["trial"]
        # raw template is 0-based index
        raw_dur = raw_Dur_sec_template[t_idx_1based - 1]
        
        proc_samples = t["x_trial"].shape[1]
        proc_dur = proc_samples / 200.0
        
        diff = abs(raw_dur - proc_dur)
        max_diff = max(max_diff, diff)
        
        alignment_logs.append({
            "trial_id": t["trial_id_str"],
            "raw_dur": raw_dur,
            "proc_dur": proc_dur,
            "diff": diff
        })
        
    bad_align = [x for x in alignment_logs if x["diff"] > 2.0]
    bad_align.sort(key=lambda x: x["diff"], reverse=True)
    
    print(f"Max Duration Diff: {max_diff:.4f}s")
    if max_diff > 2.0:
        print("FAIL FAST: Duration Mismatch > 2.0s")
        print(bad_align[:5])
        write_json(f"{out_dir}/DURATION_FAIL.json", bad_align)
        raise RuntimeError("Duration Alignment Check Failed")
        
    write_json(f"{out_dir}/DURATION_ALIGNMENT.json", {
        "max_abs_diff_sec": max_diff,
        "top10_diffs": bad_align[:10] if bad_align else alignment_logs[:10] # just logs sorted by diff
    })
    
    # Write Trial Table
    rows = []
    for x in alignment_logs: # merge info
        # find matching trial
        t = next(tr for tr in trials_all if tr["trial_id_str"] == x["trial_id"])
        # estimate n_windows
        ns = t["x_trial"].shape[1]
        w = int(cfg["window_sec"] * 200)
        h = int(cfg["hop_sec"] * 200)
        nw = (ns - w)//h + 1 if ns >= w else 0
        
        rows.append({
            "trial_id_str": t["trial_id_str"],
            "label": t["label"],
            "T_samples": ns,
            "duration_sec": x["proc_dur"],
            "n_windows_est": nw
        })
    pd.DataFrame(rows).to_csv(f"{out_dir}/TRIAL_TABLE.csv", index=False)
    
    # Layer B: Windows Audit
    print("Layer B: Window Audit...")
    
    # Split
    train_trials, test_trials = _deterministic_split(trials_all, cfg["seed"])
    tr_ids = set(t["trial_id_str"] for t in train_trials)
    te_ids = set(t["trial_id_str"] for t in test_trials)
    
    if len(tr_ids.intersection(te_ids)) > 0:
        raise RuntimeError("Split Intersection Detected")
    if len(train_trials) != 540 or len(test_trials) != 135:
        raise RuntimeError(f"Split sizes incorrect: {len(train_trials)}/{len(test_trials)}")
        
    write_json(f"{out_dir}/SPLIT_AUDIT.json", {
        "n_train": len(train_trials),
        "n_test": len(test_trials),
        "intersection": 0,
        "classes_train": list(Counter([t["label"] for t in train_trials]).keys())
    })
        
    # Window Dataset
    ds_win_train = SeedProcessedWindowDatasetRefined(train_trials, cfg["window_sec"], cfg["hop_sec"])
    ds_win_test = SeedProcessedWindowDatasetRefined(test_trials, cfg["window_sec"], cfg["hop_sec"])
    
    print(f"Windows Train: {len(ds_win_train)}")
    print(f"Windows Test: {len(ds_win_test)}")
    
    with open(f"{out_dir}/COUNTS.md", "w") as f:
        f.write("# Counts Table\n")
        f.write(f"- Trials: {n_trials} (Train {len(train_trials)}, Test {len(test_trials)})\n")
        f.write(f"- Windows Train: {len(ds_win_train)}\n")
        f.write(f"- Windows Test: {len(ds_win_test)}\n")
    
    # Layer C: DataLoader Probe
    dl = DataLoader(ds_win_train, batch_size=8, shuffle=False)
    batch = next(iter(dl))
    
    print("Batch Probe:", batch.keys())
    x_b = batch["x_win"]
    print("X Shape:", x_b.shape)
    
    if x_b.shape[1:] != (62, 800):
        raise RuntimeError(f"Batch Shape Mismatch: {x_b.shape}")
        
    write_json(f"{out_dir}/SAMPLE_BATCH.json", {
        "x_shape": list(x_b.shape),
        "labels": batch["label"].tolist(),
        "trial_ids": batch["trial_id_str"]
    })
    
    # Source of Truth
    write_json(f"{out_dir}/SOURCE_OF_TRUTH.json", {
        "processed_root": cfg["processed_root"],
        "sfreq": 200,
        "window_sec": cfg["window_sec"],
        "hop_sec": cfg["hop_sec"],
        "n_trials": 675,
        "split_seed": cfg["seed"]
    })

    print("Audit Complete. Success.")

if __name__ == "__main__":
    main()
