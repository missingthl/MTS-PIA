
import os
import sys
import glob
import re
import json
import numpy as np
import pandas as pd
import mne
import scipy.io

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# Config
RAW_ROOT = "data/SEED/SEED_EEG/SEED_RAW_EEG"
STIM_PATH = "data/SEED/SEED_EEG/SEED_stimulation.xlsx"
DE_ROOT = "data/SEED/SEED_EEG/ExtractedFeatures_1s"
OUT_DIR = "promoted_results/phase14r/step0b/seed1"

WINDOW_SEC = 1.0
HOP_SEC = 1.0

def parse_time_txt(path):
    # Format: 
    # start_point_list = [27000,290000,...];
    # end_point_list = [262000,523000,...];
    
    with open(path, 'r') as f:
        content = f.read().strip()
        
    start_match = re.search(r"start_point_list\s*=\s*\[(.*?)\]", content, re.DOTALL)
    end_match = re.search(r"end_point_list\s*=\s*\[(.*?)\]", content, re.DOTALL)
    
    if not start_match or not end_match:
        # Fallback to simple number extraction if different format
        all_nums = [int(x) for x in re.findall(r'\d+', content)]
        if len(all_nums) == 30:
             starts = all_nums[0:15]
             ends = all_nums[15:30]
        else:
             print(f"Warning: Could not parse time.txt strictly. Found {len(all_nums)} numbers.")
             return []
    else:
        starts = [int(x) for x in re.findall(r'\d+', start_match.group(1))]
        ends = [int(x) for x in re.findall(r'\d+', end_match.group(1))]
        
    trials = []
    # Note: These valid points might be 1000Hz or 200Hz.
    # We will let the raw data check decide if scaling is needed (likely not scaled if direct from file).
    
    for i in range(len(starts)):
        trials.append({
            "trial_idx": i,
            "start_point": starts[i],
            "end_point": ends[i],
            "duration_points": ends[i] - starts[i]
        })
    return trials

def load_labels(path):
    try:
        df = pd.read_excel(path, engine='openpyxl')
    except Exception:
        df = pd.read_excel(path)
        
    if "Label" not in df.columns:
        found = False
        for c in df.columns:
            if str(c).lower() == "label":
                df = df.rename(columns={c: "Label"})
                found = True
                break
        if not found:
             pass

    # Drop NaNs
    df = df.dropna(subset=["Label"])
    labels = df["Label"].values.tolist()
    
    mapped = []
    for l in labels:
        try:
            val = int(l)
        except:
            continue
            
        if val == -1: mapped.append(0)
        elif val == 0: mapped.append(1)
        elif val == 1: mapped.append(2)
        else:
            mapped.append(val)
            
    if len(mapped) > 15:
        mapped = mapped[:15]
        
    return mapped

def main():
    ensure_dir(OUT_DIR)
    
    # A) Enumerate Raw Files
    cnt_files = glob.glob(os.path.join(RAW_ROOT, "*.cnt"))
    cnt_files.sort()
    
    raw_index = []
    for p in cnt_files:
        fname = os.path.basename(p)
        # Format: 1_1.cnt (Subject_Session)
        m = re.match(r"(\d+)_(\d+)\.cnt", fname)
        if m:
            subj = int(m.group(1))
            sess = int(m.group(2))
            raw_index.append({
                "subject": subj,
                "session": sess,
                "raw_path": p,
                "file_size_bytes": os.path.getsize(p)
            })
    
    pd.DataFrame(raw_index).to_csv(f"{OUT_DIR}/raw_files_index.csv", index=False)
    print(f"Index: Found {len(raw_index)} raw files.")
    
    # Time Boundaries
    time_txt = os.path.join(RAW_ROOT, "time.txt")
    boundaries = parse_time_txt(time_txt)
    with open(f"{OUT_DIR}/trial_boundaries.json", "w") as f:
        json.dump(boundaries, f, indent=2)
    print(f"Boundaries: Found {len(boundaries)} trials definition.")
    
    # Labels
    labels = load_labels(STIM_PATH)
    if len(labels) != 15:
        print(f"WARNING: Expected 15 labels, found {len(labels)}")
    
    # B) Build Trial Table (Raw)
    raw_trial_rows = []
    raw_window_rows = []
    
    failures = []
    
    for item in raw_index:
        subj = item["subject"]
        sess = item["session"]
        p = item["raw_path"]
        
        try:
            raw = mne.io.read_raw_cnt(p, preload=False, verbose="ERROR")
            fs = raw.info['sfreq']
            n_chan = len(raw.ch_names)
            
            # SCALING CHECK: 
            # If time.txt > raw.n_times significantly, apply scaling.
            # Usually time.txt is 1000Hz, raw is 200Hz -> Scale = 5.0
            # Let's detect.
            max_bound = max(b["end_point"] for b in boundaries)
            scaling_factor = 1.0
            if max_bound > raw.n_times:
                # Likely 1000 Hz vs 200 Hz
                if raw.n_times > 0 and max_bound / raw.n_times > 4:
                    scaling_factor = 1000.0 / fs # Roughly 5
            
            # Loop trials
            for t_idx, b in enumerate(boundaries):
                t_lbl = labels[t_idx] if t_idx < len(labels) else -999
                
                # Apply scaling
                start_pt = int(b["start_point"] / scaling_factor)
                end_pt = int(b["end_point"] / scaling_factor)
                
                dur_pt = end_pt - start_pt
                dur_sec = dur_pt / fs
                
                tid_str = f"{subj}_s{sess}_t{t_idx}"
                
                raw_trial_rows.append({
                    "trial_id_str": tid_str,
                    "subject": subj,
                    "session": sess,
                    "trial": t_idx,
                    "label": t_lbl,
                    "fs": fs,
                    "n_channels": n_chan,
                    "trial_n_samples": dur_pt,
                    "trial_duration_sec": dur_sec,
                    "raw_path": p
                })
                
                # C) Window Logic
                # Standard sliding window non-overlap
                # n_windows = floor(dur_sec / 1.0)
                
                w_size = int(WINDOW_SEC * fs)
                h_size = int(HOP_SEC * fs)
                
                if dur_pt < w_size:
                    n_win = 0
                else:
                    n_win = (dur_pt - w_size) // h_size + 1
                
                covered_sec = (n_win * h_size + (w_size - h_size)) / fs if n_win > 0 else 0
                
                raw_window_rows.append({
                    "trial_id_str": tid_str,
                    "label": t_lbl,
                    "window_sec": WINDOW_SEC,
                    "hop_sec": HOP_SEC,
                    "n_windows": n_win,
                    "first_start_sec": 0.0, 
                    "last_start_sec": (n_win-1)*HOP_SEC if n_win>0 else 0.0,
                    "covered_sec": covered_sec,
                    "trial_duration_sec": dur_sec # Helper for join
                })
                
        except Exception as e:
            print(f"Error processing {p}: {e}")
            failures.append(p)
            
    pd.DataFrame(raw_trial_rows).to_csv(f"{OUT_DIR}/raw_trial_table.csv", index=False)
    pd.DataFrame(raw_window_rows).to_csv(f"{OUT_DIR}/raw_window_table.csv", index=False)
    print(f"Processed {len(raw_trial_rows)} raw trial segments.")
    
    # D) DE Truth Table
    de_files = glob.glob(os.path.join(DE_ROOT, "*.mat"))
    de_index = []
    for p in de_files:
        fname = os.path.basename(p)
        m = re.match(r"(\d+)_(\d+)\.mat", fname)
        if m:
            subj = int(m.group(1))
            date_str = m.group(2)
            de_index.append({
                "subject": subj,
                "date": date_str,
                "path": p
            })
    
    de_index.sort(key=lambda x: (x["subject"], x["date"]))
    
    curr_subj = -1
    sess_ctr = 0
    for item in de_index:
        if item["subject"] != curr_subj:
            curr_subj = item["subject"]
            sess_ctr = 1
        else:
            sess_ctr += 1
        item["session"] = sess_ctr
    
    de_rows = []
    
    for item in de_index:
        mat = scipy.io.loadmat(item["path"])
        relevant_keys = [k for k in mat.keys() if re.match(r'^de_LDS\d+$', k)]
        
        def extract_num(k):
            m = re.search(r'(\d+)$', k)
            return int(m.group(1)) if m else 999
            
        relevant_keys.sort(key=extract_num)
        
        subj = item["subject"]
        sess = item["session"]
        
        # If mismatch # of keys? 15 usually.
        for k_idx, k in enumerate(relevant_keys):
            t_idx = k_idx 
            
            data = mat[k]
            sh = data.shape
            L = sh[1] if len(sh) > 1 else sh[0] 
            
            tid_str = f"{subj}_s{sess}_t{t_idx}"
            
            de_rows.append({
                "trial_id_str": tid_str,
                "label": -999,
                "de_L": L,
                "de_shape_str": str(sh),
                "de_path": item["path"]
            })
            
    de_df = pd.DataFrame(de_rows)
    if de_df["trial_id_str"].duplicated().any():
        print("WARNING: Duplicate DE trials found. Taking first.")
        de_df = de_df.drop_duplicates(subset=["trial_id_str"])
        
    de_df.to_csv(f"{OUT_DIR}/de_trial_table.csv", index=False)
    print(f"Loaded DE metadata for {len(de_df)} trials.")
    
    # E) Alignment
    rw = pd.read_csv(f"{OUT_DIR}/raw_window_table.csv")
    de = pd.read_csv(f"{OUT_DIR}/de_trial_table.csv")
    
    # Ensure join keys type match
    merged = pd.merge(rw, de, on="trial_id_str", suffixes=("_raw", "_de"))
    
    align_rows = []
    match_counts = {"EXACT": 0, "NEAR": 0, "MISMATCH": 0}
    
    for _, row in merged.iterrows():
        n_raw = row["n_windows"]
        n_de = row["de_L"]
        
        diff = n_raw - n_de
        
        status = "MISMATCH"
        if diff == 0:
            status = "EXACT"
        elif abs(diff) <= 1:
            status = "NEAR"
            
        match_counts[status] += 1
        
        align_rows.append({
            "trial_id_str": row["trial_id_str"],
            "trial_duration_sec": row["trial_duration_sec"],
            "raw_n_windows": n_raw,
            "de_L": n_de,
            "inferred_de_step_sec": row["trial_duration_sec"] / n_de if n_de > 0 else 0,
            "ratio": n_raw / n_de if n_de > 0 else 0,
            "alignment_status": status
        })
        
    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(f"{OUT_DIR}/alignment_report.csv", index=False)
    
    # F) Report
    total = len(align_df)
    pct_exact = match_counts["EXACT"] / total * 100 if total else 0
    pct_near = match_counts["NEAR"] / total * 100 if total else 0
    pct_mismatch = match_counts["MISMATCH"] / total * 100 if total else 0
    
    verdict = "INCORRECT"
    if pct_exact > 95: verdict = "CORRECT (High Exactness)"
    elif (pct_exact + pct_near) > 95: verdict = "CORRECT (With Edge Effects)"
    
    md_lines = []
    md_lines.append(f"# Phase 14R Step 0b: Raw-DE Alignment Forensics\n")
    md_lines.append(f"## Summary\n")
    md_lines.append(f"- **Verdict**: {verdict}")
    md_lines.append(f"- **Total Trials**: {total}")
    md_lines.append(f"- **Exact Match**: {match_counts['EXACT']} ({pct_exact:.2f}%)")
    md_lines.append(f"- **Near Match (+/-1)**: {match_counts['NEAR']} ({pct_near:.2f}%)")
    md_lines.append(f"- **Mismatch**: {match_counts['MISMATCH']} ({pct_mismatch:.2f}%)")
    md_lines.append(f"\n## Provenance")
    md_lines.append(f"- Labels derived from `{STIM_PATH}` (Mapped standard 0,1,2).")
    md_lines.append(f"- Windowing assumed: Window={WINDOW_SEC}s, Hop={HOP_SEC}s.")
    md_lines.append(f"\n## Mismatch Analysis")
    if pct_mismatch > 0:
        bad = align_df[align_df["alignment_status"] == "MISMATCH"]
        md_lines.append(f"Top Mismatches:\n")
        # Use to_string since tabulate is missing
        md_lines.append(bad.head(10).to_string(index=False))
    else:
        md_lines.append("None.")

    with open(f"{OUT_DIR}/DATA_PATH_FORENSICS.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print(f"Report generated: {OUT_DIR}/DATA_PATH_FORENSICS.md")
    
    if pct_mismatch > 5: 
        print(f"FAIL-FAST TRIGGERED: Mismatch {pct_mismatch:.2f}% > 5%")

if __name__ == "__main__":
    main()
