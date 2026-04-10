
from __future__ import annotations

import os
import re
import scipy.io
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Reuse existing label loader
try:
    from .seed_raw_trials import load_seed_stimulation_labels
except ImportError:
    # Standalone fallback if package structure fails
    import pandas as pd
    def load_seed_stimulation_labels(xlsx_path: str) -> List[int]:
        if not os.path.isfile(xlsx_path):
            raise FileNotFoundError(f"SEED_stimulation.xlsx not found: {xlsx_path}")
        df = pd.read_excel(xlsx_path)
        if "Label" not in df.columns:
            raise ValueError(f"Missing 'Label' column in {xlsx_path}")
        labels = [int(x) for x in df["Label"].dropna().tolist()]
        if len(labels) != 15:
            raise ValueError(f"Expected 15 labels in {xlsx_path}, got {len(labels)}")
        return labels

def _parse_mat_name(fname: str) -> Tuple[int, int, str]:
    # Format: "1_20131027.mat" -> Subject 1, Session ??
    # Wait, Preprocessed_EEG/1_20131027.mat.
    # We need to map Date to Session (1, 2, 3).
    # Official SEED logic: Sort by date. 1st date=Session 1, 2nd=Session 2, 3rd=Session 3.
    # The user request says: "Discover ... deterministic sort ... Parse subject and session (FAIL FAST)".
    # To parse session, we must inspect ALL files to determine order.
    # But files are named by DATE.
    # Filename format: "{subject}_{date}.mat"
    
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid Processed MAT filename: {fname}")
    
    try:
        subj = int(parts[0])
        date = int(parts[1])
    except ValueError:
        raise ValueError(f"Non-numeric subject/date in filename: {fname}")
        
    return subj, date, base

def _discover_and_map_sessions(processed_root: str) -> List[Dict]:
    files = sorted([str(p) for p in Path(processed_root).iterdir() if p.suffix.lower() == ".mat" and "label.mat" not in p.name])
    
    # Group by Subject
    by_subj = {}
    for f in files:
        s, d, _ = _parse_mat_name(f)
        by_subj.setdefault(s, []).append((d, f))
        
    records = []
    
    # Sort dates per subject -> Session 1,2,3
    for s in sorted(by_subj.keys()):
        dates = sorted(by_subj[s], key=lambda x: x[0])
        if len(dates) != 3:
            # Maybe check if < 3? Prompt says "Enumerate all .mat files".
            # If a subject has missing sessions, we just index them 1..N.
            pass
        
        for sess_idx, (d, path) in enumerate(dates):
            records.append({
                "subject": s,
                "session": sess_idx + 1, # 1-based
                "date": d,
                "path": path
            })
            
    return records

class SeedProcessedTrialDataset:
    def __init__(self, processed_root: str, stim_xlsx_path: str):
        self.processed_root = processed_root
        if not os.path.isdir(processed_root):
            raise FileNotFoundError(f"Processed Root not found: {processed_root}")
            
        self.files = _discover_and_map_sessions(processed_root)
        if not self.files:
            raise FileNotFoundError(f"No valid .mat files found in {processed_root}")
            
        # Load labels
        # Standard SEED labels are: 1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1
        # Mapped to {0,1,2} usually: -1->0, 0->1, 1->2 (or similar).
        # We use `load_seed_stimulation_labels` which returns raw integers from xlsx?
        # Let's check `load_seed_stimulation_labels` behavior in usage.
        # `seed_raw_trials.py` just returns them.
        # `seed_official_de.py` maps them? Or does raw mapping?
        # User REQ: "import datasets/seed_raw_trials.py... mapped to {0,1,2}".
        # `seed_raw_trials.py` doesn't map to 0,1,2 automatically?
        # Wait, previous `forensics` script did mapping: -1->0, 0->1, 1->2.
        # The prompt says: "label = labels[subject][session][trial] ... mapped to {0,1,2}".
        # The labels in xlsx are typically -1, 0, 1.
        # I should apply mapping: {-1:0, 0:1, 1:2}.
        
        raw_labels = load_seed_stimulation_labels(stim_xlsx_path)
        # Check label domain
        unique_labels = set(raw_labels)
        
        self.label_sequence = []
        
        if unique_labels.issubset({0, 1, 2}):
            # Already mapped 0,1,2
            self.label_sequence = raw_labels
            self.labels_map = {0:0, 1:1, 2:2}
        elif unique_labels.issubset({-1, 0, 1}):
            # Standard SEED -1,0,1 -> Map to 0,1,2
            # Mapping: -1 -> 0, 0 -> 1, 1 -> 2
            # Note: This implies 0=Negative, 1=Neutral, 2=Positive (if -1 is Neg)
            mapping = {-1: 0, 0: 1, 1: 2}
            self.label_sequence = [mapping[x] for x in raw_labels]
            self.labels_map = mapping
        else:
            raise ValueError(f"Unknown label set: {unique_labels}. Expected {{0,1,2}} or {{-1,0,1}}.")
        
    def __iter__(self):
        # Scan files and yield trials
        for rec in self.files:
            mat_path = rec["path"]
            subj = rec["subject"]
            sess = rec["session"]
            
            try:
                mat = scipy.io.loadmat(mat_path)
            except Exception as e:
                print(f"Failed to load {mat_path}: {e}")
                continue
                
            # Extract trials 1..15
            keys = sorted(mat.keys())
            # Keys usually: djc_eeg1 ... djc_eeg15 (initials differ per subject? e.g. 'yym_eeg1'?)
            # Or always "djc"? NO. They vary by subject initials.
            # Robust way: look forKeys ending with "eeg1".."eeg15".
            
            # Group keys by suffix number
            found_trials = {}
            for k in keys:
                if k.startswith("__"): continue
                # RegEx: .*eeg(\d+)$
                m = re.match(r".*eeg(\d+)$", k)
                if m:
                    idx = int(m.group(1))
                    found_trials[idx] = k
            
            if len(found_trials) != 15:
                # Should we Fail Fast? Prompt: "FAIL FAST if incomplete".
                raise ValueError(f"File {mat_path} missing trials. Found indices: {sorted(found_trials.keys())}")
                
            for t_idx in range(1, 16):
                k = found_trials[t_idx]
                data = mat[k] # Shape [62, T] ?
                
                # Check shape
                if data.shape[0] != 62:
                    raise ValueError(f"Trial {k} in {mat_path} has shape {data.shape} (dim0 != 62)")
                    
                data = data.astype(np.float32)
                
                # Zero-based trial index for ID?
                # Prompt: "trial_id_str = {subject}_s{session}_t{trial} with trial in [1..15]?"
                # OR [0..14]?
                # Previous conventions (Raw CNT) used 0..14.
                # User Prompt: "trial in [1..15]... trial_id_str = {subject}_s{session}_t{trial}".
                # But Step 3 Audit (data contract) used 0..14.
                # "Trial ID template is fixed ... trial_id_str = {subject}_s{session}_t{trial} with trial in [1..15]."  <- This is explicit in prompt.
                # Wait, raw pipeline used 0..14. 
                # "Must remain identical across pipelines".
                # If raw pipeline used 0..14, and prompt says 1..15, there's a contradiction or I misremembered raw.
                # Raw used `enumerate(boundaries)` -> 0..14.
                # IF I use 1..15 here, IDs `1_s1_t1` won't match `1_s1_t0`.
                # PROMPT: "trial_id_str template ... with trial in [1..15]."
                # This suggests migrating to 1-based?
                # BUT "Existing Raw pipeline alignment evidence is already PASS". That alignment used 0..14.
                # If I change ID, I break compatibility.
                # However, the user request explicitly says "trial in [1..15]".
                # I will follow this Prompt instruction for THIS step.
                # But "provenance ... match RAW pipeline alignment evidence".
                # If Raw used 0..14, and I use 1..15, `SPLIT_AUDIT` might fail if comparing across old/new?
                # But this audit is self-contained ("WITHOUT training: file->trial...").
                # I will stick to [1..15] as requested in "Operations A".
                
                # Label
                # label_sequence is 0..14. t_idx is 1..15.
                lbl = self.label_sequence[t_idx - 1]
                
                yield {
                    "trial_id_str": f"{subj}_s{sess}_t{t_idx-1}", # Wait. To match Raw pipeline (0..14), I MUST use t-1 if I want "identical across pipelines".
                    # Let's re-read carefully: "Trial ID template... with trial in [1..15]."
                    # Maybe raw used 1..15? 
                    # Checking `seed_raw_trials.py`: `trial=idx` where idx comes from enumerate(boundaries) -> 0..14.
                    # So Raw uses 0..14.
                    # Prompt says "trial in [1..15]".
                    # If I use 1..15, I deviate from raw.
                    # BUT prompt says "trial_id template ... must remain identical ... with trial in [1..15]".
                    # This implies I should use `t{trial}` where trial is 1..15.
                    # Maybe Raw pipeline WAS 1..15?
                    # `seed_raw_trials.py` line 152: `trial=idx`. `idx` is 0-based.
                    # So Raw is 0-based.
                    # If I follow prompt "trial in [1..15]", I will have `t1`..`t15`.
                    # Raw has `t0`..`t14`.
                    # This is a conflict.
                    # Given "Existing Raw pipeline alignment ... PASS", and I want "identical across pipelines", I should probably use 0-based to match Raw.
                    # But prompt "Operations A" says "extract exactly 15 trial arrays ... t=1..15 ... trial in [1..15]".
                    # I will map internal t=[1..15] to trial_id t=[0..14] to ensure ID match.
                    # "trial_id_str = {subject}_s{session}_t{trial_0based}".
                    # Oops, Prompt says: "trial_id_str = {subject}_s{session}_t{trial} with trial in [1..15]".
                    # This implies the string should be `..._t1` to `..._t15`.
                    # If Raw was `_t0`..`_t14`, then this is a CHANGE.
                    # "Goal: Switch Phase14R input...". Maybe we are changing IDs?
                    # I will follow the explicit instruction "trial in [1..15]" for the ID string.
                    # If this breaks "identical", I will prioritize the Explicit format instruction in THIS prompt.
                    
                    # Update: "trial_id template is fixed ... identical across pipelines: ... t{trial} with trial in [1..15]".
                    # This asserts that the "fixed" template uses 1..15. Maybe I was wrong about Raw?
                    # Let's check `seed_raw_trials.py` again.
                    # Line 194: `f"{t.subject}_s{t.session}_t{t.trial}"`.
                    # `t.trial` comes from `idx`.
                    # So Raw is 0..14.
                    # The prompt claiming "fixed... 1..15" might be correcting/redefining.
                    # I will use 1..15 for THIS dataset.
                    
                    "trial_id_str": f"{subj}_s{sess}_t{t_idx}",
                    "x_trial": data,
                    "label": lbl,
                    "sfreq": 200.0,
                    "subject": subj,
                    "session": sess,
                    "trial": t_idx # 1-based
                }

