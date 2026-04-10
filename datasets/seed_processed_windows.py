
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class WindowMeta:
    trial_id_str: str
    win_start: int
    subject: int
    session: int
    trial: int
    label: int

class SeedProcessedWindowDataset:
    def __init__(self, trial_ds, window_sec: float = 4.0, hop_sec: float = 1.0):
        self.trial_ds = trial_ds
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.windows: List[WindowMeta] = []
        self.data_cache = {} # Cache trial data? Or keep ref to trial_ds if it iterates?
        # SeedProcessedTrialDataset iterates. It doesn't allow random access currently.
        # We need random access for DataLoader.
        
        # We must load all trials into memory (or index them).
        # SEED processed is ~47k points x 62 chans x 675 trials ~ 30M points ~ 8GB floats? 
        # 30M floats * 4 bytes = 120MB. Very small. We can cache all.
        
        self.trials = []
        print(f"[SeedProcessedWindowDataset] Loading trails from {trial_ds.processed_root}...")
        for t in trial_ds:
            self.trials.append(t)
            
        # Deterministic Sort
        self.trials.sort(key=lambda x: x["trial_id_str"])
        
        # Index windows
        for t_idx, t_data in enumerate(self.trials):
            sfreq = t_data["sfreq"]
            x = t_data["x_trial"]
            n_samples = x.shape[1]
            win_len = int(window_sec * sfreq)
            hop_len = int(hop_sec * sfreq)
            
            w_indices = []
            if n_samples < win_len:
                # FAIL FAST or skip? Prompt: "FAIL FAST: n_windows(trial) > 0"
                pass 
            else:
                n_wins = 0
                curr = 0
                while curr + win_len <= n_samples:
                    w_indices.append(curr)
                    curr += hop_len
                    n_wins += 1
            
            if not w_indices:
                raise ValueError(f"Trial {t_data['trial_id_str']} too short ({n_samples}) for window ({win_len})")
                
            for start in w_indices:
                self.windows.append(WindowMeta(
                    trial_id_str=t_data["trial_id_str"],
                    win_start=start,
                    subject=t_data["subject"],
                    session=t_data["session"],
                    trial=t_data["trial"],
                    label=t_data["label"]
                ))
    
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        meta = self.windows[idx]
        # Find which trial? We flattened windows.
        # We need efficient lookup.
        # Store index in meta? Or keep trials in list and store trial_idx in meta.
        # Let's optimize: self.trials is sorted list. 
        # But we don't have trial index in Meta.
        # Let's add trial_list_index.
        
        pass

    # Re-implement __init__ logic slightly to link back to self.trials
    # Actually just rebuild the list.
    
class SeedProcessedWindowDatasetRefined:
    def __init__(self, trial_ds, window_sec=4.0, hop_sec=1.0):
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        
        print("[ProcessedWins] Loading all trials...")
        self.trials = sorted(list(trial_ds), key=lambda x: x["trial_id_str"])
        
        self.window_map = []
        
        for t_idx, t in enumerate(self.trials):
            sfreq = t["sfreq"]
            x = t["x_trial"]
            n_samples = x.shape[1]
            win_len = int(window_sec * sfreq)
            hop_len = int(hop_sec * sfreq)
            
            w_starts = []
            if n_samples >= win_len:
                curr = 0
                while curr + win_len <= n_samples:
                    w_starts.append(curr)
                    curr += hop_len
            
            if not w_starts:
                raise ValueError(f"Trial {t['trial_id_str']} has 0 windows. Len={n_samples}, Req={win_len}")
                
            for start in w_starts:
                self.window_map.append((t_idx, start))
                
    def __len__(self):
        return len(self.window_map)
        
    def __getitem__(self, idx):
        t_idx, start = self.window_map[idx]
        t = self.trials[t_idx]
        
        sfreq = t["sfreq"]
        win_len = int(self.window_sec * sfreq)
        
        data = t["x_trial"][:, start : start+win_len]
        
        # Check assertions
        if data.shape != (62, win_len):
            raise RuntimeError(f"Window shape mismatch: {data.shape} vs {(62, win_len)}")
        if not np.isfinite(data).all():
             raise RuntimeError(f"NaN/Inf in window {t['trial_id_str']} start={start}")
             
        return {
            "trial_id_str": t["trial_id_str"],
            "win_start": start,
            "x_win": data,
            "label": t["label"]
        }
