
import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from datasets.seed_official_de import load_seed_official_de

def probe_semantics():
    out_dir = "experiments/phase8_rebaseline/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "time_step_probe.txt")
    
    seed_de_root = "data/SEED/SEED_EEG/ExtractedFeatures_1s"
    seed_de_var = "de_LDS1"
    manifest_path = "logs/seed_raw_trial_manifest_full.json"
    
    print(f"Loading SEED1 official DE from {seed_de_root}...")
    
    try:
        # load_seed_official_de returns: (train_x, train_y, test_x, test_y, trial_index, skipped)
        # But wait, load_seed_official_de typically flattens the data unless we inspect the internals.
        # It returns concatenated numpy arrays (N_samples, 310).
        # We want the TRIAL shape.
        
        # Actually, let's look at `load_seed_official_de` implementation (viewed partially).
        # It calls `_de_trial_to_samples` which flattens (C, T, B) -> (T, B*C).
        # So we lose the original (C, T, B) structure in the output of this function.
        # However, we can reconstruct it, OR we can hook into `_de_file_map`...
        # No, simpler: Use `load_seed_official_de` as is, take the first chunk corresponding to the first trial.
        # The `trial_index` list contains metadata including `n_windows`.
        
        trainx, trainy, testx, testy, trial_index, skipped = load_seed_official_de(
            seed_de_root=seed_de_root,
            seed_de_var=seed_de_var,
            manifest_path=manifest_path,
            freeze_align=True
        )
        
        # Get metadata for first trial
        first_trial_meta = trial_index[0]
        n_windows = first_trial_meta['n_windows']
        
        # Get first trial data segment
        # trainx is (N_total, 310)
        t0 = trainx[:n_windows] # (T_total, 310)
        
        # Write report
        with open(out_file, "w") as f:
            f.write(f"Dataset: SEED1\n")
            f.write(f"Source: {seed_de_root}\n")
            f.write(f"Variable: {seed_de_var}\n")
            f.write(f"Manifest: {manifest_path}\n")
            f.write(f"\n--- Shape Inspection (Trial 0) ---\n")
            f.write(f"Trial ID: {first_trial_meta['trial_id']}\n")
            f.write(f"Observed Flattened Shape: {t0.shape}\n")
            f.write(f"Inferred T_total (Axis 0): {n_windows}\n")
            f.write(f"Feature Dim (310): 62 Channels * 5 Bands\n")
            f.write(f"Original Structure implied: (62, {n_windows}, 5)\n")
            
            # Duration Inference
            duration_est = 240 # SEED clips are approx 4 mins
            step_sec = duration_est / n_windows
            
            f.write(f"\n--- Time Inference ---\n")
            f.write(f"Average SEED Movie Duration: ~{duration_est} seconds\n")
            f.write(f"Observed Time Steps (T): {n_windows}\n")
            f.write(f"Calculated Step Duration: {step_sec:.4f} seconds\n")
            
            path_hint = "1s" in seed_de_root
            f.write(f"Explicit '1s' in path: {path_hint}\n")
            
            conclusion = "1.0 second" if (0.9 < step_sec < 1.1) else f"{step_sec:.2f} seconds"
            f.write(f"\nConclusion: 1 DE step = {conclusion} (Inference)\n")

        print(f"Probe complete. Written to {out_file}")
            
    except Exception as e:
        with open(out_file, "w") as f:
            f.write(f"Probe Failed: {e}\n")
        print(f"Probe Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    probe_semantics()
