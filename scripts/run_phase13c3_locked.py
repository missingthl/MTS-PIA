
import os
import sys
import json
import subprocess
import pandas as pd
import time
from pathlib import Path

# --- Configuration ---
PHASE_ID = "phase13c3"
CONFIG_PATH = f"promoted_results/{PHASE_ID}/config_locked.json"
BASE_OUTPUT_DIR = f"promoted_results/{PHASE_ID}"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    t0 = time.time()
    config = load_config()
    seeds = config['seeds']
    
    summary_rows = []
    
    python_exe = sys.executable
    worker_script = "scripts/run_seed_worker_c3.py"
    
    for seed in seeds:
        print(f"\n###########################################")
        print(f"### ORCHESTRATOR: Processing Seed {seed} ###")
        print(f"###########################################")
        
        try:
            # 1. Spatial
            # print(f">>> Step 1: Spatial Inference")
            # cmd_s = [python_exe, worker_script, "--seed", str(seed), "--mode", "spatial"]
            # ret_s = subprocess.run(cmd_s, check=False)
            # if ret_s.returncode != 0:
            #     raise RuntimeError(f"Spatial Inference Failed (RC {ret_s.returncode})")
                
            # 2. Manifold
            # print(f">>> Step 2: Manifold Training")
            # cmd_m = [python_exe, worker_script, "--seed", str(seed), "--mode", "manifold"]
            # ret_m = subprocess.run(cmd_m, check=False)
            # if ret_m.returncode != 0:
            #     raise RuntimeError(f"Manifold Training Failed (RC {ret_m.returncode})")
                
            # 3. Fusion
            print(f">>> Step 3: Fusion & Audit")
            cmd_f = [python_exe, worker_script, "--seed", str(seed), "--mode", "fusion"]
            ret_f = subprocess.run(cmd_f, check=False)
            if ret_f.returncode != 0:
                raise RuntimeError(f"Fusion Audit Failed (RC {ret_f.returncode})")
                
            # Success
            print(f">>> ORCHESTRATOR: Seed {seed} PASS")
            # Load Stats
            stats_path = f"{BASE_OUTPUT_DIR}/seed1/seed{seed}/c3_worker_stats.json"
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                summary_rows.append({"seed": seed, "status": "PASS", **stats})
            else:
                summary_rows.append({"seed": seed, "status": "FAIL", "error": "Stats Missing"})

        except Exception as e:
            print(f">>> ORCHESTRATOR: Failed Seed {seed}: {e}")
            summary_rows.append({"seed": seed, "status": "FAIL", "error": str(e)})

    # --- Final Summary ---
    print("\n###########################################")
    print("\n>>> ORCHESTRATOR: Aggregating Results <<<")
    df = pd.DataFrame(summary_rows)
    df.to_csv(f"{BASE_OUTPUT_DIR}/seed1/summary.csv", index=False)
    
    # Markdown
    md_path = f"{BASE_OUTPUT_DIR}/seed1/PHASE13C3_SUMMARY.md"
    with open(md_path, "w") as f:
        f.write("# Phase 13C-3 Summary: 5-Seed Validation\n\n")
        f.write("## Aggregate Metrics\n")
        if not df.empty:
            f.write(f"| {' | '.join(df.columns)} |\n")
            f.write(f"| {' | '.join(['---']*len(df.columns))} |\n")
            for _, row in df.iterrows():
                f.write(f"| {' | '.join(str(x) for x in row.values)} |\n")
            f.write("\n")
            
            pass_df = df[df['status'] == 'PASS']
            if not pass_df.empty:
                f.write("## Mean Stats (Passed Seeds)\n")
                num_cols = pass_df.select_dtypes(include=[float, int]).columns
                desc = pass_df[num_cols].agg(['mean', 'std']).T
                
                f.write(f"| Metric | Mean | Std |\n")
                f.write(f"| --- | --- | --- |\n")
                for idx, row in desc.iterrows():
                    f.write(f"| {idx} | {row['mean']:.4f} | {row['std']:.4f} |\n")
    
    print(df.to_string())
    print(f"Total Time: {time.time() - t0:.1f}s")
    
    n_fail = len(df[df['status'] == 'FAIL'])
    sys.exit(2 if n_fail > 0 else 0)
