import os
import subprocess
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.datasets import AEON_FIXED_SPLIT_SPECS

# --- Configuration (High-Density MBA Zone) ---
MODELS = ["resnet1d", "patchtst", "timesnet"]
DATASETS = sorted(list(AEON_FIXED_SPLIT_SPECS.keys()))
GPUS = [2, 3] # MBA Zone: GPU 2 & 3 (NUMA 1)
WORKERS_PER_GPU = 10 # High density within zone
MAX_WORKERS = len(GPUS) * WORKERS_PER_GPU
OUT_DIR = "results/isolated_sweep_v1"

# NUMA 1 Core Affinity: 52-103,156-207 (104 logical cores)
AFFINITY = "52-103,156-207"

# Map model to its host-config
HOST_CONFIGS = {
    "resnet1d": "resnet1d_default",
    "patchtst": "patchtst_default",
    "timesnet": "timesnet_default"
}

def run_single_task(dataset, model, gpu_id):
    """Executes a single (dataset, model) sweep on a specific GPU with NUMA isolation."""
    host_cfg = HOST_CONFIGS[model]
    
    # 1. Env lock for CUDA visibility and internal threading
    # 20 workers * 4 threads = 80 cores (approx 80% of NUMA 1's 104 cores)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2,3" 
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["VECLIB_MAXIMUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    
    # 2. Command wrapper lock for Core Affinity (Visible in ps -ef)
    cmd = [
        "taskset", "-c", AFFINITY,
        "conda", "run", "-n", "pia", "python", "run_mba_pilot.py",
        "--dataset", dataset,
        "--model", model,
        "--host-config", host_cfg,
        "--device", f"cuda:{gpu_id}", 
        "--out-root", f"{OUT_DIR}/{model}/{dataset}",
        "--seeds", "1,2,3"
    ]
    
    print(f"[START] Model: {model} | Dataset: {ds} | Device: cuda:{gpu_id} | Core_Mask: {AFFINITY}")
    start_time = time.time()
    
    try:
        log_dir = f"{OUT_DIR}/logs/{model}"
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/{dataset}.log", "w") as log_file:
            subprocess.check_call(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        
        elapsed = time.time() - start_time
        print(f"[DONE]  Model: {model} | Dataset: {dataset} | Time: {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Model: {model} | Dataset: {dataset} | Failed with exit code {e.returncode}")
        return False

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    
    tasks = []
    for model in MODELS:
        for ds in DATASETS:
            # Check if final_results.csv already exists to allow resumption
            if not os.path.exists(f"{OUT_DIR}/{model}/{ds}/final_results.csv"):
                tasks.append((ds, model))
    
    print(f"Pending Tasks: {len(tasks)} | Concurrency: {MAX_WORKERS} (20 workers x 4 threads) | Zone: NUMA 1")
    
    results_status = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i, (ds, model) in enumerate(tasks):
            gpu_id = GPUS[i % len(GPUS)]
            future = executor.submit(run_single_task, ds, model, gpu_id)
            futures[future] = (ds, model)
        
        for future in as_completed(futures):
            ds, model = futures[future]
            try:
                success = future.result()
                results_status.append({"dataset": ds, "model": model, "success": success})
            except Exception as e:
                print(f"[CRASH] Task {model}/{ds} crashed: {e}")

    print("\n" + "="*30)
    print("ISOLATED HIGH-DENSITY SWEEP COMPLETED")
    print("="*30)
    
    # Consolidate Results
    all_dfs = []
    for model in MODELS:
        for ds in DATASETS:
            csv_path = f"{OUT_DIR}/{model}/{ds}/final_results.csv"
            if os.path.exists(csv_path):
                all_dfs.append(pd.read_csv(csv_path))
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(f"{OUT_DIR}/combined_sweep_results.csv", index=False)
        print(f"Final combined results saved to: {OUT_DIR}/combined_sweep_results.csv")
