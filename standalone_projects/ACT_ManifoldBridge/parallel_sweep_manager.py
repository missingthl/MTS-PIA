import os
import subprocess
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.datasets import AEON_FIXED_SPLIT_SPECS

# --- Configuration (Theory Atlas RECOVERY) ---
MODELS = ["patchtst", "timesnet"]
DATASETS = sorted(list(AEON_FIXED_SPLIT_SPECS.keys()))
PHYSICAL_GPUS = [2, 3] # ACT Zone: GPU 2 & 3 (NUMA 1)
WORKERS_PER_GPU = 10 # High density within zone
MAX_WORKERS = len(PHYSICAL_GPUS) * WORKERS_PER_GPU
OUT_DIR = "results/paper_theory_atlas_v1"

# NUMA 1 Core Affinity: 52-103,156-207 (104 logical cores)
AFFINITY = "52-103,156-207"

# Map model to its host-config
HOST_CONFIGS = {
    "resnet1d": "resnet1d_default",
    "patchtst": "patchtst_default",
    "timesnet": "timesnet_default"
}

def run_single_task(dataset, model, physical_gpu_id):
    """Executes a single (dataset, model) sweep on a specific GPU with NUMA isolation."""
    host_cfg = HOST_CONFIGS[model]
    
    env = os.environ.copy()
    # Lock this process to only see ONE physical GPU. 
    # That GPU will be indexed as cuda:0 inside this process.
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id) 
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["VECLIB_MAXIMUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    
    cmd = [
        "taskset", "-c", AFFINITY,
        "conda", "run", "-n", "pia", "python", "run_act_pilot.py",
        "--dataset", dataset,
        "--model", model,
        "--host-config", host_cfg,
        "--device", "cuda:0", # Always 0 because of CUDA_VISIBLE_DEVICES restriction
        "--out-root", f"{OUT_DIR}/{model}/{dataset}",
        "--seeds", "1",
        "--multiplier", "1",
        "--theory-diagnostics"
    ]
    
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
    # Cleanup only for the recovery models was done manually.
    # We allow resumption here in case of interruption.
    os.makedirs(OUT_DIR, exist_ok=True)
    
    tasks = []
    for model in MODELS:
        for ds in DATASETS:
            tasks.append((ds, model))
    
    print(f"Theory Atlas (FIXED): {len(tasks)} tasks | Concurrency: {MAX_WORKERS} | NUMA 1")
    
    results_status = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i, (ds, model) in enumerate(tasks):
            # Alternate between physical GPUs 2 and 3
            p_gpu = PHYSICAL_GPUS[i % len(PHYSICAL_GPUS)]
            future = executor.submit(run_single_task, ds, model, p_gpu)
            futures[future] = (ds, model)
        
        for future in as_completed(futures):
            ds, model = futures[future]
            try:
                success = future.result()
                results_status.append({"dataset": ds, "model": model, "success": success})
            except Exception as e:
                print(f"[CRASH] Task {model}/{ds} crashed: {e}")

    print("\n" + "="*30)
    print("THEORY ATLAS SWEEP COMPLETED")
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
