import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Proposed Gamma Scan configuration
DATASETS = ["natops", "pendigits", "basicmotions"]
MODEL = "resnet1d"
HOST_CFG = "resnet1d_default"
MULTIPLIERS = [1, 2, 5, 10, 20, 50, 100]
OUT_ROOT = "results/gamma_scan_v1"

def run_gamma_task(ds, multiplier):
    cmd = [
        "conda", "run", "-n", "pia", "python", "run_act_pilot.py",
        "--dataset", ds,
        "--model", MODEL,
        "--host-config", HOST_CFG,
        "--device", "cuda:2",
        "--out-root", f"{OUT_ROOT}/{ds}/m{multiplier}",
        "--seeds", "1",
        "--multiplier", str(multiplier),
        "--theory-diagnostics"
    ]
    log_dir = f"{OUT_ROOT}/logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/{ds}_m{multiplier}.log", "w") as f:
        subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_results = []
    
    print(f"Starting Gamma Scan (Proposition 2 Evidence) on: {DATASETS}")
    for ds in DATASETS:
        for m in MULTIPLIERS:
            print(f"Running {ds} with Multiplier {m}...")
            run_gamma_task(ds, m)
            
            # Extract results
            csv_path = f"{OUT_ROOT}/{ds}/m{m}/{MODEL}/{ds}/final_results.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['scan_multiplier'] = m
                all_results.append(df)
    
    if all_results:
        final_df = pd.concat(all_results)
        final_df.to_csv(f"{OUT_ROOT}/gamma_scan_results.csv", index=False)
        
        # Plotting the Safety Activation Curve
        plt.figure(figsize=(10, 6))
        for ds in DATASETS:
            sub = final_df[final_df['dataset'] == ds].sort_values('scan_multiplier')
            plt.plot(sub['scan_multiplier'], sub['safe_radius_ratio_mean'], marker='o', label=f"{ds} (Safety Ratio)")
            
        plt.xlabel("Augmentation Multiplier (Gamma)")
        plt.ylabel("Safe Radius Ratio (Constraint Strength)")
        plt.title("Proposition 2: Activation of the Safe Region Constraint")
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig(f"{OUT_ROOT}/gamma_sensitivity_curve.png")
        print(f"Gamma Sensitivity Curve saved to {OUT_ROOT}/gamma_sensitivity_curve.png")
