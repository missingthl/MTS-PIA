import pandas as pd
import numpy as np
from pathlib import Path

root = Path("results/csta_external_baselines_local/resnet1d_s123")
subdirs = ["controls", "diffusionts", "timevqvae", "diffusionts_recovery"]

all_dfs = []
for sd in subdirs:
    p = root / sd / "per_seed_external.csv"
    if p.exists():
        all_dfs.append(pd.read_csv(p))

if not all_dfs:
    print("No results found.")
    exit()

df = pd.concat(all_dfs, ignore_index=True)
df = df[df["status"] == "success"]

# Group by dataset and method
summary = df.groupby(["dataset", "method"]).agg(
    f1_mean=("aug_f1", "mean"),
    gain_mean=("gain", "mean"),
    count=("seed", "count")
).reset_index()

# Pivot for better viewing
pivot_f1 = summary.pivot(index="dataset", columns="method", values="f1_mean")
pivot_gain = summary.pivot(index="dataset", columns="method", values="gain_mean")

print("--- Mean F1 per Dataset/Method ---")
print(pivot_f1.to_string())
print("\n--- Mean Gain (vs No-Aug) per Dataset/Method ---")
print(pivot_gain.to_string())

# Overall means
print("\n--- Overall Mean Gain Across Pilot7 ---")
print(pivot_gain.mean().sort_values(ascending=False).to_string())

# Save merged result
df.to_csv(root / "merged_pilot7_results.csv", index=False)
print(f"\nMerged results saved to {root / 'merged_pilot7_results.csv'}")
