import os
import json
import pandas as pd
import glob

output_dir = "promoted_results/phase14r/step6/seed1"
seed0_dir = os.path.join(output_dir, "seed0")

results = []

# Walk through all variant folders in seed0
for variant_dir in glob.glob(os.path.join(seed0_dir, "*")):
    if not os.path.isdir(variant_dir):
        continue
    
    metrics_path = os.path.join(variant_dir, "metrics.json")
    run_meta_path = os.path.join(variant_dir, "run_meta.json")
    
    if not os.path.exists(metrics_path) or not os.path.exists(run_meta_path):
        continue
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    with open(run_meta_path, "r") as f:
        meta = json.load(f)
        
    # Combine info
    row = {
        "seed": meta.get("seed"),
        "window_sec": meta.get("window"),
        "cov_est": meta.get("cov_est"),
        "clf": meta.get("clf"),
        "C": meta.get("C"),
        "agg": meta.get("agg"),
        "win_acc": metrics.get("win_acc"),
        "trial_acc": metrics.get("trial_acc"),
        "collapsed": metrics.get("collapse_trial", {}).get("collapsed", False),
        "converged": metrics.get("converged", True)
    }
    results.append(row)

if not results:
    print("No results found.")
    exit(1)

df = pd.DataFrame(results)

# Save summary
summary_path = os.path.join(output_dir, "summary.csv")
df.to_csv(summary_path, index=False)
print(f"Saved summary to {summary_path}")

# Find best
best_idx = df["trial_acc"].idxmax()
best_row = df.loc[best_idx]

print("\nTop 10 Variants:")
print(df.sort_values("trial_acc", ascending=False).head(10).to_string())

print(f"\nBest Variant: \n{best_row}")
