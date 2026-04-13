
import os
import json
import numpy as np
import pandas as pd

BASE_DIR = "promoted_results/phase13"

def main():
    print(f"Aggregating Results from {BASE_DIR}")
    
    seeds = [0, 1, 2, 3, 4]
    results = []
    
    for s in seeds:
        summary_path = os.path.join(BASE_DIR, f"seed{s}", "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
                data['seed'] = s
                results.append(data)
        else:
            print(f"Warning: Results for Seed {s} not found.")
            
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    
    # Calculate Stats
    cols = ["spatial_acc", "manifold_acc", "fusion_v0_acc", "fusion_v1_acc", "agree_rate", "conflict_rate"]
    
    print("\n=== Phase 13 Results Summary (n={}) ===".format(len(df)))
    # print(df.set_index("seed")[cols].to_markdown()) # Requires tabulate
    print(df.set_index("seed")[cols].to_string())
    
    print("\n=== Statistics ===")
    stats = df[cols].agg(["mean", "std"]).T
    stats["str"] = stats.apply(lambda x: f"{x['mean']*100:.2f}% ± {x['std']*100:.2f}%", axis=1)
    # print(stats["str"].to_markdown())
    print(stats["str"].to_string())
    
    # Export Report
    report_path = os.path.join(BASE_DIR, "final_report_raw.md")
    with open(report_path, "w") as f:
        f.write("# Phase 13 Final Report\n\n")
        f.write("## Per-Seed Results\n")
        # Manual MD Table
        f.write("| Seed | " + " | ".join(cols) + " |\n")
        f.write("|---| " + " | ".join(["---"]*len(cols)) + " |\n")
        for idx, row in df.set_index("seed")[cols].iterrows():
             row_str = " | ".join([f"{x:.4f}" for x in row])
             f.write(f"| {idx} | {row_str} |\n")
             
        f.write("\n\n## Aggregate Statistics\n")
        # f.write(stats["str"].to_markdown())
        f.write("| Metric | Mean ± Std |\n|---|---|\n")
        for idx, val in stats["str"].items():
            f.write(f"| {idx} | {val} |\n")
    
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
