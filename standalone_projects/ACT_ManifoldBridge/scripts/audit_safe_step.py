import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Audit Safe-Step and Gamma usage from CSTA results.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the sweep results.")
    parser.add_argument("--output", type=str, default="safe_step_audit.csv", help="Output filename for the audit.")
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.is_dir():
        print(f"Error: {root_path} is not a directory.")
        return

    # Find all result CSVs
    csv_files = list(root_path.glob("**/*_results.csv"))
    if not csv_files:
        print(f"No results.csv files found in {root_path}")
        return

    print(f"Found {len(csv_files)} result files. Loading...")
    
    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Add some context if missing
            if "method" not in df.columns:
                # Infer method from path: .../_csta_runs/{method}/{dataset}/s{seed}/...
                parts = f.parts
                if "_csta_runs" in parts:
                    idx = parts.index("_csta_runs")
                    df["method"] = parts[idx+1]
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Filter for successful runs
    if "status" in full_df.columns:
        full_df = full_df[full_df["status"] == "success"]

    # Key diagnostic columns requested by user
    diag_cols = [
        "gamma_requested_mean",
        "gamma_used_mean",
        "safe_radius_ratio_mean",
        "safe_clip_rate",
        "gamma_zero_rate",
        "manifold_margin_mean",
        "transport_error_logeuc_mean",
        "gain"
    ]

    # Ensure all columns exist
    for col in diag_cols:
        if col not in full_df.columns:
            full_df[col] = np.nan

    # Group by Dataset and Method
    group_cols = ["dataset", "method"]
    summary = full_df.groupby(group_cols)[diag_cols].mean().reset_index()

    # Sort for readability
    summary = summary.sort_values(["dataset", "method"])

    # Print summary to console
    print("\n--- Safe-Step Audit Summary (Mean across seeds) ---")
    print(summary.to_string(index=False))

    # Save to CSV
    summary.to_csv(args.output, index=False)
    print(f"\nAudit results saved to {args.output}")

    # Additional Analysis: Datasets with high clipping or zero-rate
    print("\n--- High Clipping / Zero-Rate Watchlist ---")
    watchlist = summary[(summary["safe_clip_rate"] > 0.5) | (summary["gamma_zero_rate"] > 0.1)]
    if watchlist.empty:
        print("None. Manifold seems safe.")
    else:
        print(watchlist[["dataset", "method", "safe_clip_rate", "gamma_zero_rate", "manifold_margin_mean"]])

if __name__ == "__main__":
    main()
