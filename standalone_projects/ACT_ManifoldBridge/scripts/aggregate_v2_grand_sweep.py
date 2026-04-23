import pandas as pd
import numpy as np
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out", type=str, default="v2_summary.md")
    args = parser.parse_args()
    
    all_files = glob.glob(os.path.join(args.root, "**/final_results.csv"), recursive=True)
    if not all_files:
        print(f"No results found in {args.root}")
        return

    rows = []
    for f in all_files:
        df = pd.read_csv(f)
        # Group by dataset to average seeds
        for ds in df['dataset'].unique():
            ds_df = df[df['dataset'] == ds]
            try:
                res = {
                    "Dataset": ds,
                    "Base F1": ds_df['base_f1'].mean() if 'base_f1' in ds_df.columns else 0.0,
                    "ACT F1": ds_df['act_f1'].mean() if 'act_f1' in ds_df.columns else 0.0,
                    "Gain": ds_df['gain'].mean() if 'gain' in ds_df.columns else 0.0,
                    "Gain%": ds_df['f1_gain_pct'].mean() if 'f1_gain_pct' in ds_df.columns else 0.0,
                    "N_Seeds": len(ds_df),
                }
                # Add V2 diagnostics
                v2_cols = ["feedback_weight_mean", "consistency_loss_mean", "router_p_lraes_final"]
                for c in v2_cols:
                    if c in ds_df.columns:
                        res[c] = ds_df[c].mean()
                rows.append(res)
            except Exception as e:
                print(f"Warning: Skipping dataset {ds} in file {f} due to error: {e}")
            
    final_df = pd.DataFrame(rows).sort_values("Gain", ascending=False)
    
    md_table = final_df.to_markdown(index=False)
    with open(args.out, "w") as f_out:
        f_out.write("# ACT-V2 Grand Sweep Summary\n\n")
        f_out.write(md_table)
    
    print(f"Summary written to {args.out}")
    print(md_table)

if __name__ == "__main__":
    main()
