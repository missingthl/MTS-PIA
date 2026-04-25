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
    
    all_files = [
        path
        for path in glob.glob(os.path.join(args.root, "**/*_results.csv"), recursive=True)
        if os.path.basename(path) not in {"sweep_results.csv", "final_results.csv"}
    ]
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
                v2_cols = [
                    "feedback_weight_mean",
                    "consistency_loss_mean",
                    "router_p_lraes_final",
                    "osf_structure_overflow_rate",
                    "osf_alpha_eff_mean",
                    "osf_risk_scale_mean",
                    "osf_risk_zero_perp_rate",
                    "osf_risk_clipped_rate",
                ]
                for c in v2_cols:
                    if c in ds_df.columns:
                        res[c] = ds_df[c].mean()
                rows.append(res)
            except Exception as e:
                print(f"Warning: Skipping dataset {ds} in file {f} due to error: {e}")
            
    final_df = pd.DataFrame(rows).sort_values("Gain", ascending=False)
    
    # Custom markdown table generation (avoiding tabulate dependency)
    def df_to_markdown(df):
        cols = df.columns.tolist()
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = []
        for _, row in df.iterrows():
            formatted_row = "| " + " | ".join([str(round(v, 4)) if isinstance(v, (float, np.float64)) else str(v) for v in row]) + " |"
            body.append(formatted_row)
        return "\n".join([header, sep] + body)

    md_table = df_to_markdown(final_df)
    with open(args.out, "w") as f_out:
        f_out.write("# ACT-V2 Grand Sweep Summary\n\n")
        f_out.write(md_table)
    
    print(f"Summary written to {args.out}")
    print(md_table)

if __name__ == "__main__":
    main()
