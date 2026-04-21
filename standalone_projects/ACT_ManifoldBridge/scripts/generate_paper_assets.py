import pandas as pd
import numpy as np
import os
import argparse

def generate_latex_table(csv_path, out_file):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Group by dataset and mode
    cols = ['dataset', 'base_f1', 'act_f1', 'gain', 'f1_gain_pct']
    stats = df.groupby('dataset').agg({
        'base_f1': ['mean', 'std'],
        'act_f1': ['mean', 'std'],
        'gain': ['mean', 'std'],
        'f1_gain_pct': ['mean']
    })

    stats.columns = [f"{c[0]}_{c[1]}" for c in stats.columns]
    stats = stats.reset_index()

    print("\n--- LaTeX Table Snippet ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l|c|c|c}")
    print("\\hline")
    print("Dataset & Base F1 (mean $\\pm$ std) & ACT F1 (mean $\\pm$ std) & Gain (\\%) \\\\")
    print("\\hline")
    
    for _, row in stats.iterrows():
        base_str = f"{row['base_f1_mean']:.4f} \\pm {row['base_f1_std']:.4f}"
        act_str = f"{row['act_f1_mean']:.4f} \\pm {row['act_f1_std']:.4f}"
        gain_str = f"{row['f1_gain_pct_mean']:.1f}\\%"
        
        # Bold if gain > 2.0%
        if row['f1_gain_pct_mean'] > 2.0:
            act_str = "\\mathbf{" + act_str + "}"
            
        print(f"{row['dataset'].capitalize()} & {base_str} & {act_str} & {gain_str} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Performance comparison of ACT against Baseline across multiple datasets.}")
    print("\\end{table}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/final_paper_sweep_v1/final_results.csv")
    args = parser.parse_args()
    generate_latex_table(args.csv, None)
