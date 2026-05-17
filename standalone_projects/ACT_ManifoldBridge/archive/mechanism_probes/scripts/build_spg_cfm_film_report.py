import os
import pandas as pd
import numpy as np
from typing import List

def build_report():
    film_path = "results/csta_spg_cfm_align_tuned_pilot7/resnet1d_s123/per_seed_external.csv"
    prev_path = "results/csta_spg_cfm_one_step_pilot7/resnet1d_s123/per_seed_external.csv"
    
    if not os.path.exists(film_path):
        print(f"Waiting for {film_path} to be generated...")
        return

    df_film = pd.read_csv(film_path)
    df_prev = pd.read_csv(prev_path)
    
    # Filter only relevant methods from previous run
    # wdba_sameclass, dba_sameclass
    df_wdba = df_prev[df_prev['method'].isin(['wdba_sameclass', 'dba_sameclass'])]
    
    # Combine
    df_all = pd.concat([df_film, df_wdba], ignore_index=True)
    
    # Calculate Mean F1
    summary = df_all.groupby('method')['aug_f1'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    print("\nOverall Performance (F1):")
    print(summary)
    
    # Build W/T/L vs U5
    u5_f1 = df_all[df_all['method'] == 'csta_topk_uniform_top5'].set_index(['dataset', 'seed'])['aug_f1']
    
    methods = df_all['method'].unique()
    wtl_rows = []
    for m in methods:
        if m == 'csta_topk_uniform_top5': continue
        m_f1 = df_all[df_all['method'] == m].set_index(['dataset', 'seed'])['aug_f1']
        
        # Align
        common_idx = u5_f1.index.intersection(m_f1.index)
        if len(common_idx) == 0: continue
        
        diff = m_f1.loc[common_idx] - u5_f1.loc[common_idx]
        win = (diff > 0.01).sum()
        tie = (np.abs(diff) <= 0.01).sum()
        loss = (diff < -0.01).sum()
        
        wtl_rows.append({
            'method': m,
            'mean_f1': m_f1.mean(),
            'win': win,
            'tie': tie,
            'loss': loss,
            'score': win - loss
        })
        
    wtl_df = pd.DataFrame(wtl_rows).sort_values('mean_f1', ascending=False)
    print("\nW/T/L vs U5 (Threshold=0.01):")
    print(wtl_df)
    
    # Save a combined report
    report_file = "results/csta_spg_cfm_film_pilot7/resnet1d_s123/spg_cfm_film_comparison_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, "w") as f:
        f.write("# SPG-CFM Align & FiLM Architecture Validation Report (Pilot 7)\n\n")
        f.write("## Executive Summary\n")
        f.write(f"Comparing Align-based Loss vs FiLM-based injection vs Concat-based injection.\n\n")
        
        f.write("### Leaderboard (Mean F1)\n")
        f.write(summary.to_string() + "\n\n")
        
        f.write("### Comparison vs U5 (W/T/L)\n")
        f.write(wtl_df.to_string() + "\n\n")
        
        f.write("## Dataset Level Analysis\n")
        pivot = df_all.pivot_table(index='dataset', columns='method', values='aug_f1', aggfunc='mean')
        f.write(pivot.to_string() + "\n")

if __name__ == "__main__":
    build_report()
