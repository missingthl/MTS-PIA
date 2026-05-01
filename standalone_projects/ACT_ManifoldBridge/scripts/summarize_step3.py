import pandas as pd
import numpy as np
from pathlib import Path

root = Path("/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results/csta_step3_diagnostic_sweep/resnet1d_s123")
combos = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("g")])

all_rows = []
for combo_dir in combos:
    p = combo_dir / "per_seed_external.csv"
    if p.exists():
        df = pd.read_csv(p)
        mean_f1 = df['aug_f1'].mean()
        name = combo_dir.name
        parts = name.split("_")
        gamma = float(parts[0][1:])
        eta = float(parts[1][1:])
        
        all_rows.append({
            "gamma": gamma,
            "eta_safe": eta,
            "mean_f1": mean_f1,
            "gamma_used": df.get('gamma_used_mean', pd.Series([0.0]*len(df))).mean(),
            "safe_clip": df.get('safe_clip_rate', pd.Series([0.0]*len(df))).mean(),
        })

summary = pd.DataFrame(all_rows).sort_values("mean_f1", ascending=False)
print("### Part 3 Step 3 Optimization Sweep Summary")
print(summary.to_string(index=False))

if not summary.empty:
    best_f1 = summary.iloc[0]['mean_f1']
    best_name = f"g{summary.iloc[0]['gamma']}_e{summary.iloc[0]['eta_safe']}"
    print(f"\nBest Combo: {best_name} (F1={best_f1:.6f})")

    print("\n### Final Leaderboard (NeurIPS Draft Status)")
    leaderboard = [
        {"method": f"PIA (Uniform-Top5, {best_name})", "mean_f1": best_f1, "gap_vs_wdba": best_f1 - 0.667922},
        {"method": "wDBA (Same-Class)", "mean_f1": 0.667922, "gap_vs_wdba": 0.0},
        {"method": "DBA (Same-Class)", "mean_f1": 0.663309, "gap_vs_wdba": 0.663309 - 0.667922},
        {"method": "CSTA (Top-1, Baseline)", "mean_f1": 0.650467, "gap_vs_wdba": 0.650467 - 0.667922},
        {"method": "No-Aug", "mean_f1": 0.621237, "gap_vs_wdba": 0.621237 - 0.667922},
    ]
    lb_df = pd.DataFrame(leaderboard)
    print(lb_df.to_string(index=False))
