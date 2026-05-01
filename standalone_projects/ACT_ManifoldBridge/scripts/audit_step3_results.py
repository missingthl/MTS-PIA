import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

def audit_step3(root_path: Path, external_ref_path: Path = None):
    root_path = Path(root_path)
    if not root_path.exists():
        print(f"Error: Root path {root_path} does not exist.")
        return

    combos = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("g")])
    if not combos:
        print(f"No valid result directories (gX_eY) found in {root_path}")
        return

    all_data = []
    dataset_details = {}

    for combo_dir in combos:
        p = combo_dir / "dataset_summary_external.csv"
        if not p.exists():
            continue
        
        df = pd.read_csv(p)
        # Filter for the actual CSTA method (assuming only one non-no_aug method per sweep)
        df_csta = df[df['method'] != 'no_aug']
        if df_csta.empty:
            continue
            
        mean_f1 = df_csta['aug_f1_mean'].mean()
        
        name = combo_dir.name
        try:
            parts = name.split("_")
            gamma = float(parts[0][1:])
            eta = float(parts[1][1:])
        except:
            gamma, eta = 0.0, 0.0

        all_data.append({
            "combo": name,
            "gamma": gamma,
            "eta_safe": eta,
            "mean_f1": mean_f1,
            "n_datasets": len(df_csta)
        })
        dataset_details[name] = df_csta.set_index('dataset')['aug_f1_mean'].to_dict()

    if not all_data:
        print("No valid CSTA results found in subdirectories.")
        return

    summary = pd.DataFrame(all_data).sort_values("mean_f1", ascending=False)
    
    print("\n" + "="*60)
    print(" STEP 3 OPTIMIZATION SWEEP AUDIT REPORT")
    print("="*60)
    print(summary.to_string(index=False))
    
    best_row = summary.iloc[0]
    best_combo = best_row['combo']
    print(f"\n>>> BEST CONFIGURATION: {best_combo} (F1: {best_row['mean_f1']:.6f})")

    # Load External Reference if provided
    ref_data = {}
    if external_ref_path and external_ref_path.exists():
        ref_df = pd.read_csv(external_ref_path)
        # Group by method and get mean f1 across datasets
        ref_summary = ref_df.groupby('method')['aug_f1_mean'].mean().to_dict()
        ref_data = ref_summary
    
    print("\n### Comparative Leaderboard")
    leaderboard = []
    leaderboard.append({"Method": f"PIA (Uniform-Top5, {best_combo})", "Mean F1": best_row['mean_f1']})
    
    for method, f1 in sorted(ref_data.items(), key=lambda x: x[1], reverse=True):
        leaderboard.append({"Method": method, "Mean F1": f1})
    
    lb_df = pd.DataFrame(leaderboard)
    if 'wdba_sameclass' in ref_data:
        wdba_val = ref_data['wdba_sameclass']
        lb_df['Gap_vs_wDBA'] = lb_df['Mean F1'] - wdba_val
    
    print(lb_df.to_string(index=False))

    # Dataset Breakdown for Best vs No-Aug
    print("\n### Per-Dataset Analysis (Best vs Reference)")
    if best_combo in dataset_details:
        best_details = dataset_details[best_combo]
        # Try to find no_aug in the same dir or ref
        breakdown = []
        for ds, f1 in best_details.items():
            breakdown.append({"Dataset": ds, "PIA_F1": f1})
        
        print(pd.DataFrame(breakdown).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to step3 sweep root")
    parser.add_argument("--ref", type=str, help="Path to external baseline summary CSV")
    args = parser.parse_args()
    audit_step3(Path(args.root), Path(args.ref) if args.ref else None)
