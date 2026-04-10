
import os
import pandas as pd
import numpy as np

BASE_DIR = "promoted_results/phase13/seed0"
SPATIAL_PATH = os.path.join(BASE_DIR, "spatial_trial_pred.csv")
MANIFOLD_PATH = os.path.join(BASE_DIR, "manifold_trial_pred_guided_p2.csv")
REPORT_PATH = os.path.join(BASE_DIR, "phase13A_seed0_regression_report.md")

def main():
    print("Starting Triage Diagnostics...")
    
    # Load Data
    sa = pd.read_csv(SPATIAL_PATH)
    ma = pd.read_csv(MANIFOLD_PATH)
    
    # === S4: Alignment Red-Line Checks ===
    print("\n[S4] Alignment Checks")
    checks = {}
    
    # 1. Uniqueness
    s_unique = sa['trial_id'].nunique() == len(sa)
    m_unique = ma['trial_id'].nunique() == len(ma)
    checks['trial_id_unique'] = s_unique and m_unique
    print(f"Unique Trial IDs: Spatial={s_unique}, Manifold={m_unique}")
    
    # 2. Set Equality
    s_ids = set(sa['trial_id'].astype(str))
    m_ids = set(ma['trial_id'].astype(str))
    set_match = (s_ids == m_ids)
    checks['trial_id_match'] = set_match
    print(f"Trial ID Set Match: {set_match}")
    if not set_match:
        print(f"Spatial Extra: {len(s_ids - m_ids)}")
        print(f"Manifold Extra: {len(m_ids - s_ids)}")
        
    # 3. Join Count
    # Ensure trial_ids are strings
    sa['trial_id'] = sa['trial_id'].astype(str)
    ma['trial_id'] = ma['trial_id'].astype(str)
    
    merged = pd.merge(sa, ma, on='trial_id', suffixes=('_s', '_m'))
    count_match = len(merged) == len(sa) == len(ma)
    checks['join_count_match'] = count_match
    print(f"Join Count Match: {count_match} (Merged={len(merged)}, S={len(sa)}, M={len(ma)})")
    
    # 4. Label Consistency
    # Check if true_label_s == true_label_m
    if 'true_label_s' in merged.columns and 'true_label_m' in merged.columns:
        label_match = (merged['true_label_s'] == merged['true_label_m']).all()
        checks['label_match'] = label_match
        print(f"Label Consistency: {label_match}")
    else:
        checks['label_match'] = False
        print("Labels missing in merge?")

    # === S3: Aggregation Drift ===
    # We can't strictly recompute majority vote without window data, 
    # but we can check if current acc matches 'mean_prob' logic.
    # The Manifold CSV 'pred_label' is likely argmax(mean_prob).
    # Let's verify that.
    
    merged['calc_pred_m'] = np.argmax(merged[['prob_0_m', 'prob_1_m', 'prob_2_m']].values, axis=1)
    calc_acc_m_mean = (merged['calc_pred_m'] == merged['true_label_m']).mean()
    
    # Current reported acc
    reported_acc = (ma['pred_label'] == ma['true_label']).mean()
    
    print(f"\n[S3] Aggregation Checks")
    print(f"Reported Acc: {reported_acc:.4f}")
    print(f"Calculated (Mean Prob): {calc_acc_m_mean:.4f}")
    
    # === Conflict Diagnostics ===
    print("\n[D] Conflict Diagnostics")
    merged['correct_s'] = merged['pred_label_s'] == merged['true_label_s']
    merged['correct_m'] = merged['pred_label_m'] == merged['true_label_m']
    
    agree = merged[merged['pred_label_s'] == merged['pred_label_m']]
    conflict = merged[merged['pred_label_s'] != merged['pred_label_m']]
    
    agree_rate = len(agree) / len(merged)
    conflict_rate = len(conflict) / len(merged)
    
    s_acc_conflict = conflict['correct_s'].mean()
    m_acc_conflict = conflict['correct_m'].mean()
    
    print(f"Agree Rate: {agree_rate:.2%}")
    print(f"Conflict Rate: {conflict_rate:.2%}")
    print(f"Spatial Acc on Conflict: {s_acc_conflict:.2%}")
    print(f"Manifold Acc on Conflict: {m_acc_conflict:.2%}")
    
    # Generate Report
    with open(REPORT_PATH, "w") as f:
        f.write("# Phase 13A Seed 0 Regression Triage Report\n\n")
        
        f.write("## Executive Summary\n")
        all_s4_pass = all(checks.values())
        if not all_s4_pass:
            f.write("**Status: FAIL (No-Go)**\n")
            f.write("Root Cause: S4 (Alignment Failure)\n")
        else:
            f.write("**Status: PASS (Go)**\n")
            f.write("Root Cause: Likely S3/Model Variance (Triage Cleared)\n")
            
        f.write("\n## Alignment Red-Line Checks (S4)\n")
        for k, v in checks.items():
            f.write(f"- {k}: {'PASS' if v else 'FAIL'}\n")
            
        f.write("\n## Aggregation Drift (S3)\n")
        f.write(f"- Reported Acc: {reported_acc:.4f}\n")
        f.write(f"- Recalculated (Mean Prob): {calc_acc_m_mean:.4f}\n")
        
        f.write("\n## Conflict Diagnostics\n")
        f.write(f"- Conflict Rate: {conflict_rate:.2%}\n")
        f.write(f"- Spatial Acc (Conflict): {s_acc_conflict:.2%}\n")
        f.write(f"- Manifold Acc (Conflict): {m_acc_conflict:.2%}\n")
        
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
