
import os
import glob
import json
import pandas as pd
import numpy as np

def run_analysis():
    pred_dir = "experiments/phase9_fusion/preds"
    report_dir = "experiments/phase9_fusion/reports"
    os.makedirs(report_dir, exist_ok=True)
    
    seeds = [0, 1, 2]
    
    summary_rows = []
    
    for seed in seeds:
        print(f"\nAnalyzing Seed {seed}...")
        
        # Paths
        m_tr_path = os.path.join(pred_dir, f"manifold_train_preds_seed{seed}.csv")
        m_te_path = os.path.join(pred_dir, f"manifold_trial_preds_seed{seed}.csv")
        s_tr_path = os.path.join(pred_dir, f"spatial_train_preds_seed{seed}.csv")
        s_te_path = os.path.join(pred_dir, f"spatial_trial_preds_seed{seed}.csv")
        
        # Check files
        if not all(os.path.exists(p) for p in [m_tr_path, m_te_path, s_tr_path, s_te_path]):
            print(f"Skipping Seed {seed}: Missing files.")
            continue
            
        # Load & Index by trial_id
        df_m_tr = pd.read_csv(m_tr_path).set_index("trial_id")
        df_m_te = pd.read_csv(m_te_path).set_index("trial_id")
        df_s_tr = pd.read_csv(s_tr_path).set_index("trial_id")
        df_s_te = pd.read_csv(s_te_path).set_index("trial_id")
        
        # Align (Intersect & Sort)
        common_tr = df_m_tr.index.intersection(df_s_tr.index)
        common_te = df_m_te.index.intersection(df_s_te.index)
        
        df_m_tr = df_m_tr.loc[common_tr].sort_index()
        df_s_tr = df_s_tr.loc[common_tr].sort_index()
        df_m_te = df_m_te.loc[common_te].sort_index()
        df_s_te = df_s_te.loc[common_te].sort_index()
        
        # Save trial_id list
        with open(os.path.join(report_dir, f"trial_id_list_seed{seed}.txt"), "w") as f:
            f.write("\n".join(common_te))
            
        # Extract Probs
        # Columns: prob_0, prob_1, prob_2
        # Note: Spatial might be only prob_0/1 or fill 2 with 0.0. My code handled it.
        # Manifold now has p0, p1, p2.
        
        def get_probs(df):
            return df[["prob_0", "prob_1", "prob_2"]].values.astype(float)
        
        Pm_tr = get_probs(df_m_tr)
        Ps_tr = get_probs(df_s_tr)
        y_tr = df_m_tr["true_label"].values
        
        Pm_te = get_probs(df_m_te)
        Ps_te = get_probs(df_s_te)
        y_te = df_m_te["true_label"].values
        
        # --- Weight Search (Validation on Train) ---
        # Grid: 0.0 to 1.0 step 0.05
        weights = np.arange(0.0, 1.05, 0.05)
        best_w = 0.0
        best_acc = -1.0
        search_log = []
        
        for w in weights:
            # P = w * Ps + (1-w) * Pm
            P_fused = w * Ps_tr + (1-w) * Pm_tr
            preds = np.argmax(P_fused, axis=1)
            acc = np.mean(preds == y_tr)
            search_log.append({"w": w, "acc": acc})
            if acc > best_acc:
                best_acc = acc
                best_w = w
                
        # Save Search Log
        pd.DataFrame(search_log).to_csv(os.path.join(report_dir, f"weight_search_seed{seed}.csv"), index=False)
        with open(os.path.join(report_dir, f"selected_weight_seed{seed}.txt"), "w") as f:
            f.write(f"selected_weight: {best_w:.2f}\nval_acc: {best_acc:.4f}\n")
            
        print(f"Seed {seed}: Selected w={best_w:.2f} (Val Acc: {best_acc:.4f})")
        
        # --- Evaluation on Test ---
        # Apply w*
        P_final = best_w * Ps_te + (1-best_w) * Pm_te
        pred_final = np.argmax(P_final, axis=1)
        acc_final = np.mean(pred_final == y_te)
        
        # Baselines
        pred_m = np.argmax(Pm_te, axis=1)
        acc_m = np.mean(pred_m == y_te)
        
        pred_s = np.argmax(Ps_te, axis=1)
        acc_s = np.mean(pred_s == y_te)
        
        # Complementarity
        m_corr = (pred_m == y_te)
        s_corr = (pred_s == y_te)
        
        both_correct = np.mean(m_corr & s_corr)
        m_only = np.mean(m_corr & (~s_corr))
        s_only = np.mean((~m_corr) & s_corr)
        both_wrong = np.mean((~m_corr) & (~s_corr))
        disagreement = np.mean(pred_m != pred_s)
        
        # Save Comp Report
        comp_df = pd.DataFrame([{
            "seed": seed,
            "acc_manifold": acc_m,
            "acc_spatial": acc_s,
            "acc_fused": acc_final,
            "best_w": best_w,
            "both_correct": both_correct,
            "manifold_only": m_only,
            "spatial_only": s_only,
            "both_wrong": both_wrong,
            "disagreement": disagreement
        }])
        comp_df.to_csv(os.path.join(report_dir, f"complementarity_seed{seed}.csv"), index=False)
        
        summary_rows.append(comp_df.iloc[0])

    # Summary
    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        sum_path = os.path.join(report_dir, "phase9_fusion_summary.csv")
        sum_df.to_csv(sum_path, index=False)
        print("\n=== Phase 9 Summary ===")
        print(sum_df[["seed", "acc_manifold", "acc_spatial", "acc_fused", "best_w"]])
        
        # Generate Markdown Summary
        md_path = os.path.join(report_dir, "phase9_fusion_summary.md")
        avg = sum_df.mean(numeric_only=True)
        std = sum_df.std(numeric_only=True)
        
        with open(md_path, "w") as f:
            f.write(f"# Phase 9.0 Fusion Summary\n\n")
            f.write(f"## Metrics (Mean ± Std)\n")
            f.write(f"- Manifold Acc: {avg['acc_manifold']:.4f} ± {std['acc_manifold']:.4f}\n")
            f.write(f"- Spatial Acc: {avg['acc_spatial']:.4f} ± {std['acc_spatial']:.4f}\n")
            f.write(f"- Fused Acc: {avg['acc_fused']:.4f} ± {std['acc_fused']:.4f}\n")
            f.write(f"- Disagreement: {avg['disagreement']:.4f}\n\n")
            f.write(f"## Complementarity\n")
            f.write(f"- Both Correct: {avg['both_correct']:.4f}\n")
            f.write(f"- Manifold Only: {avg['manifold_only']:.4f}\n")
            f.write(f"- Spatial Only: {avg['spatial_only']:.4f}\n")
            f.write(f"- Both Wrong: {avg['both_wrong']:.4f}\n")

if __name__ == "__main__":
    run_analysis()
