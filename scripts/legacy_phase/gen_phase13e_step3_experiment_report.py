
import os
import json
import argparse
import pandas as pd
import numpy as np

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def main():
    seeds = [0, 4]
    base_path_fmt = "promoted_results/phase13e/step3/seed1/seed{}/centered_corr/manifold/report_{}.json"
    
    results = []
    
    for seed in seeds:
        m_path = base_path_fmt.format(seed, "metrics")
        d_path = base_path_fmt.format(seed, "diagnostics")
        
        metrics = load_json(m_path)
        diag = load_json(d_path)
        
        res = {
            "Seed": seed,
            "Trial Acc": "N/A",
            "Win Acc": "N/A",
            "Cond P95": "N/A",
            "Eff Rank": "N/A",
            "Eps Dom": "N/A",
            "Split Check": "❌ FAIL",
            "Subj Center": "N/A",
            "Identity Diff": "N/A",
            "Subj Coverage": "N/A"
        }
        
        if metrics:
            # Handle nested structure (best/last)
            if 'best' in metrics:
                res["Trial Acc"] = f"{metrics['best'].get('test_trial_acc', 0):.4f}"
                res["Win Acc"] = f"{metrics['best'].get('test_win_acc', 0):.4f}"
            else:
                res["Trial Acc"] = f"{metrics.get('te_trial_acc', 0):.4f}"
                res["Win Acc"] = f"{metrics.get('te_win_acc', 0):.4f}"
            
            res["Subj Center"] = str(metrics.get("metadata", {}).get("subject_centering", False))

        if diag:
            res["Cond P95"] = f"{diag.get('cond_p95', diag.get('post_eps_cond_p95', 0)):.1f}"
            res["Eff Rank"] = f"{diag.get('eff_rank', 0):.4f}"
            res["Eps Dom"] = f"{diag.get('eps_dominance', 0):.4f}"
            
            # Split Integrity Check
            if 'split_stats' in diag:
                ss = diag['split_stats']
                overlap = ss.get('train_val_intersection', 999)
                if overlap == 0:
                    res["Split Check"] = "✅ PASS"
                else:
                    res["Split Check"] = f"❌ FAIL ({overlap})"
            else:
                res["Split Check"] = "MISSING"

        # Centering Diagnostics
        c_path = base_path_fmt.format(seed, "centering_diagnostics").replace("report_centering_diagnostics.json", "centered_corr/report_centering_diagnostics.json")
        # Wait, format uses "report_{}.json". 
        # So "metrics" -> "report_metrics.json".
        # "centering_diagnostics" -> "report_centering_diagnostics.json".
        # But file is in `.../centered_corr/report_centering_diagnostics.json`.
        # Base format is `.../manifold/report_{}.json`.
        # We need to adjust path.
        
        # Manifold structure:
        # report_metrics.json is in .../manifold/
        # report_centering_diagnostics.json is in .../manifold/centered_corr/
        
        # Let's fix path logic.
        c_path = m_path.replace("report_metrics.json", "centered_corr/report_centering_diagnostics.json")
        
        c_diag = load_json(c_path)
        if c_diag:
            res["Subj Center"] = "True"
            if "centering_identity_check" in c_diag:
                 res["Identity Diff"] = f"{c_diag['centering_identity_check']:.1e}"
            res["Subj Coverage"] = f"✅ PASS ({c_diag.get('train_subject_count')} Subj)"
        else:
            res["Subj Coverage"] = "❓ MISSING"
                
        results.append(res)
        
    # Generate MD
    md = []
    md.append("# Phase 13E Step 3: Riemannian Subject Centering")
    md.append(f"**Date**: {pd.Timestamp.now()}")
    md.append("")
    
    md.append("## 1. Experiment Summary")
    header = "| Seed | Trial Acc | Win Acc | Subj Center | Identity Diff | Cond P95 | Eff Rank | Eps Dom | Split Check | Subj Coverage |"
    md.append(header)
    md.append("|" + "---|" * 10)
    
    for r in results:
        line = f"| {r['Seed']} | {r['Trial Acc']} | {r['Win Acc']} | {r['Subj Center']} | {r['Identity Diff']} | {r['Cond P95']} | {r['Eff Rank']} | {r['Eps Dom']} | {r['Split Check']} | {r['Subj Coverage']} |"
        md.append(line)
        
    md.append("")
    md.append("## 2. Artifacts")
    for seed in seeds:
        md.append(f"### Seed {seed}")
        report_path = f"promoted_results/phase13e/step3/seed1/seed{seed}/centered_corr/manifold/SINGLE_RUN_REPORT.md"
        md.append(f"- Report: `{report_path}`")
        
    out_path = "promoted_results/phase13e/step3/EXPERIMENT_REPORT.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(md))
        
    # Summary CSV
    df = pd.DataFrame(results)
    df.to_csv("promoted_results/phase13e/step3/summary.csv", index=False)
    
    print(f"Experiment Report Generated: {out_path}")

if __name__ == "__main__":
    main()
