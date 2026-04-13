
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
    base_path_fmt = "promoted_results/phase13e/step2/seed1/seed{}/corr_tsm_manifold_only/manifold/report_{}.json"
    
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
            "Matrix Mode": "N/A"
        }
        
        if metrics:
            # Handle nested structure (best/last)
            if 'best' in metrics:
                res["Trial Acc"] = f"{metrics['best'].get('test_trial_acc', 0):.4f}"
                res["Win Acc"] = f"{metrics['best'].get('test_win_acc', 0):.4f}"
            else:
                res["Trial Acc"] = f"{metrics.get('te_trial_acc', 0):.4f}"
                res["Win Acc"] = f"{metrics.get('te_win_acc', 0):.4f}"
            
        if diag:
            res["Matrix Mode"] = diag.get("matrix_mode", "N/A")
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
                
        results.append(res)
        
    # Generate MD
    md = []
    md.append("# Phase 13E Step 2: Correlation-TSM Robustness Validation")
    md.append(f"**Date**: {pd.Timestamp.now()}")
    md.append("")
    
    md.append("## 1. Experiment Summary")
    header = "| Seed | Trial Acc | Win Acc | Matrix Mode | Cond P95 | Eff Rank | Eps Dom | Split Check |"
    md.append(header)
    md.append("|" + "---|" * 8)
    
    for r in results:
        line = f"| {r['Seed']} | {r['Trial Acc']} | {r['Win Acc']} | {r['Matrix Mode']} | {r['Cond P95']} | {r['Eff Rank']} | {r['Eps Dom']} | {r['Split Check']} |"
        md.append(line)
        
    md.append("")
    md.append("## 2. Artifacts")
    for seed in seeds:
        md.append(f"### Seed {seed}")
        report_path = f"promoted_results/phase13e/step2/seed1/seed{seed}/corr_tsm_manifold_only/manifold/SINGLE_RUN_REPORT.md"
        md.append(f"- Report: `{report_path}`")
        
    out_path = "promoted_results/phase13e/step2/EXPERIMENT_REPORT.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(md))
        
    # Summary CSV
    df = pd.DataFrame(results)
    df.to_csv("promoted_results/phase13e/step2/summary.csv", index=False)
    
    print(f"Experiment Report Generated: {out_path}")

if __name__ == "__main__":
    main()
