
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
    base_path_fmt = "promoted_results/phase13e/step5/seed1/seed{}/proto_mdm_logeuclid/manifold/report_{}.json"
    
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
            "Cond P95": "N/A", # Not relevant for Proto except pre-eps
            "Proto Dist": "N/A",
            "Split Check": "❌ FAIL",
            "Log Id Norm": "N/A",
            "Proto Counts": "N/A"
        }
        
        if metrics:
            # Prototype runner saves simplified metrics
            res["Trial Acc"] = f"{metrics.get('trial_agg_acc', 0):.4f}"
            res["Win Acc"] = f"{metrics.get('win_acc', 0):.4f}"
            
        if diag:
            # Prototype meta json has: counts, log_identity_norm, global_centering
            res["Log Id Norm"] = f"{diag.get('log_identity_norm', 0):.4e}"
            res["Proto Counts"] = str(diag.get('counts', 'N/A'))
        
        # Check split (saved in separate file in parent dir usually, or handled by runner)
        # ManifoldDeepRunner saves "report_split.json"
        s_path = base_path_fmt.format(seed, "split")
        split_data = load_json(s_path)
        if split_data:
            overlap = split_data.get('intersection_count', 999)
            if overlap == 0:
                res["Split Check"] = "✅ PASS"
            else:
                res["Split Check"] = f"❌ FAIL ({overlap})"
        else:
            res["Split Check"] = "MISSING"

        results.append(res)
        
    # Generate MD
    md = []
    md.append("# Phase 13E Step 5: Global-Centered Prototype Classifier (Log-Euclid MDM)")
    md.append(f"**Date**: {pd.Timestamp.now()}")
    md.append("")
    
    md.append("## 1. Experiment Summary")
    header = "| Seed | Trial Acc | Win Acc | Log Id Norm | Proto Counts | Split Check |"
    md.append(header)
    md.append("|" + "---|" * 6)
    
    for r in results:
        line = f"| {r['Seed']} | {r['Trial Acc']} | {r['Win Acc']} | {r['Log Id Norm']} | {r['Proto Counts']} | {r['Split Check']} |"
        md.append(line)
        
    md.append("")
    md.append("## 2. Artifacts")
    for seed in seeds:
        md.append(f"### Seed {seed}")
        report_path = f"promoted_results/phase13e/step5/seed1/seed{seed}/proto_mdm_logeuclid/manifold/SINGLE_RUN_REPORT.md"
        preds_path = f"promoted_results/phase13e/step5/seed1/seed{seed}/proto_mdm_logeuclid/manifold/manifold_trial_pred.csv"
        md.append(f"- Report: `{report_path}`")
        md.append(f"- Predictions: `{preds_path}`")
        
    out_path = "promoted_results/phase13e/step5/EXPERIMENT_REPORT.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(md))
        
    # Summary CSV
    df = pd.DataFrame(results)
    df.to_csv("promoted_results/phase13e/step5/summary.csv", index=False)
    
    print(f"Experiment Report Generated: {out_path}")

if __name__ == "__main__":
    main()
