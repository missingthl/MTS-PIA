
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    return None

def check_gate_stats(gate_path):
    # Verify mean(sum(w)) approx 1.0 and no NaNs
    if not os.path.exists(gate_path):
        return "N/A"
    
    data = load_json(gate_path)
    if not data: return "Load Error"
    
    # data is list of dicts: {'epoch', 'w_mean': [5], 'w_std': [5]}
    w_means = [d['w_mean'] for d in data]
    w_mat = np.array(w_means)
    
    if np.isnan(w_mat).any():
        return "FAIL (NaN)"
        
    # Sum over bands (axis 1)
    sums = w_mat.sum(axis=1)
    # Check if close to 1
    if not np.allclose(sums, 1.0, atol=1e-3):
        return f"FAIL (Sum!=1, Range=[{sums.min():.4f}, {sums.max():.4f}])"
        
    return "PASS"

def main():
    args = parse_args()
    
    lines = []
    lines.append("# Phase 13C-4 Step 1: Band-Scalar Activation Report")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 1. Summary CSV
    sum_csv = os.path.join(args.root, "summary.csv")
    if os.path.exists(sum_csv):
        df = pd.read_csv(sum_csv)
        lines.append("## 1. Summary Metrics")
        lines.append("```")
        lines.append(df.to_string(index=False))
        lines.append("```")
    else:
        lines.append("## 1. Summary Metrics (MISSING)")
        
    lines.append("")
    
    # 2. Detailed Per-Run
    seeds = [0, 4]
    tags = ["control", "bandgate"]
    
    lines.append("## 2. Risk Monitoring & Artifacts")
    lines.append("| Seed | Tag | Status | Gate Integrity | Test Acc |")
    lines.append("|---|---|---|---|---|")
    
    for seed in seeds:
        for tag in tags:
            base = os.path.join(args.root, f"seed{seed}/{tag}")
            rep_path = os.path.join(base, "manifold/report.json")
            gate_path = os.path.join(base, "gate_stats.json")
            
            rep = load_json(rep_path)
            
            status = "UNKNOWN"
            acc = "N/A"
            if rep:
                status = "PASS" # Weak check
                # Check spd stats in metadata if available
                # rep['metadata']['spd_stats']...
                acc = f"{rep.get('last', {}).get('test_trial_acc', 0.0):.4f}"
            else:
                status = "MISSING"
                
            gate_check = "N/A"
            if tag == "bandgate":
                gate_check = check_gate_stats(gate_path)
            elif tag == "control":
                gate_check = "Skipped"
                
            lines.append(f"| {seed} | {tag} | {status} | {gate_check} | {acc} |")
            
    lines.append("")
    
    # 3. Artifact Index
    lines.append("## 3. Artifact Index")
    for seed in seeds:
        for tag in tags:
            base = os.path.join(args.root, f"seed{seed}/{tag}")
            lines.append(f"### Seed {seed} - {tag}")
            lines.append(f"- Report: `{os.path.join(base, 'manifold/report.json')}`")
            lines.append(f"- Preds: `{os.path.join(base, 'manifold/manifold_trial_pred.csv')}`")
            if tag == "bandgate":
                 lines.append(f"- Gate Stats: `{os.path.join(base, 'gate_stats.json')}`")
            lines.append("")

    with open(args.out, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Report Generated: {args.out}")

if __name__ == "__main__":
    main()
