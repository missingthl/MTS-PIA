
import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Phase 13C-3 Experiment Report")
    parser.add_argument("--root", type=str, required=True, help="Root directory (promoted_results/phase13c3)")
    parser.add_argument("--out", type=str, required=True, help="Output markdown path")
    return parser.parse_args()

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def check_path(path_str):
    return path_str if os.path.exists(path_str) else "NOT FOUND"

def main():
    args = parse_args()
    
    # 1. Run Context
    lines = []
    lines.append("# Phase 13C-3 Experiment Report")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Root**: `{args.root}`")
    lines.append("")
    
    # 2. Locked Config
    config_path = os.path.join(args.root, "config_locked.json")
    lines.append("## 1. Locked Configuration")
    if os.path.exists(config_path):
        lines.append(f"Source: `{config_path}`")
        lines.append("```json")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            lines.append(content)
        except Exception as e:
            lines.append(f"Error reading config: {e}")
        lines.append("```")
    else:
        lines.append(f"**CRITICAL**: Config not found at `{config_path}`")
    lines.append("")
        
    # 3. Artifact Index
    lines.append("## 2. Artifact Index")
    summary_csv = os.path.join(args.root, "seed1/summary.csv")
    summary_md = os.path.join(args.root, "seed1/PHASE13C3_SUMMARY.md")
    
    lines.append("### Global Artifacts")
    lines.append(f"- Summary CSV: `{check_path(summary_csv)}`")
    lines.append(f"- Summary MD: `{check_path(summary_md)}`")
    lines.append("")
    
    seeds = [0, 1, 2, 3, 4]
    
    for seed in seeds:
        seed_dir = os.path.join(args.root, f"seed1/seed{seed}")
        m_rept = os.path.join(seed_dir, "manifold/report.json")
        m_pred = os.path.join(seed_dir, "manifold/manifold_trial_pred.csv")
        s_pred = os.path.join(seed_dir, "fusion/spatial_trial_pred.csv")
        pass_txt = os.path.join(seed_dir, "INTEGRITY_PASS.txt")
        fail_txt = os.path.join(seed_dir, "INTEGRITY_FAIL.txt")
        stats_json = os.path.join(seed_dir, "c3_worker_stats.json")
        
        status = "UNKNOWN"
        if os.path.exists(pass_txt): status = "PASS"
        elif os.path.exists(fail_txt): status = "FAIL"
        
        lines.append(f"### Seed {seed} [{status}]")
        lines.append(f"- Manifold Report: `{check_path(m_rept)}`")
        lines.append(f"- Manifold Preds: `{check_path(m_pred)}`")
        lines.append(f"- Spatial Preds: `{check_path(s_pred)}`")
        lines.append(f"- Stats JSON: `{check_path(stats_json)}`")
        lines.append("")

    # 4. Metrics Table
    lines.append("## 3. Per-Seed Metrics")
    lines.append("| Seed | Status | Manifold Acc | Fusion Acc (V0) | Agree Rate | Conflict Rate | Conflict (S_Acc) | Conflict (M_Acc) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    
    pass_count = 0
    fail_count = 0
    
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        # Check standard columns
        # Expected: seed, status, spatial_acc, manifold_acc, fusion_v0_acc, agree_rate, conflict_rate, ...
        
        for _, row in df.iterrows():
            s = row.get('seed', 'N/A')
            st = row.get('status', 'N/A')
            
            if st == 'PASS': pass_count += 1
            if st == 'FAIL': fail_count += 1
            
            m_acc = f"{row.get('manifold_acc', 0.0):.4f}"
            f_acc = f"{row.get('fusion_v0_acc', 0.0):.4f}"
            agr = f"{row.get('agree_rate', 0.0):.4f}"
            con = f"{row.get('conflict_rate', 0.0):.4f}"
            s_con = f"{row.get('spatial_acc_conflict', 0.0):.4f}"
            m_con = f"{row.get('manifold_acc_conflict', 0.0):.4f}"
            
            lines.append(f"| {s} | {st} | {m_acc} | {f_acc} | {agr} | {con} | {s_con} | {m_con} |")
    else:
        lines.append("| ERROR | Summary CSV Missing | - | - | - | - | - | - |")
        
    lines.append("")
    
    # 5. Integrity Summary
    lines.append("## 4. Integrity Summary")
    lines.append(f"- **Expected Seeds**: {len(seeds)}")
    lines.append(f"- **PASS**: {pass_count}")
    lines.append(f"- **FAIL**: {fail_count}")
    if (pass_count + fail_count) < len(seeds):
        lines.append(f"- **MISSING**: {len(seeds) - (pass_count + fail_count)}")
        
    # 6. Repro
    lines.append("")
    lines.append("## 5. Reproduction Commands")
    lines.append("```bash")
    lines.append(f"# Generate this report")
    lines.append(f"{sys.executable} scripts/gen_phase13c3_experiment_report.py --root {args.root} --out {args.out}")
    lines.append("")
    lines.append(f"# Run Experiment")
    lines.append(f"{sys.executable} scripts/run_phase13c3_locked.py")
    lines.append("```")
    
    # Write
    with open(args.out, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Report Generated: {args.out}")

if __name__ == "__main__":
    main()
