
import os
import sys
import argparse
import pandas as pd
import json

def load_json(path):
    if not os.path.exists(path): return None
    try:
        with open(path) as f: return json.load(f)
    except: return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    seeds = [0, 4]
    rows = []
    
    status_lines = []
    
    for seed in seeds:
        seed_dir = os.path.join(args.root, f"seed{seed}", "roi_manifold_validate", "manifold")
        
        # Artifacts
        path_metrics = os.path.join(seed_dir, "report_metrics.json")
        path_diag = os.path.join(seed_dir, "report_diagnostics.json")
        path_probe1 = os.path.join(seed_dir, "report_probe_pre_cov.json")
        path_probe2 = os.path.join(seed_dir, "report_probe_post_roi.json")
        path_pred = os.path.join(seed_dir, "report_preds_test_last_trial.csv") # Assuming name from runner
        # Just check file existence
        # Runner output names: `report_preds_test_last_trial.csv`
        
        exists_metrics = os.path.exists(path_metrics)
        exists_diag = os.path.exists(path_diag)
        exists_probe1 = os.path.exists(path_probe1)
        exists_probe2 = os.path.exists(path_probe2)
        exists_pred = os.path.exists(path_pred)
        
        metrics = load_json(path_metrics)
        diag = load_json(path_diag)
        probe1 = load_json(path_probe1)
        
        # Extract Integrity
        roi_flag = metrics['metadata']['use_roi_pooling'] if metrics else False
        roi_k = metrics['metadata']['roi_k'] if metrics else 0
        cov_dim = diag['cov_dim'] if diag and 'cov_dim' in diag else 0
        probe_shape = probe1['x_shape'] if probe1 else []
        
        # Check Integrity
        pass_integrity = (
            exists_metrics and exists_diag and exists_probe1 and exists_pred and
            roi_flag and roi_k == 13 and cov_dim == 13 and
            len(probe_shape) == 3 and probe_shape[1] == 13
        )
        
        fail_reasons = []
        if not exists_metrics: fail_reasons.append("MissingMetrics")
        if not exists_diag: fail_reasons.append("MissingDiag")
        if not exists_probe1: fail_reasons.append("MissingProbe")
        if not roi_flag: fail_reasons.append("FlagFalse")
        if roi_k != 13: fail_reasons.append(f"K={roi_k}")
        if cov_dim != 13: fail_reasons.append(f"CovDim={cov_dim}")
        
        status = "PASS" if pass_integrity else f"FAIL ({','.join(fail_reasons)})"
        
        # Metrics
        trial_acc = metrics['last']['test_trial_acc'] if metrics else 0.0
        win_acc = metrics['last']['test_win_acc'] if metrics else 0.0
        
        # Diag
        eff_rank = diag['eff_rank'] if diag else 0.0
        cond = diag['cond_p95'] if diag else 0.0
        eps_dom = diag['eps_dominance'] if diag else 0.0
        low_eigs = diag['eigs_le_10eps_count'] if diag else 0.0
        
        rows.append({
            "Seed": seed,
            "Trial Acc": f"{trial_acc:.4f}",
            "Win Acc": f"{win_acc:.4f}",
            "Cov Dim": cov_dim,
            "Eff Rank": f"{eff_rank:.2f}",
            "Cond P95": f"{cond:.1f}",
            "Eps Dom": f"{eps_dom:.4f}",
            "Low Eigs": f"{low_eigs:.1f}",
            "Status": status
        })
        
        status_lines.append(f"Seed {seed}: {status}")

    df = pd.DataFrame(rows)
    
    # Generate Markdown
    md = []
    md.append("# Phase 13D Step 2b: ROI Manifold-Only Validation")
    md.append(f"**Date**: {pd.Timestamp.now()}")
    md.append("")
    md.append("## 1. Summary Table")
    md.append("| Seed | Trial Acc | Win Acc | Cov Dim | Eff Rank | Cond P95 | Eps Dom | Low Eigs | Status |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['Seed']} | {r['Trial Acc']} | {r['Win Acc']} | {r['Cov Dim']} | {r['Eff Rank']} | {r['Cond P95']} | {r['Eps Dom']} | {r['Low Eigs']} | {r['Status']} |")
    md.append("")
    md.append("## 2. Integrity Check")
    for line in status_lines:
        md.append(f"- {line}")
        
    with open(args.out, "w") as f:
        f.write("\n".join(md))
        
    print(f"Report generated: {args.out}")
    print(df.to_string())

if __name__ == "__main__":
    main()
