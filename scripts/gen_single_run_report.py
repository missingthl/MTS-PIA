
import os
import sys
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    
    root = args.root
    # Artifact paths (Runner adds `_metrics.json` suffix to fold_name)
    # The fold_name passed to runner was "phase13d/.../report".
    # So files are "report_metrics.json", "report_gate_stats.json", etc.
    # Actually, verify orchestrator passed `rel_path` ending with `report`.
    # Yes: `f"phase13d/step1/{args.dataset}/seed{seed}/{tag}/manifold/report"`
    # So `promoted_results/phase13d/.../report_metrics.json`
    
    # Wait, Orchestrator prints `Output: promoted_results/.../manifold`.
    # The file is `promoted_results/.../manifold/report_metrics.json`.
    # `root` passed to this script should be the directory containing `report_metrics.json`.
    
    metrics_path = os.path.join(root, "report_metrics.json")
    gate_path = os.path.join(root, "report_gate_stats.json")
    diag_path = os.path.join(root, "report_diagnostics.json")
    meta_path = os.path.join(root, "report_metadata.json") # Often implicit in metrics or separate
    
    metrics = load_json(metrics_path)
    gate_stats = load_json(gate_path)
    diag = load_json(diag_path)
    
    lines = []
    lines.append(f"# Single Run Report: {os.path.basename(root)}")
    lines.append(f"**Date**: {pd.Timestamp.now()}")
    lines.append("")
    
    # 1. Run Context
    lines.append("## 1. Run Context")
    if metrics and 'metadata' in metrics:
        m = metrics['metadata']
        lines.append(f"- **Seed**: {m.get('seed')}")
        lines.append(f"- **Mode**: {m.get('bands_mode')} / {m.get('band_norm_mode')}")
        lines.append(f"- **Gate**: {m.get('band_activation_mode') if 'band_activation_mode' in m else 'N/A'}")
        lines.append(f"- **ROI Pooling**: {m.get('use_roi_pooling', False)} (K={m.get('roi_k', 'N/A')})")
        lines.append(f"- **Eps**: {m.get('spd_eps')}")
        lines.append(f"- **Epochs**: {m.get('epochs')}")
    else:
        lines.append("Metadata missing in metrics.")
    lines.append("")
    
    # 2. Training Summary
    lines.append("## 2. Training Summary")
    if metrics:
        best = metrics.get('best', {})
        last = metrics.get('last', {})
        lines.append(f"- **Best Val Acc**: {best.get('val_acc', 0):.4f} (Ep {best.get('epoch')})")
        lines.append(f"- **Final Test Win Acc**: {last.get('test_win_acc', 0):.4f}")
        lines.append(f"- **Final Test Trial Acc**: {last.get('test_trial_acc', 0):.4f}")
    else:
        lines.append("Metrics missing.")
    lines.append("")
    
    # 3. Gate Stats
    lines.append("## 3. Gate Statistics")
    if gate_stats:
        # Show first and last epoch
        first = gate_stats[0]
        last = gate_stats[-1]
        
        lines.append("### First Epoch")
        lines.append(f"- Means: {first.get('w_mean')}")
        lines.append(f"- Stds: {first.get('w_std')}")
        
        lines.append("### Last Epoch")
        lines.append(f"- Means: {last.get('w_mean')}")
        lines.append(f"- Stds: {last.get('w_std')}")
        
        # Collapse Check
        means = np.array(last.get('w_mean'))
        if np.max(means) > 0.8:
            lines.append("⚠️ **Collapse Warning**: Single band dominates (>0.8).")
        else:
            lines.append("✅ **Balanced**: No single band > 0.8.")
    else:
        lines.append("Gate stats missing.")
    lines.append("")
    
    lines.append("## 4. Covariance/Correlation Spectrum (Diagnostics)")
    if diag:
        matrix_mode = diag.get('matrix_mode', 'cov')
        lines.append(f"- **Matrix Mode**: {matrix_mode}")
        if 'cov_dim' in diag:
            lines.append(f"- **Dimension**: {diag['cov_dim']}")

        # Pre-Eps
        lines.append("### Pre-Eps Stats")
        if matrix_mode == 'corr':
            lines.append(f"- **Diag Mean**: {diag.get('pre_eps_diag_mean'):.4f} (Target ~1.0)")
            lines.append(f"- **Diag Std**: {diag.get('pre_eps_diag_std'):.4f}")
            lines.append(f"- **Off-Diag Abs Mean**: {diag.get('pre_eps_offdiag_abs_mean'):.4f}")
        lines.append(f"- **Min Eig (P05)**: {diag.get('pre_eps_min_eig_p05'):.1e}")
        lines.append(f"- **Cond No (P95)**: {diag.get('pre_eps_cond_p95'):.1f}")

        # Post-Eps
        lines.append("### Post-Eps Stats")
        lines.append(f"- **Min Eig (P05)**: {diag.get('post_eps_min_eig_p05'):.1e}")
        lines.append(f"- **Cond No (P95)**: {diag.get('post_eps_cond_p95'):.1f}")

        lines.append(f"- **Effective Rank**: {diag.get('eff_rank', 0.0):.4f}")
        lines.append(f"- **Eps Dominance**: {diag.get('eps_dominance', 0.0):.4f}")
        lines.append(f"- **Eigs <= 10eps**: {diag.get('eigs_le_10eps_count', 0):.1f}")
        
        # Success Logic
        pass_rank = diag.get('eff_rank', 0) > 12.0
        pass_dom = diag.get('eps_dominance', 1.0) < 0.05

        lines.append(f"- **Rank Check**: {'✅ PASS (>12)' if pass_rank else '⚠️ LOW'}")
        lines.append(f"- **Energy Check**: {'✅ PASS (<0.05)' if pass_dom else '⚠️ LOW ENERGY'}")
    else:
        lines.append("Diagnostics missing.")
    lines.append("")

    # 5. Riemannian Centering (Phase 13E Step 3)
    centering_path = os.path.join(root, "report_centering_diagnostics.json")
    centering_diag = load_json(centering_path)
    
    lines.append("## 5. Riemannian Centering")
    if metrics and metrics.get('metadata', {}).get('subject_centering', False) or centering_diag:
         lines.append("- **Subject Centering**: Enabled")
         lines.append("- **Mean Type**: Log-Euclidean") # Hardcoded for this step
         
         if centering_diag:
             lines.append(f"- **Train Subjects**: {centering_diag.get('train_subject_count')}")
             # lines.append(f"- **Subjects**: {centering_diag.get('subjects')}")
         
         if diag and 'centering_identity_check' in diag:
             diff_val = diag['centering_identity_check']
             pass_identity = diff_val < 1e-4 # Strict
             lines.append(f"- **Identity Check**: ||Mean(Batch) - I|| = {diff_val:.6f} {'✅ PASS' if pass_identity else '⚠️ HIGH'}")
    else:
         lines.append("Subject Centering: Disabled or Missing Stats.")
    lines.append("")

    # 6. Split Integrity
    lines.append("## 6. Split Integrity (Audit)")
    if diag and 'split_stats' in diag:
        ss = diag['split_stats']
        lines.append(f"- **Train**: {ss.get('n_train')}")
        lines.append(f"- **Val**: {ss.get('n_val')}")
        lines.append(f"- **Test**: {ss.get('n_test')}")
        lines.append(f"- **Intersection (Train/Val)**: {ss.get('train_val_intersection')} " + ("✅ PASS" if ss.get('train_val_intersection') == 0 else "❌ FAIL"))
    else:
        lines.append("Split stats missing.")
    lines.append("")

    
    # 7. Artifacts
    lines.append("## 7. Artifact Index")
    lines.append(f"- Metrics: `{metrics_path}`")
    lines.append(f"- Gate Stats: `{gate_path}`")
    lines.append(f"- Diagnostics: `{diag_path}`")
    
    out_path = os.path.join(root, "SINGLE_RUN_REPORT.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report generated: {out_path}")

if __name__ == "__main__":
    main()
