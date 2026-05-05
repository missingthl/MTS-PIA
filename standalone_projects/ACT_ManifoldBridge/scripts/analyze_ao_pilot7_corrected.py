import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def bootstrap_ci(data, n_boot=2000, ci=0.95):
    if len(data) == 0: return [np.nan, np.nan]
    resamples = np.random.choice(data, (n_boot, len(data)), replace=True)
    means = np.mean(resamples, axis=1)
    lower = (1 - ci) / 2
    upper = 1 - lower
    return np.quantile(means, [lower, upper])

def wtl(deltas, tol=1e-5):
    wins = (deltas > tol).sum()
    ties = (deltas.abs() <= tol).sum()
    losses = (deltas < -tol).sum()
    return f"{wins}/{ties}/{losses}"

def report():
    # Paths
    ao_path = Path('results/csta_ao_pia_pilot7_v2/resnet1d_s123/per_seed_external.csv')
    u5_path = Path('results/csta_step3_diagnostic_sweep_etafix/resnet1d_s123/g0.1_e0.75/per_seed_external.csv')
    p1_path = Path('results/csta_external_baselines_phase1/resnet1d_s123/per_seed_external.csv')
    
    # 1. Load Data
    df_ao = pd.read_csv(ao_path)
    df_u5 = pd.read_csv(u5_path)
    df_p1 = pd.read_csv(p1_path)
    
    # Extract Specific Methods
    u5_baseline = df_u5[df_u5['method'] == 'csta_topk_uniform_top5'][['dataset', 'seed', 'aug_f1']].rename(columns={'aug_f1': 'u5_f1'})
    rand_cov = df_p1[df_p1['method'] == 'random_cov_state'][['dataset', 'seed', 'aug_f1']].rename(columns={'aug_f1': 'rand_cov_f1'})
    pca_cov = df_p1[df_p1['method'] == 'pca_cov_state'][['dataset', 'seed', 'aug_f1']].rename(columns={'aug_f1': 'pca_cov_f1'})
    wdba = df_p1[df_p1['method'] == 'wdba_sameclass'][['dataset', 'seed', 'aug_f1']].rename(columns={'aug_f1': 'wdba_f1'})
    
    if len(u5_baseline) == 0:
        raise ValueError("Could not find csta_topk_uniform_top5 rows in etafix baseline file!")
        
    # Pivot AO
    ao_fisher = df_ao[df_ao['method'] == 'csta_topk_uniform_top5_ao_fisher'][['dataset', 'seed', 'aug_f1']].rename(columns={'aug_f1': 'ao_fisher_f1'})
    ao_contrast = df_ao[df_ao['method'] == 'csta_topk_uniform_top5_ao_contrastive'][['dataset', 'seed', 'aug_f1']].rename(columns={'aug_f1': 'ao_contrast_f1'})
    
    # Merge Everything
    merged = u5_baseline.merge(ao_fisher, on=['dataset', 'seed'], how='inner')
    merged = merged.merge(ao_contrast, on=['dataset', 'seed'], how='inner')
    merged = merged.merge(rand_cov, on=['dataset', 'seed'], how='left')
    merged = merged.merge(pca_cov, on=['dataset', 'seed'], how='left')
    merged = merged.merge(wdba, on=['dataset', 'seed'], how='left')
    
    # 2. Analysis
    report_lines = [
        "# AO-PIA Pilot7 Corrected Report",
        "",
        "## Summary",
        f"Analyzed {len(merged)} dataset-seed pairs.",
        "",
        "## Comparison Tables",
        ""
    ]
    
    stats_rows = []
    
    for ao_col in ['ao_fisher_f1', 'ao_contrast_f1']:
        for ref_col in ['u5_f1', 'rand_cov_f1', 'pca_cov_f1', 'wdba_f1']:
            if ref_col not in merged.columns: continue
            
            sub = merged[[ao_col, ref_col]].dropna()
            if len(sub) == 0: continue
            
            delta = sub[ao_col] - sub[ref_col]
            
            # Seed-level W/T/L
            seed_wtl = wtl(delta)
            
            # Dataset-level W/T/L
            ds_delta = sub.groupby(merged['dataset'])[ao_col].mean() - sub.groupby(merged['dataset'])[ref_col].mean()
            ds_wtl = wtl(ds_delta)
            
            # Wilcoxon
            try:
                p_val = wilcoxon(sub[ao_col], sub[ref_col]).pvalue
            except:
                p_val = np.nan
                
            ci = bootstrap_ci(delta.to_numpy())
            
            stats_rows.append({
                "Method": ao_col,
                "Reference": ref_col,
                "Mean Delta": delta.mean(),
                "Median Delta": delta.median(),
                "Seed W/T/L": seed_wtl,
                "Dataset W/T/L": ds_wtl,
                "95% CI": f"[{ci[0]:.6f}, {ci[1]:.6f}]",
                "Wilcoxon p": p_val
            })
            
    stats_df = pd.DataFrame(stats_rows)
    # report_lines.append(stats_df.to_markdown(index=False))
    report_lines.append(stats_df.to_string(index=False))
    
    # 3. Save Files
    out_dir = Path('results/csta_ao_pia_pilot7_v2/resnet1d_s123')
    merged.to_csv(out_dir / 'ao_vs_canonical_head_to_head_raw.csv', index=False)
    stats_df.to_csv(out_dir / 'ao_vs_canonical_u5_corrected.csv', index=False)
    
    (out_dir / 'ao_pilot7_corrected_report.md').write_text("\n".join(report_lines))
    print(f"Report and CSVs generated in {out_dir}")
    print("\n" + stats_df.to_string())

if __name__ == "__main__":
    report()
