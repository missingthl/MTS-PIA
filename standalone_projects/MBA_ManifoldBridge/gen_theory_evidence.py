import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def gen_evidence_plots(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if empty
    if df.empty:
        print("Empty CSV. Skipping plots.")
        return

    # 1. Transport Accuracy vs Classification Gain
    plt.figure(figsize=(8, 6))
    for model in df['model'].unique():
        m_df = df[df['model'] == model]
        plt.scatter(m_df['transport_error_fro_mean'], m_df['gain'], label=model, alpha=0.7)
    
    plt.title("Proposition 1: Transport Fidelity vs Gain")
    plt.xlabel("Covariance Transport Error (Frobenius)")
    plt.ylabel("Macro F1 Gain")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{out_dir}/evidence_prop1_transport_vs_gain.png")
    plt.close()

    # 2. Radius Sensitivity Curve
    plt.figure(figsize=(8, 6))
    plt.scatter(df['safe_radius_ratio_mean'], df['gain'], alpha=0.5)
    plt.title("Proposition 2: Safe Radius Sensitivity")
    plt.xlabel("Safe Radius Ratio (Actual/Requested)")
    plt.ylabel("Macro F1 Gain")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{out_dir}/evidence_prop2_radius_sensitivity.png")
    plt.close()

    # 3. Host Response Heatmap (Mock with bar chart if heatmap is too complex without seaborn)
    pivot_df = df.pivot_table(index='dataset', columns='model', values='host_geom_cosine_mean', aggfunc='mean')
    if not pivot_df.empty:
        pivot_df.plot(kind='bar', figsize=(10, 6))
        plt.title("Proposition 3: Host-Geometry Alignment (Cosine)")
        plt.ylabel("Cosine Similarity")
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/evidence_prop3_host_alignment.png")
        plt.close()

    # 4. Local Manifold Margin vs Classification Gain
    # Question: Is MBA safer/more effective in sparse regions (large margin)?
    plt.figure(figsize=(8, 6))
    plt.scatter(df['manifold_margin_mean'], df['gain'], alpha=0.6, edgecolors='w')
    if len(df) > 1:
        # Simple trendline
        try:
            z = np.polyfit(df['manifold_margin_mean'], df['gain'], 1)
            p = np.poly1d(z)
            plt.plot(df['manifold_margin_mean'], p(df['manifold_margin_mean']), "r--", alpha=0.8)
        except: pass
    plt.title("Proposition 2: Local Manifold Margin vs Gain")
    plt.xlabel("Local Manifold Margin (d_min proxy)")
    plt.ylabel("Macro F1 Gain")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{out_dir}/evidence_prop2_margin_vs_gain.png")
    plt.close()

    print(f"Evidence plots saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results/theory_evidence")
    args = parser.parse_args()
    gen_evidence_plots(args.csv, args.out_dir)
