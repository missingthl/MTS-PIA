import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

def generate_paper_plots(root_dir='results/paper_matrix_v2_final'):
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('bmh')
    sns.set_context('paper', font_scale=1.4)

    # Search for all final_results.csv
    all_rows = []
    files = glob.glob(os.path.join(root_dir, 'phase_all/*/*/*/final_results.csv'))
    print(f"Found {len(files)} result files.")

    for f in files:
        try:
            df_temp = pd.read_csv(f)
            # Path expected: root/phase_all/{model}/{dataset}/{cond}/final_results.csv
            parts = f.split('/')
            # Find the index of phase_all and count forward
            p_idx = parts.index('phase_all')
            model = parts[p_idx + 1]
            dataset = parts[p_idx + 2]
            
            df_temp['model_name'] = model
            df_temp['dataset_name'] = dataset
            all_rows.append(df_temp)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not all_rows:
        print("No valid results found.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    out_dir = 'results/viz_harvest/plots/theory'
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Prop 1
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='transport_error_fro_mean', y='gain', hue='model_name', alpha=0.6, s=120)
    plt.title('Proposition 1: Transport Fidelity vs Gain')
    plt.xlabel('Transport Error (Frobenius) ↓')
    plt.ylabel('Macro F1 Gain ↑')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(out_dir, 'prop1_fidelity_vs_gain.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Prop 3
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='host_geom_cosine_mean', y='gain', hue='model_name', style='model_name', s=150, alpha=0.8)
    plt.axvline(0, color='red', linestyle='--', alpha=0.3)
    plt.title('Proposition 3: Host-Geometry Alignment vs Gain')
    plt.xlabel('Host-Geometry Alignment (Cosine) ↑')
    plt.ylabel('Macro F1 Gain ↑')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(out_dir, 'prop3_alignment_vs_gain.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Publication-quality theory plots saved to {out_dir}")

if __name__ == '__main__':
    generate_paper_plots()
