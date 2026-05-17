import pandas as pd
import glob
import os

def generate_grand_summary():
    print("Generating Final Backbone Robustness Summary (V4 - Best-of-All)...")
    
    backbones = ["resnet1d", "moderntcn", "minirocket", "patchtst", "timesnet"]
    grand_dfs = {}

    for bb in backbones:
        all_files = []
        if bb == "resnet1d":
            all_files = ["results/full_scale_resnet1d_v1/per_seed_external.csv"]
        elif bb == "moderntcn":
            all_files = ["results/moderntcn_final20_robustness_v1/per_seed_external_REBUILT.csv"]
        elif bb == "minirocket":
            all_files = glob.glob("results/minirocket_final20_*/per_seed_external*.csv")
        elif bb == "patchtst":
            all_files = glob.glob("results/patchtst_final20_v1/**/per_seed_external*.csv", recursive=True)
        elif bb == "timesnet":
            all_files = glob.glob("results/timesnet_final20_v1/**/per_seed_external*.csv", recursive=True)

        dfs = []
        for f in all_files:
            try:
                temp = pd.read_csv(f)
                if 'dataset' in temp.columns:
                    dfs.append(temp)
            except: continue
        
        if not dfs: continue
        
        df = pd.concat(dfs, ignore_index=True)
        # Sort by F1 to keep the most complete/successful one if duplicates exist
        df = df.sort_values(by=['aug_f1'], ascending=False)
        df = df.drop_duplicates(subset=['dataset', 'seed', 'method'])
        grand_dfs[bb] = df

    summary_rows = []
    csta_name = 'csta_topk_uniform_top5'

    for name, df in grand_dfs.items():
        df = df.copy()
        df['method'] = df['method'].replace({csta_name: 'csta_u5'})
        
        pivot = df.pivot_table(index=['dataset', 'seed'], columns='method', values='aug_f1').reset_index()
        if 'no_aug' not in pivot.columns or 'csta_u5' not in pivot.columns:
            continue
            
        pivot = pivot.dropna(subset=['no_aug', 'csta_u5'])
        delta = pivot['csta_u5'] - pivot['no_aug']
        win = len(delta[delta > 0.001])
        loss = len(delta[delta < -0.001])
        tie = len(pivot) - win - loss
        
        summary_rows.append({
            'Backbone': name.capitalize(),
            'Datasets': pivot['dataset'].nunique(),
            'N_Pairs': len(pivot),
            'Mean_Delta': delta.mean(),
            'W/T/L': f"{win}/{tie}/{loss}",
            'Win_Rate': f"{win / len(pivot) * 100:.1f}%" if len(pivot) > 0 else "0%"
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("results/grand_robustness_summary_final", exist_ok=True)
    summary_df.to_csv("results/grand_robustness_summary_final/grand_robustness_summary.csv", index=False)
    
    with open("results/grand_robustness_summary_final/FINAL_PAPER_EVIDENCE.md", "w") as f:
        f.write("# FINAL PAPER EVIDENCE: Grand Backbone Robustness Summary\n\n")
        f.write("| Backbone | Datasets | N_Pairs | Mean_Delta | W/T/L | Win_Rate |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['Backbone']} | {row['Datasets']} | {row['N_Pairs']} | {row['Mean_Delta']:+.4f} | {row['W/T/L']} | {row['Win_Rate']} |\n")

    print("Final grand summary artifacts generated.")

if __name__ == "__main__":
    generate_grand_summary()
