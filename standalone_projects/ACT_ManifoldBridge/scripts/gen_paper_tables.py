import pandas as pd
import glob
import os
import numpy as np
import argparse

def aggregate_paper_results(root_dir='results/paper_matrix_v2_final'):
    # Search for final_results.csv in phase_all
    search_path = os.path.join(root_dir, 'phase_all/*/*/*/final_results.csv')
    all_files = glob.glob(search_path)
    if not all_files:
        print(f"No results found yet in {search_path}")
        return

    data = []
    for f in all_files:
        parts = f.split('/')
        # Path: root_dir/phase_all/{model}/{dataset}/{s1_lraes_safe}/final_results.csv
        # Parts usually: ['results', 'paper_matrix_v2_final', 'phase_all', '{model}', '{dataset}', '{condition}', 'final_results.csv']
        # But split by / depends on how root_dir was passed.
        # Let's count from the end to be safe.
        model = parts[-4]
        dataset = parts[-3]
        condition_str = parts[-2] # e.g. s1_lraes_safe
        
        cond_parts = condition_str.split('_')
        seed = int(cond_parts[0].replace('s', ''))
        algo = cond_parts[1]
        safe_mode = cond_parts[2]
        
        df = pd.read_csv(f)
        df['model'] = model
        df['dataset'] = dataset
        df['seed'] = seed
        df['algo'] = algo
        df['safe_mode'] = safe_mode
        data.append(df)
        
    full_df = pd.concat(data)
    
    # 1. Main Results Table (Tab 1)
    # ACT (lraes) + Safe (safe) + up to 3 seeds
    main_subset = full_df[(full_df['algo'] == 'lraes') & (full_df['safe_mode'] == 'safe')]
    if not main_subset.empty:
        tab1 = main_subset.groupby(['dataset', 'model']).agg({
            'base_f1': ['mean', 'std'],
            'act_f1': ['mean', 'std'],
            'gain': ['mean', 'std']
        }).reset_index()
        tab1.columns = ['dataset', 'model', 'base_mean', 'base_std', 'act_mean', 'act_std', 'gain_mean', 'gain_std']
        tab1.to_csv(os.path.join(root_dir, 'tab1_main_results.csv'), index=False)
    
    # 2. Direction Ablation (Tab 2)
    dir_subset = full_df[(full_df['safe_mode'] == 'safe') & (full_df['seed'] == 1)]
    if not dir_subset.empty:
        tab2 = dir_subset.pivot_table(index=['dataset', 'model'], columns='algo', values='gain').reset_index()
        if 'lraes' in tab2.columns and 'pia' in tab2.columns:
            tab2.rename(columns={'lraes': 'ACT_Bank_Gain', 'pia': 'Random_Bank_Gain'}, inplace=True)
            tab2.to_csv(os.path.join(root_dir, 'tab2_direction_ablation.csv'), index=False)

    # 3. Safe-Step Ablation (Tab 3)
    safe_subset = full_df[(full_df['algo'] == 'lraes') & (full_df['seed'] == 1)]
    if not safe_subset.empty:
        tab3 = safe_subset.pivot_table(index=['dataset', 'model'], columns='safe_mode', values='gain').reset_index()
        if 'safe' in tab3.columns and 'nosafe' in tab3.columns:
            tab3.rename(columns={'safe': 'SafeStep_ON_Gain', 'nosafe': 'SafeStep_OFF_Gain'}, inplace=True)
            tab3.to_csv(os.path.join(root_dir, 'tab3_safe_ablation.csv'), index=False)

    # 4. Bridge Evidence (Tab 4)
    if not main_subset.empty:
        tab4 = main_subset.groupby(['dataset', 'model']).agg({
            'transport_error_fro_mean': 'mean',
            'transport_error_logeuc_mean': 'mean',
            'bridge_cond_A_mean': 'mean',
            'metric_preservation_error_mean': 'mean'
        }).reset_index()
        tab4.to_csv(os.path.join(root_dir, 'tab4_bridge_evidence.csv'), index=False)
    
    print(f"Aggregation complete for {root_dir}. Tables saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='results/paper_matrix_v2_final')
    args = parser.parse_args()
    aggregate_paper_results(args.root_dir)
