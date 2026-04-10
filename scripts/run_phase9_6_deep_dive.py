
import os
import sys
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix

class DeepDiveAnalyzer:
    def __init__(self, spatial_dir, manifold_dir, seeds=[0, 1, 2]):
        self.spatial_dir = spatial_dir
        self.manifold_dir = manifold_dir
        self.seeds = seeds
        self.results = []

    def run(self):
        print(f"=== Phase 9.6 Deep Dive Analysis (Seeds {self.seeds}) ===")
        for seed in self.seeds:
            self._analyze_seed(seed)
            
        self._aggregate_and_report()

    def _analyze_seed(self, seed):
        s_path = f"{self.spatial_dir}/spatial_trial_preds_seed{seed}.csv"
        m_path = f"{self.manifold_dir}/manifold5_trial_preds_seed{seed}.csv"
        
        if not os.path.exists(s_path) or not os.path.exists(m_path):
            print(f"[Seed {seed}] Missing files. Skipping.")
            return

        # Load
        df_s = pd.read_csv(s_path)
        df_m = pd.read_csv(m_path)
        
        # Ensure split column exists or infer
        if 'split' not in df_s.columns:
            df_s['split'] = df_s['trial_id'].apply(lambda x: 'train' if int(x.split('_t')[-1]) < 9 else 'test')
            
        # Merge
        # Standardize columns
        s = df_s.rename(columns={'prob_0': 's_p0', 'prob_1': 's_p1', 'prob_2': 's_p2', 'split': 'split_s', 'true_label': 'true_label_s'})
        m = df_m.rename(columns={'prob_0': 'm_p0', 'prob_1': 'm_p1', 'prob_2': 'm_p2', 'split': 'split_m', 'true_label': 'true_label_m'})
        
        merged = pd.merge(s, m, on='trial_id', how='inner')
        # Trust Manifold Split/Label (verified consistent)
        merged['split'] = merged['split_m']
        merged['true_label'] = merged['true_label_m']
        
        # --- Metrics ---
        
        # 1. Overfitting Gap
        train = merged[merged['split'] == 'train']
        test = merged[merged['split'] == 'test']
        
        acc_s_tr = accuracy_score(train['true_label'], self._pred(train, 's'))
        acc_s_te = accuracy_score(test['true_label'], self._pred(test, 's'))
        acc_m_tr = accuracy_score(train['true_label'], self._pred(train, 'm'))
        acc_m_te = accuracy_score(test['true_label'], self._pred(test, 'm'))
        
        # 2. Confidence Analysis (on Test)
        # "Confidence" = Max Prob
        # Metric: Mean Confidence Correct vs Mean Confidence Wrong
        conf_s_corr, conf_s_wrong = self._conf_stats(test, 's')
        conf_m_corr, conf_m_wrong = self._conf_stats(test, 'm')
        
        # 3. Disagreement (on Test)
        # When they disagree, who is right?
        dis_mask = self._pred(test, 's') != self._pred(test, 'm')
        disagreements = test[dis_mask]
        if len(disagreements) > 0:
            s_right = np.mean(self._pred(disagreements, 's') == disagreements['true_label'])
            m_right = np.mean(self._pred(disagreements, 'm') == disagreements['true_label'])
            both_wrong = np.mean((self._pred(disagreements, 's') != disagreements['true_label']) & 
                                 (self._pred(disagreements, 'm') != disagreements['true_label']))
        else:
            s_right, m_right, both_wrong = 0, 0, 0

        # 4. Weight Sensitivity (Fusion Landscape)
        # Calculate Acc vs W for Train and Test
        ws = np.linspace(0, 1, 21)
        train_accs = [self._fused_acc(train, w) for w in ws]
        test_accs = [self._fused_acc(test, w) for w in ws]
        
        best_w_train = ws[np.argmax(train_accs)]
        best_acc_train = np.max(train_accs)
        
        # What Acc did we get on Test using this "Best Train W"?
        # This is the "Selected" fusion performance
        eval_acc_test = self._fused_acc(test, best_w_train)
        
        # What was the ACTUAL best w on Test (Oracle)?
        best_w_oracle = ws[np.argmax(test_accs)]
        best_acc_oracle = np.max(test_accs)
        
        res = {
            "seed": seed,
            "S_Train_Acc": acc_s_tr, "S_Test_Acc": acc_s_te, "S_Gap": acc_s_tr - acc_s_te,
            "M_Train_Acc": acc_m_tr, "M_Test_Acc": acc_m_te, "M_Gap": acc_m_tr - acc_m_te,
            "S_Conf_Corr": conf_s_corr, "S_Conf_Wrong": conf_s_wrong,
            "M_Conf_Corr": conf_m_corr, "M_Conf_Wrong": conf_m_wrong,
            "Disagreement_Rate": np.mean(dis_mask),
            "Dis_S_Right": s_right, "Dis_M_Right": m_right, "Dis_Both_Wrong": both_wrong,
            "Best_W_Train": best_w_train, "Eval_Acc": eval_acc_test,
            "Oracle_W_Test": best_w_oracle, "Oracle_Acc": best_acc_oracle
        }
        self.results.append(res)
        
        # Print Summary for this Seed
        print(f"\n[Seed {seed}] Analysis:")
        print(f"  Overfitting: Spatial Gap={res['S_Gap']:.2f}, Manifold Gap={res['M_Gap']:.2f} (Manifold Train={res['M_Train_Acc']:.2f})")
        print(f"  Confidence (Test):")
        print(f"    Spatial:  Corr={res['S_Conf_Corr']:.2f}, Wrong={res['S_Conf_Wrong']:.2f}")
        print(f"    Manifold: Corr={res['M_Conf_Corr']:.2f}, Wrong={res['M_Conf_Wrong']:.2f} <- High avg confidence on wrong answers implies overconfidence.")
        print(f"  Disagreement (Rate={res['Disagreement_Rate']:.2f}):")
        print(f"    When disagreeing: Spatial Right={res['Dis_S_Right']:.2f}, Manifold Right={res['Dis_M_Right']:.2f}")
        print(f"  Fusion Landscape:")
        print(f"    Selected W (on Train) = {res['Best_W_Train']:.2f} -> Test Acc = {res['Eval_Acc']:.4f}")
        print(f"    Oracle W (on Test)    = {res['Oracle_W_Test']:.2f} -> Test Acc = {res['Oracle_Acc']:.4f}")
        
    def _pred(self, df, prefix):
        # Return predicted label for stream s or m
        probs = df[[f'{prefix}_p0', f'{prefix}_p1', f'{prefix}_p2']].values
        return np.argmax(probs, axis=1)
        
    def _conf_stats(self, df, prefix):
        # Mean max prob for correct vs wrong
        probs = df[[f'{prefix}_p0', f'{prefix}_p1', f'{prefix}_p2']].values
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        correct_mask = (preds == df['true_label'])
        
        mean_conf_corr = np.mean(confs[correct_mask]) if np.any(correct_mask) else 0
        mean_conf_wrong = np.mean(confs[~correct_mask]) if np.any(~correct_mask) else 0
        return mean_conf_corr, mean_conf_wrong

    def _fused_acc(self, df, w):
        sp = df[['s_p0', 's_p1', 's_p2']].values
        mp = df[['m_p0', 'm_p1', 'm_p2']].values
        fused = w*sp + (1-w)*mp
        pred = np.argmax(fused, axis=1)
        return accuracy_score(df['true_label'], pred)
        
    def _aggregate_and_report(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        
        print("\n=== Aggregate Deep Dive (Mean) ===")
        print(df.mean(numeric_only=True).round(4))
        
        out_path = "experiments/phase9_6_analysis/deep_dive_summary.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved full table to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial_dir", default="experiments/phase9_fusion/preds")
    parser.add_argument("--manifold_dir", default="experiments/phase9_5_multiband/preds")
    args = parser.parse_args()
    
    analyzer = DeepDiveAnalyzer(args.spatial_dir, args.manifold_dir)
    analyzer.run()
