
import os
import sys
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class StrictFusionEngine:
    """
    Implements STRICT fusion protocol:
    1. Weight Selection on Train/Val ONLY (via 5-Fold CV on Training set).
    2. Test Set LOCKED until evaluation.
    3. Trial-level metrics.
    """
    def __init__(self, spatial_preds_path, manifold_preds_path, seed):
        self.seed = seed
        self.spatial = pd.read_csv(spatial_preds_path)
        self.manifold = pd.read_csv(manifold_preds_path)
        
        # Filter for current seed
        self.spatial = self.spatial[self.spatial['seed'] == seed].copy()
        self.manifold = self.manifold[self.manifold['seed'] == seed].copy()
        
        # Merge
        # Spatial cols: seed,split,trial_id,true_label,pred_label,mean_prob_max,prob_0,prob_1,prob_2,n_windows
        # Manifold cols: seed,stream,subject,session,trial_id... prob_0,prob_1,prob_2...
        
        # We need to ensure trial_id matches.
        # Spatial Phase 9 export trial_ids are like "1_s1_t0".
        # Manifold Phase 9.5 export trial_ids are like "1_s1_t0".
        
        self.merged = self._merge_preds()
        
    def _merge_preds(self):
        if 'split' not in self.spatial.columns:
            # We don't infer, we ignore. We rely on Manifold split.
            pass
            
        # Rename probs for clarity
        s = self.spatial.rename(columns={'prob_0': 's_p0', 'prob_1': 's_p1', 'prob_2': 's_p2', 'split': 'split_s'})
        m = self.manifold.rename(columns={'prob_0': 'm_p0', 'prob_1': 'm_p1', 'prob_2': 'm_p2', 'split': 'split_m'})
        
        # Select relevant columns for merge
        # merge_on = ['trial_id', 'split', 'true_label'] 
        # merging on split/true_label can fail if mismatched.
        # We merge on trial_id.
        
        # Inner join
        merged = pd.merge(s, m, on='trial_id', suffixes=('_s', '_m'), how='inner')
        
        # Verify length
        if len(merged) != len(s):
            print(f"[Seed {self.seed}] Warning: Spatial rows {len(s)} != Merged {len(merged)}")
            # This is expected if Spatial has fewer (e.g. only 270 vs 675?)
            # But we only care about the intersection.
        
        # Use Manifold split as definitive
        merged['split'] = merged['split_m']
        merged['true_label'] = merged['true_label_m']
        
        return merged
        
    def select_weight_cv(self, n_splits=5):
        """
        Grid search w using CV on Training set.
        Returns best_w.
        """
        train_df = self.merged[self.merged['split'] == 'train'].reset_index(drop=True)
        
        if len(train_df) == 0:
            raise ValueError("No training data found in merged dataframe!")
            
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        ws = np.linspace(0, 1, 21) # 0.0 to 1.0 step 0.05
        avg_accs = []
        
        # Convert to numpy for speed
        s_probs = train_df[['s_p0', 's_p1', 's_p2']].values
        m_probs = train_df[['m_p0', 'm_p1', 'm_p2']].values
        y_true = train_df['true_label'].values
        
        for w in ws:
            fold_accs = []
            for tr_idx, val_idx in kf.split(train_df):
                # We interpret w as weight for Spatial (bias towards strong model)
                # but usually w is for "model 2" vs "model 1".
                # Let's define fused = w * Spatial + (1-w) * Manifold
                
                # Spatial (s_probs) is primary.
                # w=1.0 -> Spatial Only. w=0.0 -> Manifold Only.
                
                sp_val = s_probs[val_idx]
                mp_val = m_probs[val_idx]
                y_val = y_true[val_idx]
                
                fused_probs = w * sp_val + (1 - w) * mp_val
                pred = np.argmax(fused_probs, axis=1)
                acc = accuracy_score(y_val, pred)
                fold_accs.append(acc)
                
            avg_accs.append(np.mean(fold_accs))
            
        # Select best w
        best_idx = np.argmax(avg_accs)
        best_w = ws[best_idx]
        best_cv_acc = avg_accs[best_idx]
        
        print(f"[Seed {self.seed}] Best w_spatial={best_w:.2f} (CV Acc: {best_cv_acc:.4f})")
        return best_w
        
    def evaluate_test(self, w):
        test_df = self.merged[self.merged['split'] == 'test'].reset_index(drop=True)
        
        s_probs = test_df[['s_p0', 's_p1', 's_p2']].values
        m_probs = test_df[['m_p0', 'm_p1', 'm_p2']].values
        y_true = test_df['true_label'].values
        
        # Individual Acc
        s_pred = np.argmax(s_probs, axis=1)
        m_pred = np.argmax(m_probs, axis=1)
        acc_s = accuracy_score(y_true, s_pred)
        acc_m = accuracy_score(y_true, m_pred)
        
        # Fused Acc
        fused_probs = w * s_probs + (1 - w) * m_probs
        f_pred = np.argmax(fused_probs, axis=1)
        acc_f = accuracy_score(y_true, f_pred)
        
        # Complementarity
        both_correct = np.mean((s_pred == y_true) & (m_pred == y_true))
        s_only = np.mean((s_pred == y_true) & (m_pred != y_true))
        m_only = np.mean((s_pred != y_true) & (m_pred == y_true))
        both_wrong = np.mean((s_pred != y_true) & (m_pred != y_true))
        disagreement = np.mean(s_pred != m_pred)
        
        metrics = {
            "seed": self.seed,
            "w_spatial": w,
            "acc_spatial": acc_s,
            "acc_manifold": acc_m,
            "acc_fused": acc_f,
            "both_correct": both_correct,
            "spatial_only": s_only,
            "manifold_only": m_only,
            "both_wrong": both_wrong,
            "disagreement": disagreement
        }
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial_dir", type=str, default="experiments/phase9_fusion/preds")
    parser.add_argument("--manifold_dir", type=str, default="experiments/phase9_5_multiband/preds")
    args = parser.parse_args()
    
    seeds = [0, 1, 2]
    all_metrics = []
    
    print("\n=== Phase 9.5 Strict Fusion Analysis ===")
    
    for seed in seeds:
        s_path = f"{args.spatial_dir}/spatial_trial_preds_seed{seed}.csv"
        m_path = f"{args.manifold_dir}/manifold5_trial_preds_seed{seed}.csv"
        
        if not os.path.exists(s_path) or not os.path.exists(m_path):
            print(f"Skipping Seed {seed}: Missing files.")
            continue
            
        engine = StrictFusionEngine(s_path, m_path, seed)
        best_w = engine.select_weight_cv(n_splits=5)
        metrics = engine.evaluate_test(best_w)
        all_metrics.append(metrics)
        
        print(f"Seed {seed}: S={metrics['acc_spatial']:.4f} M={metrics['acc_manifold']:.4f} -> F={metrics['acc_fused']:.4f} (Disagreement: {metrics['disagreement']:.4f})")
        
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        print("\n=== Aggregate Results ===")
        print(df.describe().loc[['mean', 'std']])
        
        out_path = "experiments/phase9_5_multiband/reports/fusion_summary.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved report to {out_path}")
