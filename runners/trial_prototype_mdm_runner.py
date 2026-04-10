
import os
import torch
import numpy as np
import pandas as pd
import json
import traceback
from torch.utils.data import DataLoader
from runners.manifold_deep_runner import ManifoldDeepRunner
from models.prototype_mdm import logm_spd, expm_sym, logeuclid_mean, logeuclid_dist_log_domain

class TrialPrototypeMDMRunner(ManifoldDeepRunner):
    def __init__(self, args, num_classes=3):
        super().__init__(args, num_classes)
        # Force these settings for Step 6
        self.matrix_mode = 'corr'
        self.global_centering = True
        self.log_ref_global = None # Will store logm(C_ref)
        self.prototypes_log = None # Will store logm(P_y) (3, 62, 62)
        
    def fit_predict(self, fold, fold_name="fold"):
        """
        Trial-Level Prototype MDM.
        1. Compute Global Mean (Train Only - Window Level)
        2. Compute Trial Matrices (Aggregate Centered Windows -> Trial Mean)
        3. Compute Class Prototypes (Train Trials Only)
        4. Predict (Test Trials)
        """
        print(f"[{fold_name}] Starting Trial-Level Prototype MDM (Log-Euclidean)...")
        
        # 1. Setup Data
        X_tr_list = fold.trials_train
        y_tr_list = fold.y_trial_train
        X_te_list = fold.trials_test
        y_te_list = fold.y_trial_test
        
        # Sub-split 80/20 on Train (Identical to Step 4/5)
        n_train_total = len(X_tr_list)
        n_sub_train = int(0.8 * n_train_total)
        
        # Shuffle (Relies on set_seed in main script)
        perm = np.random.permutation(n_train_total)
        tr_indices_sub = perm[:n_sub_train]
        
        X_train_sub = [X_tr_list[i] for i in tr_indices_sub]
        y_train_sub = [y_tr_list[i] for i in tr_indices_sub]
        
        # We need loaders to compute Global Mean (Window Level)
        from runners.manifold_deep_runner import TrialDataset 
        
        window_len = 24
        stride = 12
        return_5band = (self.bands_mode == "all5_timecat")
        dataset_norm_mode = "none" if self.band_norm_mode in ["manual_per_band_time", "per_band_global_z"] else self.band_norm_mode
        
        # Helper to make generic loader
        def make_loader(X_list, y_list, shuffle=False):
             dset = TrialDataset(X_list, y_list, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=self.bands_mode, band_norm_mode=dataset_norm_mode)
             # Batch size large for speed
             return DataLoader(dset, batch_size=64, shuffle=shuffle)

        train_loader = make_loader(X_train_sub, y_train_sub, shuffle=True)
        # For trial aggregation, we need ORDERED loaders (shuffle=False) and we need to group by trial_id
        # TrialDataset returns trial_id in __getitem__?
        # Yes: X, y, trial_id
        
        # 2. Compute Global Mean (Train Only - Window Level)
        # Reuse parent method
        self.log_ref_global = self._compute_global_log_euclidean_mean(train_loader, fold_name) # Returns logm(C_ref)
        
        # 3. Compute Trial Matrices (Train)
        # We need a loader that yields windows, but we need to know which windows belong to which trial.
        # TrialDataset generates windows on the fly.
        # The `trial_id` returned by dataset helps us group.
        
        train_loader_ordered = make_loader(X_train_sub, y_train_sub, shuffle=False)
        print("Aggregating Train Trials...")
        train_mats, train_labels, train_ids = self._compute_trial_matrices(train_loader_ordered)
        
        # 4. Compute Class Prototypes (Train Trials Only)
        self._compute_trial_prototypes(train_mats, train_labels, fold_name)
        
        # 5. Predict (Test Trials)
        # Use Test Split
        test_loader = make_loader(X_te_list, y_te_list, shuffle=False)
        print("Aggregating Test Trials...")
        test_mats, test_labels, test_ids = self._compute_trial_matrices(test_loader)
        
        test_metrics, test_preds = self._predict_trials(test_mats, test_labels, test_ids, fold_name)
        
        # 6. Save results
        save_dir = f"promoted_results/{fold_name.split('/report')[0]}"
        os.makedirs(save_dir, exist_ok=True)
        
        test_preds.to_csv(f"{save_dir}/manifold_trial_pred.csv", index=False)
        
        with open(f"{save_dir}/report_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
            
        print(f"Trial-Level Prototype MDM Finished. Acc: {test_metrics['trial_acc']:.4f}")
        return test_metrics


    def _compute_trial_matrices(self, loader):
        """
        Aggregate windows -> Trial Matrix.
        1. Compute C_win.
        2. Global Center C_win -> C_win_centered.
        3. Group by trial_id.
        4. Mean(C_win_centered) -> C_trial.
        """
        trial_accum = {} # tid -> [matrices]
        trial_labels = {}
        
        with torch.no_grad():
             for Xb, yb, tib in loader:
                 Xb = Xb.to(self.device).double()
                 tib = tib.numpy()
                 yb = yb.numpy()
                 
                 # Preprocess
                 if Xb.ndim == 4:
                      Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()
                      if self.band_norm_mode == "per_band_global_z":
                            mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                            std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                            Xb_perm = (Xb_perm - mean) / std
                      model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                 else:
                      model_in = Xb
                      
                 # Matrix
                 mat = self._compute_matrix_from_input(model_in)
                 mat = mat + torch.eye(mat.size(1), device=self.device).double() * self.spd_eps
                 
                 # Global Center
                 # Cg = expm( logm(C) - logm(C_ref) )
                 # Note: Step 4 defined centering as subtracting log_ref in log domain.
                 # "Global-center in log domain: Cg = expm( logm(C) - logm(C_ref) )" (From Step 5 Request)
                 # Wait, Step 5 request said that. Step 4 implementation was:
                 # "log_centered = log_mats - log_ref; C_centered = expm(log_centered)"
                 # Yes, that matches.
                 
                 log_C = logm_spd(mat)
                 log_ref = self.log_ref_global
                 if log_ref.ndim == 2: log_ref = log_ref.unsqueeze(0)
                 
                 log_centered = log_C - log_ref
                 
                 # Expm to get back to SPD domain for aggregation
                 # "C_trial = mean(C_win_centered)"
                 C_centered = expm_sym(log_centered).cpu()
                 
                 for i, tid in enumerate(tib):
                     if tid not in trial_accum:
                         trial_accum[tid] = []
                         trial_labels[tid] = yb[i]
                     trial_accum[tid].append(C_centered[i])
                     
        # Aggregate
        trial_mats = []
        labels = []
        ids = []
        
        sorted_tids = sorted(trial_accum.keys())
        for tid in sorted_tids:
             mats = torch.stack(trial_accum[tid]) # (N_win, 62, 62)
             # Arithmetic Mean
             mean_mat = mats.mean(dim=0)
             # Symmetrize check
             mean_mat = 0.5 * (mean_mat + mean_mat.transpose(-1, -2))
             # Add eps just in case aggregation drift
             mean_mat = mean_mat + torch.eye(62).double() * 1e-6
             
             trial_mats.append(mean_mat)
             labels.append(trial_labels[tid])
             ids.append(tid)
             
        return torch.stack(trial_mats), np.array(labels), np.array(ids)

    def _compute_trial_prototypes(self, mats, labels, fold_name):
        print("Computing Trial Prototypes...")
        class_mats = {0: [], 1: [], 2: []}
        
        for i, y in enumerate(labels):
            class_mats[y].append(mats[i])
            
        self.prototypes_log = {}
        counts = {}
        
        # Log-Euclidean Mean of Trial Matrices
        # P_y = logeuclid_mean({C_trial})
        for cls in [0, 1, 2]:
            ms = torch.stack(class_mats[cls]).to(self.device).double()
            mean_C, mean_log = logeuclid_mean(ms)
            self.prototypes_log[cls] = mean_log # Store log for distance
            counts[cls] = len(class_mats[cls])
            
        diag = {
            "counts": counts,
            "aggregation": "arithmetic_mean_of_centered_windows",
            "global_centering": True
        }
        
        save_dir = f"promoted_results/{fold_name.split('/report')[0]}/global_centered_corr"
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/prototypes_trial_meta.json", "w") as f:
            json.dump(diag, f, indent=2)
            
        with open(f"{save_dir}/../report_diagnostics.json", "w") as f:
             json.dump(diag, f, indent=2)
             
        print(f"Trial Prototypes Computed. Counts: {counts}")

    def _predict_trials(self, mats, labels, ids, fold_name):
        print("Predicting Trials...")
        
        mats = mats.to(self.device).double()
        # Compute Distances
        # logm(C_trial) - logm(P_y)
        
        log_C = logm_spd(mats) # (N_trial, 62, 62)
        
        protos_log_stack = torch.stack([self.prototypes_log[i] for i in [0,1,2]])
        
        dists = logeuclid_dist_log_domain(log_C, protos_log_stack) # (N_trial, 3)
        
        preds = torch.argmin(dists, dim=1).cpu().numpy()
        probs = dists.cpu().numpy()
        
        acc = (preds == labels).mean()
        
        df = pd.DataFrame({
            "trial_id": ids,
            "true_label": labels,
            "pred_label": preds,
            "d0": probs[:, 0],
            "d1": probs[:, 1],
            "d2": probs[:, 2]
        })
        
        metrics = {
            "trial_acc": acc,
            "count": len(preds)
        }
        
        return metrics, df
