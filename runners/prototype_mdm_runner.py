
import os
import torch
import numpy as np
import pandas as pd
import json
import traceback
from torch.utils.data import DataLoader
from runners.manifold_deep_runner import ManifoldDeepRunner
from models.prototype_mdm import logm_spd, expm_sym, logeuclid_mean, logeuclid_dist_log_domain

class PrototypeMDMRunner(ManifoldDeepRunner):
    def __init__(self, args, num_classes=3):
        super().__init__(args, num_classes)
        # Force these settings for Step 5
        self.matrix_mode = 'corr'
        self.global_centering = True
        self.log_ref_global = None # Will store logm(C_ref)
        self.prototypes_log = None # Will store logm(P_y) (3, 62, 62)
        
    def fit_predict(self, fold, fold_name="fold"):
        """
        Modified fit_predict for Prototype MDM.
        1. Compute Global Mean (Train Only)
        2. Compute Class Prototypes (Train Only)
        3. Predict (Test Only)
        """
        print(f"[{fold_name}] Starting Prototype MDM (Log-Euclidean)...")
        
        # 1. Setup Data
        X_tr_list = fold.trials_train
        y_tr_list = fold.y_trial_train
        X_te_list = fold.trials_test
        y_te_list = fold.y_trial_test
        
        # Sub-split 80/20 on Train (Identical to Step 4)
        n_train_total = len(X_tr_list)
        n_sub_train = int(0.8 * n_train_total)
        
        # Shuffle (Relies on set_seed in main script)
        perm = np.random.permutation(n_train_total)
        tr_indices_sub = perm[:n_sub_train]
        
        X_train_sub = [X_tr_list[i] for i in tr_indices_sub]
        y_train_sub = [y_tr_list[i] for i in tr_indices_sub]

        

        
        # Loaders
        # Need TrialDataset class. It's in runners.manifold_deep_runner but not exported nicely?
        # It is defined in the file. I can import it if I move it or just copy it.
        # Since I'm in a new file, I can't easily import `TrialDataset` if it's not in `__all__` or separate file.
        # I'll rely on `from runners.manifold_deep_runner import TrialDataset` if possible.
        # Let's check imports.
        pass # Will assume I can import it or define a simple one.
        
        # Actually `TrialDataset` is importable if it's at top level.
        from runners.manifold_deep_runner import TrialDataset 
        
        window_len = 24
        stride = 12
        return_5band = (self.bands_mode == "all5_timecat")
        dataset_norm_mode = "none" if self.band_norm_mode in ["manual_per_band_time", "per_band_global_z"] else self.band_norm_mode
        
        train_dset = TrialDataset(X_train_sub, y_train_sub, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=self.bands_mode, band_norm_mode=dataset_norm_mode)
        test_dset = TrialDataset(X_te_list, y_te_list, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=self.bands_mode, band_norm_mode=dataset_norm_mode)
        
        batch_size = 32
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
        
        # 2. Compute Global Mean (Train Only)
        # Use parent method (I need to ensure it's available)
        # Note: ManifoldDeepRunner.fit_predict computes it and stores in self.log_ref_global
        self.log_ref_global = self._compute_global_log_euclidean_mean(train_loader, fold_name) # Returns logm(C_ref)
        
        # 3. Compute Class Prototypes (Train Only)
        self._compute_prototypes(train_loader, fold_name)
        
        # 4. Predict
        test_metrics, test_preds, test_agg = self._predict_prototypes(test_loader, fold_name, mode="test")
        
        # 5. Save Results
        # Similar to parent
        save_dir = f"promoted_results/{fold_name.split('/report')[0]}"
        os.makedirs(save_dir, exist_ok=True)
        
        test_preds.to_csv(f"{save_dir}/manifold_window_pred.csv", index=False)
        test_agg.to_csv(f"{save_dir}/manifold_trial_pred.csv", index=False)
        
        with open(f"{save_dir}/report_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
            
        print(f"Prototype MDM Finished. Acc: {test_metrics['trial_agg_acc']:.4f}")
        return test_metrics


    def _compute_prototypes(self, loader, fold_name):
        print("Computing Class Prototypes...")
        # Accumulate logm(C_centered) per class
        # Center: log(C) - log(C_ref)
        
        class_logs = {0: [], 1: [], 2: []}
        
        with torch.no_grad():
            for Xb, yb, _ in loader:
                Xb = Xb.to(self.device).double()
                yb = yb.numpy()
                
                # Preprocess (Shape)
                if Xb.ndim == 4: # (B, Win, 62, 5) -> (B, 62, 5*Win)
                     # Same logic as parent
                     Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()
                     if self.band_norm_mode == "per_band_global_z":
                           mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                           std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                           Xb_perm = (Xb_perm - mean) / std
                     model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                else:
                     model_in = Xb
                
                # Matrix
                mat = self._compute_matrix_from_input(model_in) # (B, C, C)
                # Add eps
                mat = mat + torch.eye(mat.size(1), device=self.device).double() * self.spd_eps
                
                # Log Global Center
                # logC_centered = logm(C) - log_ref
                log_C = logm_spd(mat)
                log_ref = self.log_ref_global
                if log_ref.ndim == 2: log_ref = log_ref.unsqueeze(0)
                
                log_centered = log_C - log_ref # (B, 62, 62)
                
                for i, y in enumerate(yb):
                    class_logs[y].append(log_centered[i].cpu())

        # Compute Mean per class
        self.prototypes_log = {}
        counts = {}
        
        for cls in [0, 1, 2]:
            logs = torch.stack(class_logs[cls]).to(self.device) # (N_c, 62, 62)
            mean_log = logs.mean(dim=0) # (62, 62)
            self.prototypes_log[cls] = mean_log
            counts[cls] = len(class_logs[cls])
            
        # Diagnostics: Identity Norm of Centered Data
        # Average norm of all centered logs?
        # Or mean of means?
        # The user asked for || mean_train( logm(Cg) ) ||_F
        all_logs_cat = torch.cat([torch.stack(class_logs[c]) for c in [0,1,2]], dim=0)
        mean_all_log = all_logs_cat.mean(dim=0) # Global mean of centered data (should be close to 0 if C_ref is global mean)
        log_identity_norm = torch.norm(mean_all_log).item()
        
        print(f"Prototypes Computed. Counts: {counts}. IdNorm: {log_identity_norm:.4e}")
        
        # Save
        save_dir = f"promoted_results/{fold_name.split('/report')[0]}/global_centered_corr"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save prototypes (as Log Matrices? Or as Expm?)
        # User: "Save prototypes... prototypes.npy".
        # MDM usually stores the Point on Manifold.
        # But we do calculation in Tangent Space (Log Domain).
        # Let's save the LOG maps for efficiency/reproducibility of the classifier state.
        # But typically "Prototype" implies the SPD matrix.
        # Let's save SPD prototypes: P_y = expm(mean_log) AND log prototypes?
        # User defined: "P_y = logeuclid_mean({Cg ...})". This returns SPD.
        # "Predict by nearest-prototype under Log-Euclidean distance".
        # logeuclid_dist(C, P) = || logm(C) - logm(P) ||.
        # If we save SPD P, we have to logm it again.
        # Let's save SPD for artifact validity, but keep self.prototypes_log for inference.
        
        protos_spd = []
        for cls in [0, 1, 2]:
            protos_spd.append(expm_sym(self.prototypes_log[cls]))
        protos_spd = torch.stack(protos_spd).cpu().numpy()
        
        np.save(f"{save_dir}/prototypes.npy", protos_spd)
        
        diag = {
            "counts": counts,
            "log_identity_norm": log_identity_norm,
            "global_centering": True
        }
        with open(f"{save_dir}/prototypes_meta.json", "w") as f:
            json.dump(diag, f, indent=2)
            
        with open(f"{save_dir}/../report_diagnostics.json", "w") as f:
             json.dump(diag, f, indent=2)

    def _predict_prototypes(self, loader, fold_name, mode="test"):
        print(f"Predicting ({mode})...")
        
        all_preds = []
        all_labels = []
        all_trial_ids = []
        all_probs = [] # Distances actually
        
        # Stack prototypes log
        protos_log_stack = torch.stack([self.prototypes_log[i] for i in [0,1,2]]) # (3, 62, 62)
        
        # Get trial IDs from dataset if available
        # My dataset setup above didn't explicitly pass trial_ids correctly via loader iteration?
        # `TrialDataset` __getitem__ returns (X, y, trial_id)
        
        with torch.no_grad():
            for Xb, yb, tib in loader:
                Xb = Xb.to(self.device).double()
                
                # Preprocess (Same as above)
                if Xb.ndim == 4:
                     Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()
                     if self.band_norm_mode == "per_band_global_z":
                           mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                           std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                           Xb_perm = (Xb_perm - mean) / std
                     model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                else:
                     model_in = Xb
                
                mat = self._compute_matrix_from_input(model_in)
                mat = mat + torch.eye(mat.size(1), device=self.device).double() * self.spd_eps
                
                # Log Global Center
                log_C = logm_spd(mat)
                log_ref = self.log_ref_global
                if log_ref.ndim == 2: log_ref = log_ref.unsqueeze(0)
                log_centered = log_C - log_ref 
                
                # Distances to Prototypes
                # log_centered: (B, 62, 62)
                # protos_log_stack: (3, 62, 62)
                # dists: (B, 3)
                dists = logeuclid_dist_log_domain(log_centered, protos_log_stack)
                
                # Predict argmin
                preds = torch.argmin(dists, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
                all_trial_ids.extend(tib.numpy())
                all_probs.extend(dists.cpu().numpy())
                
        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_trial_ids = np.array(all_trial_ids)
        all_probs = np.array(all_probs)
        
        win_acc = (all_preds == all_labels).mean()
        
        # Trial Aggregation (Mean Distance?)
        # Usually we aggregage probabilities/logits. Here we have distances.
        # Smaller is better.
        # "Aggregate window->trial using the existing aggregation_method=mean"
        # Since smaller distance = higher probability, we can mean the distances and take argmin.
        # Or convert to soft probs: exp(-dist) / sum(exp(-dist)).
        # Simple Mean Distances is robust for MDM.
        
        df = pd.DataFrame({
            "trial_id": all_trial_ids,
            "true_label": all_labels,
            "d0": all_probs[:, 0],
            "d1": all_probs[:, 1],
            "d2": all_probs[:, 2]
        })
        
        agg = df.groupby("trial_id").agg({
            "true_label": "first",
            "d0": "mean",
            "d1": "mean",
            "d2": "mean"
        }).reset_index()
        
        agg["pred_label"] = agg[["d0", "d1", "d2"]].idxmin(axis=1).apply(lambda x: int(x[1])) # d0->0
        
        trial_acc = (agg["pred_label"] == agg["true_label"]).mean()
        
        metrics = {
            "win_acc": win_acc,
            "trial_agg_acc": trial_acc,
            "raw_pred_count": len(all_preds),
            "trial_count": len(agg)
        }
        
        return metrics, df, agg

