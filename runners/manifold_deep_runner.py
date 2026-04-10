import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import traceback
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from models.spdnet import DeepSPDClassifier
from runners.spatial_dcnet_torch import DCNetTorch

class TrialDataset(torch.utils.data.Dataset):


    def __init__(self, trials, labels, trial_ids=None, band_idx=4, window_len=None, stride=None, return_5band=False, bands_mode="single", band_norm_mode="none"):
        self.band_idx = band_idx 
        self.window_len = window_len
        self.stride = stride
        self.return_5band = return_5band
        self.bands_mode = bands_mode
        self.band_norm_mode = band_norm_mode

        self.samples = [] 
        self.labels = []
        self.trial_indices = []
        self.trial_ids_map = trial_ids if trial_ids is not None else [str(i) for i in range(len(trials))]

        if window_len is None or stride is None:
            # Original "trial" mode
            self.mode = "trial"
            self.samples = trials 
            self.labels = labels
            self.trial_indices = list(range(len(trials)))
        else:
            # Window mode
            self.mode = "window"
            for i, trial in enumerate(trials):
                # trial: (T, 310)
                T_total = trial.shape[0]
                label = labels[i]
                
                # Reshape to (T, 5, 62) -> (T, 62, 5) -> (T, Channel, Band)
                # Wait, original logic: reshape(T, 5, 62).transpose(0, 2, 1) to get (T, 62, 5)
                # Explanation: 310 is B0C0..B0C61..B1C0.
                # reshape(T, 5, 62) separates bands.
                # transpose(0, 2, 1) puts channels before bands? No.
                # trial.reshape(T, 5, 62) -> dim1=Band, dim2=Channel.
                # transpose(0, 2, 1) -> (T, 62, 5). dim1=Channel, dim2=Band.
                t_reshaped = trial.reshape(T_total, 5, 62).transpose(0, 2, 1)

                for start in range(0, T_total - self.window_len + 1, self.stride):
                   end = start + self.window_len
                   if self.return_5band:
                       window_data = t_reshaped[start:end, :, :] # (Win, 62, 5)
                       self.samples.append(window_data)
                   elif self.bands_mode == "all5_timecat":
                       # (Win, 62, 5)
                       w_data = t_reshaped[start:end, :, :] 
                       self.samples.append(w_data) 
                   else:
                       x_band = t_reshaped[start:end, :, self.band_idx] # (Win, 62)
                       # Transpose to (62, Win) for CovPool
                       self.samples.append(x_band.transpose(1, 0))
                   
                   self.labels.append(label)
                   self.trial_indices.append(i)

    def _apply_norm(self, x):
        # x: (Win, 62, 5) or (T, 62, 5)
        # return: normalized x
        if self.band_norm_mode == "none":
            return x
        
        # Permute to (Band, Channel, Time) for easier broadcasting?
        # x is (Time, Channel, Band)
        # Permute to (Band, Channel, Time) = (2, 1, 0)
        x_p = x.transpose(2, 1, 0) # (5, 62, Time)
        
        if self.band_norm_mode == "per_band_channel_z":
            # Mean/Std over Time (dim 2)
            mean = x_p.mean(dim=2, keepdim=True)
            std = x_p.std(dim=2, keepdim=True) + 1e-6
            x_norm = (x_p - mean) / std
        elif self.band_norm_mode == "per_band_global_z":
            # Mean/Std over Channel AND Time (dim 1, 2)
            mean = x_p.mean(dim=(1, 2), keepdim=True)
            std = x_p.std(dim=(1, 2), keepdim=True) + 1e-6
            x_norm = (x_p - mean) / std
        else:
            return x
            
        # Permute back to (Time, Channel, Band) = (2, 1, 0)
        return x_norm.transpose(2, 1, 0)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        if self.mode == "trial":
            # trial: (T, 310)
            t = self.samples[idx]
            # Fixed Reshape Logic (Input 310 is Band-Major: B0C0..B0C61..B1C0)
            # reshape(T, 5, 62) -> (T, Band, Channel)
            # transpose(0, 2, 1) -> (T, Channel, Band)
            t_reshaped = t.reshape(t.shape[0], 5, 62).transpose(0, 2, 1)
            
            if self.return_5band:
                x = t_reshaped
                # Apply Norm
                x = torch.tensor(x, dtype=torch.float64)
                x = self._apply_norm(x).numpy()
            elif self.bands_mode == "all5_timecat":
                 # (T, 62, 5)
                 x_temp = torch.tensor(t_reshaped, dtype=torch.float64)
                 x_temp = self._apply_norm(x_temp) # (T, 62, 5)
                 
                 # New logic: Return (5, 62, T) for Runner to decide strategy
                 # Permute (T, 62, 5) -> (5, 62, T)
                 x = x_temp.permute(2, 1, 0).numpy()
            else:
                x_band = t_reshaped[:, :, self.band_idx]
                x = x_band.transpose(1, 0) # (62, T)
            
            y = self.labels[idx]
            trial_idx = self.trial_indices[idx]
            return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long), torch.tensor(trial_idx, dtype=torch.long)
        else:
            # Window mode
            x_in = self.samples[idx] 
            
            if self.return_5band:
                 x_t = torch.tensor(x_in, dtype=torch.float64)
                 x = self._apply_norm(x_t).numpy()
            elif self.bands_mode == "all5_timecat":
                 # Stored as (Win, 62, 5)
                 x_t = torch.tensor(x_in, dtype=torch.float64)
                 x_t = self._apply_norm(x_t) # Band Norm First
                 # New Logic: Return (5, 62, Win)
                 # Stores as (Win, 62, 5), permute to (5, 62, Win)
                 x = x_t.permute(2, 1, 0).numpy()
            else:
                 x = x_in # Already configured
            # else: x is already (62, Win) from init
            
            y = self.labels[idx]
            trial_idx = self.trial_indices[idx]
            return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long), torch.tensor(trial_idx, dtype=torch.long)

class ManifoldDeepRunner:
    def __init__(self, args, num_classes=3):
        self.args = args
        self.num_classes = num_classes
        dev = args.torch_device if args.torch_device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.epochs = getattr(args, 'epochs', 50) # Default 50, but allow override
        self.batch_size = getattr(args, 'batch_size', 32) # Default 32, but allow override
        self.lr = getattr(args, 'lr', 1e-3)
        self.matrix_mode = getattr(args, 'matrix_mode', 'cov') # 'cov' or 'corr'
        self.subject_centering = getattr(args, 'subject_centering', False) # Phase 13E Step 3
        self.global_centering = getattr(args, 'global_centering', False) # Phase 13E Step 4
        
        # MVP1 Args
        self.mvp1_guided_cov = getattr(args, 'mvp1_guided_cov', False)
        self.mvp1_attn_method = getattr(args, 'mvp1_attn_method', 'grad')
        self.mvp1_attn_power = getattr(args, 'mvp1_attn_power', 0.5) # Default 0.5 (sqrt)
        self.mvp1_attn_power = getattr(args, 'mvp1_attn_power', 0.5) # Default 0.5 (sqrt)
        self.bands_mode = getattr(args, 'bands_mode', 'single')
        self.band_norm_mode = getattr(args, 'band_norm_mode', 'none')
        self.spd_eps = getattr(args, 'spd_eps', 1e-4) # Pass to model
        self.dcnet_ckpt = getattr(args, 'dcnet_ckpt', None)
        self.dcnet_model = None
        self.metrics_csv = getattr(args, 'metrics_csv', None)
        
        # BandGate
        self.use_band_gate = getattr(args, 'use_band_gate', False)
        self.band_activation_mode = getattr(args, 'band_activation_mode', 'original')
        self.band_gate_model = None
        self.cov_pool_weighted = None
        
        if self.use_band_gate:
            from models.band_activation import BandScalarGateV1
            from models.spdnet import CovPoolWeightedBandsV1
            print("Initializing BandScalarGateV1...")
            self.band_gate_model = BandScalarGateV1(n_bands=5).to(self.device).double()
            self.cov_pool_weighted = CovPoolWeightedBandsV1(eps=self.spd_eps).to(self.device).double()
            
        # ROI Pooling
        self.use_roi_pooling = getattr(args, 'use_roi_pooling', False)
        self.roi_pool = None
        
        if self.use_roi_pooling:
            from models.roi_pooling import ROIPooling
            print("Initializing ROIPooling (K=13)...")
            self.roi_pool = ROIPooling(mode="mean", strict=True).to(self.device).double()
            
        # Stats container
        self.gate_stats = []

    def _load_dcnet(self):
        if not self.dcnet_ckpt:
            raise ValueError("DCNet guided covariance requested but no 'dcnet_ckpt' provided.")
        print(f"Loading DCNet from {self.dcnet_ckpt}...")
        # Assuming input_dim=310 (Refactored)
        model = DCNetTorch(input_dim=310, num_classes=3, classifier_type="conv")
        state = torch.load(self.dcnet_ckpt, map_location=self.device)
        # Check if state has 'model_state' key or is raw state_dict
        if 'model_state' in state:
            model.load_state_dict(state['model_state'])
        else:
            model.load_state_dict(state)
        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.dcnet_model = model
        print("DCNet loaded and frozen.")

    def _compute_saliency(self, x_win_5band):
        """
        x_win_5band: (B, Win, 62, 5)
        Returns: A62 (B, 62) normalized saliency
        """
        # Snapshot: Last time step
        x_snap = x_win_5band[:, -1, :, :] # (B, 62, 5)
        B = x_snap.size(0)
        
        # Flatten -> (B, 310, 1, 1)
        x_snap_flat = x_snap.reshape(B, 310, 1, 1).detach() # Detach to make it a leaf for saliency graph
        
        if self.mvp1_attn_method == 'grad':
            # Ensure gradient tracking is ON even if model parameters are frozen
            with torch.enable_grad():
                x_snap_flat.requires_grad_(True)
                logits = self.dcnet_model(x_snap_flat)
                y_hat = logits.argmax(dim=1)
                score = logits.gather(1, y_hat[:, None]).sum()
                
                # Create graph=True NOT needed for saliency itself, usually.
                # But we need grad w.r.t input.
                grad = torch.autograd.grad(score, x_snap_flat, create_graph=False)[0] # (B, 310, 1, 1)
            saliency_310 = grad.abs().view(B, 310)
            
            # Reshape back to (B, 62, 5) and mean over bands
            A62 = saliency_310.view(B, 62, 5).mean(dim=2) # (B, 62)
            
        elif self.mvp1_attn_method == 'input_energy':
            # Fallback
            A62 = x_snap.abs().mean(dim=2) # (B, 62)
        else:
            raise ValueError(f"Unknown attn method: {self.mvp1_attn_method}")
        
        # Normalize (Mean=1)
        A62 = torch.nn.functional.softplus(A62)
        mean_val = A62.mean(dim=1, keepdim=True) + 1e-8
        A62 = A62 / mean_val
        A62 = A62.clamp(0.2, 5.0)
        
        return A62.detach()

    def _compute_effect_size(self, x_orig, x_weighted):
        """
        Compute Frobenius delta between Covariances of x_orig and x_weighted.
        x: (B, 62, T)
        """
        # Center data? CovPool usually assumes centered or centers it. 
        # DeepSPDClassifier CovPool typically does: x = x - x.mean(dim=2, keepdim=True)
        # We'll mimic raw correlation matrix diff slightly simplified.
        
        # B, C, T
        B, C, T = x_orig.shape
        if T > 1:
            # Baseline Cov
            x0 = x_orig - x_orig.mean(dim=2, keepdim=True)
            cov0 = torch.matmul(x0, x0.transpose(1, 2)) / (T - 1)
            
            # Weighted Cov
            xw = x_weighted - x_weighted.mean(dim=2, keepdim=True)
            covw = torch.matmul(xw, xw.transpose(1, 2)) / (T - 1)
            
            # 1. Delta Cov Frobenius
            # ||Cw - C0||_F / ||C0||_F
            diff = covw - cov0
            norm_diff = torch.norm(diff, p='fro', dim=(1,2))
            norm_0 = torch.norm(cov0, p='fro', dim=(1,2)) + 1e-8
            delta_cov_fro = (norm_diff / norm_0).mean().item()
            
            # 2. Delta Eig L1
            # Eigvals
            eig0 = torch.linalg.eigvalsh(cov0) # (B, 62)
            eigw = torch.linalg.eigvalsh(covw)
            
            abs_diff = (eigw - eig0).abs()
            mean_eig0 = eig0.abs().mean(dim=1) + 1e-8
            
            # Relative L1 per sample
            delta_eig_l1 = (abs_diff.mean(dim=1) / mean_eig0).mean().item()
            
            return delta_cov_fro, delta_eig_l1
        else:
            return 0.0, 0.0
            
    def _compute_spd_stats(self, loader, n_samples=256):
        """
        Compute eigenvalues, condition numbers, and effective rank stats for the first N samples.
        Also check band energy balance if available.
        """
        pre_eigs = []
        post_eigs = []
        pre_conds = []
        post_conds = []
        pre_eff_ranks = []
        post_eff_ranks = []
        
        band_rms_list = [] # (5,) per sample
        
        count = 0
        with torch.no_grad():
            for Xb, _, _ in loader:
                Xb = Xb.to(self.device).double()
                B = Xb.size(0)
                
                # Check Band RMS BEFORE any processing if possible
                # If Xb is (B, 62, Win*5) and bands_mode==all5_timecat
                # Or if guided (B, Win, 62, 5)
                # Note: If dataset applied norm, Xb is already normalized.
                # Band RMS check on normalized data confirms normalization.
                
                if self.bands_mode == "all5_timecat" and Xb.dim() == 3:
                    # Xb: (B, 62, Win*5)
                    # We need to unstack to check bands
                    Win5 = Xb.size(2)
                    Win = Win5 // 5
                    # Reshape to (B, 62, Win, 5) ? No
                    # Dataset: Permute (62, Win, 5) -> Reshape (62, Win*5)
                    # Reverse: Xb.view(B, 62, Win, 5) ?? No.
                    # Reshape (62, -1) flattens last dim.
                    # So (62, Win, 5) -> (62, Win*5)
                    # view(B, 62, Win, 5) should work if C-ordered
                    x_bands = Xb.view(B, 62, Win, 5)
                    # RMS per band: sqrt(mean(x^2)) over 62, Win
                    rms = x_bands.pow(2).mean(dim=(1, 2)).sqrt() # (B, 5)
                    band_rms_list.append(rms.cpu().numpy())
                elif self.mvp1_guided_cov and Xb.dim() == 4:
                    # (B, Win, 62, 5)
                    # RMS over Win, 62
                    rms = Xb.pow(2).mean(dim=(1, 2)).sqrt() # (B, 5)
                    band_rms_list.append(rms.cpu().numpy())

                # Get Covariance input
                if self.bands_mode == "all5_timecat":
                     # Xb is (B, Win, 62, 5)
                     Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()
                     if self.band_norm_mode == "manual_per_band_time":
                        mean = Xb_perm.mean(dim=3, keepdim=True)
                        std = Xb_perm.std(dim=3, keepdim=True) + 1e-6
                        Xb_perm = (Xb_perm - mean) / std
                     elif self.band_norm_mode == "per_band_global_z":
                        mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                        std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                        Xb_perm = (Xb_perm - mean) / std

                     if self.use_band_gate:
                         w, _ = self.band_gate_model(Xb_perm)
                         if self.band_activation_mode == "input_gate_v1":
                             w_sqrt = torch.sqrt(w + 1e-12).view(w.size(0), 5, 1, 1)
                             X_gated = Xb_perm * w_sqrt
                             model_in = X_gated.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                         else:
                             # Weighted Cov mode - Handle specially or approximate?
                             # For stats, let's just use flattened concatenation approx or error?
                             # Actually we can compute cov directly here and set a flag to skip standard computation.
                             # But simpler to just support input_gate and no-gate for now (Step 1.5).
                             # If we are here in Step 1.5, use_band_gate is False.
                             # If we were in Step 1, input_gate_v1 was used.
                             # So this should cover our immediate needs.
                             # Fallback for old weighted cov:
                             model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                     else:
                         model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                     
                     model_in = model_in.double()

                elif self.mvp1_guided_cov:
                    A62 = self._compute_saliency(Xb.float())
                    x_gamma = Xb[:, :, :, 4]
                    weights = A62.unsqueeze(1).pow(self.mvp1_attn_power).double()
                    x_w = x_gamma * weights
                    model_in = x_w.permute(0, 2, 1).double() # (B, 62, Win)
                else:
                    model_in = Xb # (B, 62, T)
                
                # Compute Cov (Pre-Eps)
                mean = model_in.mean(dim=2, keepdim=True)
                X_c = model_in - mean
                T = model_in.size(2)
                cov = torch.matmul(X_c, X_c.transpose(1, 2)) / (T - 1) # (B, 62, 62)
                
                # Stats Pre-Eps
                try:
                    eigs0 = torch.linalg.eigvalsh(cov)
                    mins0 = eigs0[:, 0]
                    maxs0 = eigs0[:, -1]
                    # cond might be inf
                    conds0 = maxs0 / (mins0.abs() + 1e-12)
                    eff_ranks0 = (eigs0 > 1e-6).sum(dim=1).double()
                    
                    pre_eigs.append(eigs0.cpu().numpy())
                    pre_conds.append(conds0.cpu().numpy())
                    pre_eff_ranks.append(eff_ranks0.cpu().numpy())
                except:
                    pass

                # Add eps (Post-Eps)
                cov = cov + torch.eye(cov.size(1), device=self.device).double() * self.spd_eps
                eigs = torch.linalg.eigvalsh(cov)
                
                mins = eigs[:, 0]
                maxs = eigs[:, -1]
                conds = maxs / (mins + 1e-12)
                eff_ranks = (eigs > 1e-6).sum(dim=1).double()
                
                post_eigs.append(eigs.cpu().numpy())
                post_conds.append(conds.cpu().numpy())
                post_eff_ranks.append(eff_ranks.cpu().numpy())
                
                count += B
                if count >= n_samples:
                    break
        
        stats = {}
        
        # Helper stats
        def box_stats(arr, prefix):
            flat = np.concatenate(arr, axis=0)[:n_samples]
            if flat.ndim == 2: # eigs
                vals_min = flat[:, 0]
                return {
                    f"{prefix}_min_eig_p05": float(np.quantile(vals_min, 0.05)),
                    f"{prefix}_min_eig_median": float(np.median(vals_min)),
                }
            else: # cond/rank
                return {
                    f"{prefix}_cond_p95": float(np.quantile(flat, 0.95)),
                    f"{prefix}_eff_rank_p50": float(np.median(flat))
                }
                
        if pre_eigs:
            stats.update(box_stats(pre_eigs, "pre"))
            stats.update(box_stats(pre_conds, "pre"))
            stats.update(box_stats(pre_eff_ranks, "pre"))
            
        if post_eigs:
            stats.update(box_stats(post_eigs, "post"))
            stats.update(box_stats(post_conds, "post"))
            stats.update(box_stats(post_eff_ranks, "post"))
            
        if band_rms_list:
            # Ratio of average RMS per band to sum
            all_rms = np.concatenate(band_rms_list, axis=0)[:n_samples] # (N, 5)
            mean_rms = np.mean(all_rms, axis=0) # (5,)
            total = np.sum(mean_rms) + 1e-9
            ratio = mean_rms / total
            stats['band_rms_ratio'] = ratio.tolist() # [b0, b1, b2, b3, b4]
            
        return stats

    def _log_spectrum_diagnostics(self, x, fold_name):
        """
        Compute spectrum diagnostics for Phase 13D.
        x: (B, 62, T)
        """
        try:
            # 1. Compute Matrix (Cov or Corr)
            B, C, T = x.shape
            x_c = x - x.mean(dim=2, keepdim=True)
            
            if self.matrix_mode == 'corr':
                 x_std = x_c.std(dim=2, keepdim=True) + 1e-6
                 x_z = x_c / x_std
                 mat = torch.matmul(x_z, x_z.transpose(1, 2)) / (T - 1)
                 mat = 0.5 * (mat + mat.transpose(1, 2))
            else:
                 mat = torch.matmul(x_c, x_c.transpose(1, 2)) / (T - 1) # (B, C, C)
                 mat = 0.5 * (mat + mat.transpose(1, 2))
            
            # 2. Stats (Pre-Eps)
            # Diagonals
            diags = torch.diagonal(mat, dim1=1, dim2=2) # (B, C)
            diag_mean = diags.mean().item()
            diag_std = diags.std().item()
            
            # Off-Diagonals
            # Mask diagonal
            mask = ~torch.eye(C, device=self.device, dtype=bool)
            off_diags = mat[:, mask] # (B, C*C-C)
            off_diag_abs_mean = off_diags.abs().mean().item()
            
            # Eigen
            eigs = torch.linalg.eigvalsh(mat)
            mask_low = (eigs <= 10 * self.spd_eps)
            count_low = mask_low.sum(dim=1).float().mean().item()
            
            # Pre-Eps Cond
            mins = eigs[:, 0].clamp(min=1e-12)
            maxs = eigs[:, -1]
            pre_cond_p95 = np.percentile((maxs/mins).detach().cpu().numpy(), 95)
            pre_min_eig_p05 = np.percentile(mins.detach().cpu().numpy(), 5)
            
            # 3. Post-Eps Stats
            mat_eps = mat + torch.eye(C, device=self.device).double() * self.spd_eps
            eigs_eps = torch.linalg.eigvalsh(mat_eps)
            
            # Eff Rank
            vals_sum = eigs_eps.sum(dim=1, keepdim=True)
            p = eigs_eps / vals_sum
            entropy = -(p * torch.log(p)).sum(dim=1)
            eff_rank = torch.exp(entropy).mean().item()
            
            # Post-Eps Cond
            mins_eps = eigs_eps[:, 0]
            maxs_eps = eigs_eps[:, -1]
            post_cond_p95 = np.percentile((maxs_eps/mins_eps).detach().cpu().numpy(), 95)
            post_min_eig_p05 = np.percentile(mins_eps.detach().cpu().numpy(), 5)
            
            # Eps Dominance
            eps_norm = self.spd_eps * np.sqrt(C)
            c_norm = torch.norm(mat, dim=(1,2)).mean().item()
            eps_dom = eps_norm / (c_norm + 1e-9)
            
            diag = {
                "phase": "13e_step1",
                "matrix_mode": self.matrix_mode,
                "cov_dim": C,
                "T_eff": T,
                "eff_rank": eff_rank,
                
                "pre_eps_diag_mean": diag_mean,
                "pre_eps_diag_std": diag_std,
                "pre_eps_offdiag_abs_mean": off_diag_abs_mean,
                "pre_eps_min_eig_p05": pre_min_eig_p05,
                "pre_eps_cond_p95": pre_cond_p95,
                
                "post_eps_min_eig_p05": post_min_eig_p05,
                "post_eps_cond_p95": post_cond_p95,
                
                "eigs_le_10eps_count": count_low,
                "eps_dominance": eps_dom,
                "timestamp": str(pd.Timestamp.now())
            }
            
            # Save
            path = f"promoted_results/{fold_name}_diagnostics.json"
            
            # Safe Update
            existing = {}
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        existing = json.load(f)
                except:
                    pass
            
            existing.update(diag) # Merge new spectrum stats
            
            # Ensure dir
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"[Diagnostics] Saved spectrum stats to {path}")
            
        except Exception as e:
            print(f"[Diagnostics] Failed: {e}")
                


    def _compute_matrix_from_input(self, x):
        """
        Compute Covariance or Correlation matrix from input (B, C, T).
        Returns SPD-regularized matrix (pre-centering).
        """
        B, C, T = x.shape
        x_c = x - x.mean(dim=2, keepdim=True)
        
        if self.matrix_mode == 'corr':
             x_std = x_c.std(dim=2, keepdim=True) + 1e-6
             x_z = x_c / x_std
             mat = torch.matmul(x_z, x_z.transpose(1, 2)) / (T - 1)
             mat = 0.5 * (mat + mat.transpose(1, 2))
        else:
             mat = torch.matmul(x_c, x_c.transpose(1, 2)) / (T - 1)
             mat = 0.5 * (mat + mat.transpose(1, 2))
        
        # SPD-ize (Base regularization)
        # Note: Centering might need 'pure' matrix or regularized one. 
        # User says: "Regularize to SPD: C_spd = sym(C) + eps * I ... Compute Mean"
        mat = mat + torch.eye(C, device=self.device).double() * self.spd_eps
        return mat

    def _compute_log_euclidean_mean(self, loader, fold_name):
        """
        Compute per-subject Log-Euclidean Mean from Training Loader.
        Returns: {subj_id (str): mean_matrix (Tensor)}
        """
        print("[Centering] Computing Subject Means...")
        subj_sum = {}
        subj_count = {}
        
        # Disable tracking
        with torch.no_grad():
            for Xb, _, tib in loader:
                Xb = Xb.to(self.device).float() # Assuming float is enough for mean, or double? 
                # Input usually double in pipeline? 
                # fit_predict converts to double if needed. Let's stick to model precision.
                # Actually pipeline uses float/double mixed. model is double.
                Xb = Xb.double()
                
                # Check formatting (all5 etc) to get (B, C, T)
                # This logic duplicates fit_predict prep. 
                # Ideally we should refactor input prep.
                # For now copy essential prep logic:
                if self.bands_mode == "all5_timecat":
                     Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()
                     if self.band_norm_mode == "per_band_global_z":
                         mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                         std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                         Xb_perm = (Xb_perm - mean) / std
                     
                     # Flatten to (B, 62, T)
                     # Assuming no gating for this step (Manifold Only)
                     model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                else:
                     model_in = Xb # Single band (B, 62, T)
                     
                mats = self._compute_matrix_from_input(model_in)
                
                # Iterate batch
                ids_map = loader.dataset.trial_ids_map
                for i in range(len(mats)):
                    tid = tib[i].item()
                    sid = ids_map[tid].split('_')[0]
                    
                    # Log map
                    eps, vec = torch.linalg.eigh(mats[i])
                    # Clamp for log stability
                    eps = eps.clamp(min=1e-8)
                    log_eps = torch.log(eps)
                    log_mat = vec @ torch.diag(log_eps) @ vec.t()
                    
                    if sid not in subj_sum:
                        subj_sum[sid] = torch.zeros_like(log_mat)
                        subj_count[sid] = 0
                    
                    subj_sum[sid] += log_mat
                    subj_count[sid] += 1
        
        # Exp map
        means = {}
        for sid in subj_sum:
            avg_log = subj_sum[sid] / subj_count[sid]
            eps, vec = torch.linalg.eigh(avg_log)
            exp_eps = torch.exp(eps)
            avg_mat = vec @ torch.diag(exp_eps) @ vec.t()
            means[sid] = avg_mat
            
        # Save cache
        cache_path = f"promoted_results/{os.path.dirname(fold_name)}/centered_corr/subject_mean_corr.pt"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(means, cache_path)
        print(f"[Centering] Means computed for {len(means)} subjects. Saved to {cache_path}")
        
        # Save Stats to new JSON
        diag_path = f"promoted_results/{os.path.dirname(fold_name)}/centered_corr/report_centering_diagnostics.json"
        
        # Diagnostic Stats
        stats = {
            "train_subject_count": len(means),
            "subjects": list(means.keys()),
            "counts": subj_count
        }
        with open(diag_path, "w") as f:
            json.dump(stats, f, indent=2)
            
        return means, diag_path

    def _apply_centering(self, mats, tib, dataset):
        """
        Apply Riemannian Centering to batch of matrices.
        mats: (B, C, C) SPD
        tib: (B,) trial indices
        dataset: TrialDataset with trial_ids_map
        Returns: Centered Mats (B, C, C)
        """
        centered_mats = []
        ids_map = dataset.trial_ids_map
        
        # Identity Check accumulator (first batch only)
        # Should be handled by diagnostics logging, but can print debug here if needed.
        
        for i in range(len(mats)):
            tid = tib[i].item()
            sid = ids_map[tid].split('_')[0]
            
            if sid not in self.subject_means:
                 # Fallback? Should fail for strictness.
                 # User said: "Hard fail if any subject_id appears in TEST but not in subject_mean_corr.pt"
                 # Raise error
                 raise ValueError(f"Subject {sid} not found in computed means (unseen subject).")
                 
            Cbar = self.subject_means[sid].to(self.device).double()
            C = mats[i].double()
            
            # Whitening Matrix W = Cbar^{-1/2}
            # Cbar is SPD. 
            evals, evecs = torch.linalg.eigh(Cbar)
            evals = evals.clamp(min=1e-8)
            W = evecs @ torch.diag(evals.rsqrt()) @ evecs.t()
            
            # Center: W * C * W^T
            C_center = W @ C @ W.t()
            
            # Symmetrize numerical errors
            C_center = 0.5 * (C_center + C_center.t())
            
            centered_mats.append(C_center)
            
        return torch.stack(centered_mats)

    # Phase 13E Step 4: Global Centering (Log-Euclidean)
    def _compute_global_log_euclidean_mean(self, loader, fold_name):
        print("[Global Centering] Computing Global Template (Train Only)...")
        all_logs = []
        count = 0
        try:
             for Xb, _, _ in loader:
                 Xb = Xb.to(self.device).double()
                 
                 # Handling for 4D input (bands)
                 if Xb.ndim == 4: # (B, Win, 62, 5)
                      Xb_perm = Xb.permute(0, 3, 2, 1).contiguous() # (B, 5, 62, Win)
                      if self.band_norm_mode == "per_band_global_z":
                           mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                           std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                           Xb_perm = (Xb_perm - mean) / std
                      # To (B, 62, 5*Win)
                      model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                 else:
                      model_in = Xb
                 
                 # Compute matrix from properly shaped input
                 mat = self._compute_matrix_from_input(model_in) # (B, C, C) SPD

                 
                 # Log-Euclidean: logm(C)
                 # Using eigh
                 L, V = torch.linalg.eigh(mat)
                 # Clamp
                 L = torch.clamp(L, min=1e-8)
                 log_L = torch.diag_embed(torch.log(L))
                 log_mat = V @ log_L @ V.transpose(1, 2)
                 
                 all_logs.append(log_mat.cpu()) # Store on CPU to avoid OOM
                 count += Xb.size(0)
                 
             # Compute Mean
             all_logs_cat = torch.cat(all_logs, dim=0) # (N, C, C)
             mean_log = all_logs_cat.mean(dim=0).to(self.device) # (C, C)
             
             # Expm for C_ref (though user formula subtracts log_ref, so maybe keep both)
             # C_ref = expm(mean_log)
             L_m, V_m = torch.linalg.eigh(mean_log)
             # V_m is (C, C). diagonal is (C, C). V_m.t() is (C, C).
             C_ref = V_m @ torch.diag(torch.exp(L_m)) @ V_m.t()
             
             # Save
             save_dir = f"promoted_results/{fold_name.split('/report')[0]}/global_centered_corr"
             os.makedirs(save_dir, exist_ok=True)
             
             # Save standard npy format for transparency
             np.save(f"{save_dir}/global_ref_spd.npy", C_ref.cpu().numpy())
             
             meta = {
                 "count": count,
                 "mean_type": "logeuclid",
                 "source": "train_only",
                 "matrix_mode": self.matrix_mode
             }
             with open(f"{save_dir}/global_ref_meta.json", "w") as f:
                 json.dump(meta, f, indent=2)
                 
             print(f"[Global Centering] Computed Global Mean from {count} samples.")
             return mean_log # Return the log(C_ref) since we subtract logs
             
        except Exception as e:
             traceback.print_exc()
             print(f"Error computing global mean: {e}")
             sys.exit(1)

    def _apply_global_centering(self, mats, log_ref):
        # mats: (B, C, C) - Already SPD-regularized
        # log_ref: (C, C) - The Mean in Log Domain
        
        # 1. logm(C)
        L, V = torch.linalg.eigh(mats)
        L = torch.clamp(L, min=1e-8)
        log_L = torch.diag_embed(torch.log(L))
        log_mats = V @ log_L @ V.transpose(1, 2)
        
        # 2. Subtract: log(C_center) = log(C) - log(C_ref)
        if log_ref.ndim == 2:
             log_ref = log_ref.unsqueeze(0)
        
        log_centered = log_mats - log_ref
        
        # 3. expm
        L_c, V_c = torch.linalg.eigh(log_centered)
        # Note: log_centered is symmetric, so eigenvalues are real.
        C_centered = V_c @ torch.diag_embed(torch.exp(L_c)) @ V_c.transpose(1, 2)
        
        # 4. Symmetrize
        C_centered = 0.5 * (C_centered + C_centered.transpose(1, 2))
        
        return C_centered
    
    def fit_predict(self, fold, fold_name="fold"):
        # fold is TrialFoldData

        
        # Use simple list reference - don't flatten!
        if hasattr(fold, 'trials_train'):
             X_tr_list = fold.trials_train
             y_tr_list = fold.y_trial_train
             X_te_list = fold.trials_test
             y_te_list = fold.y_trial_test
        else:
             raise ValueError("ManifoldDeepRunner requires TrialFoldData with 'trials_train' (list of trials).")
        
        # Datasets (Float64 for Riemannian stability)
        # Fix Issue 1.1: Use TrialDataset to pick band and preserve Time dim
        # Using windowing for training and testing
        
        # Init DCNet if guided
        if self.mvp1_guided_cov:
            self._load_dcnet()
        
        # Splitting Logic (80/20 Train/Val) - Deterministic per seed
        import random
        # Get seed from args if strict
        seed_val = getattr(self.args, 'seed', 0)
        
        # Indices
        n_train_total = len(X_tr_list)
        indices = list(range(n_train_total))
        
        # Shuffle with local random
        r = random.Random(seed_val)
        r.shuffle(indices)
        
        split_idx = int(n_train_total * 0.8)
        tr_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Log Split Stats
        split_stats = {
            "n_train": len(tr_indices),
            "n_val": len(val_indices),
            "n_test": len(X_te_list),
            "tr_ids_preview": tr_indices[:5],
            "val_ids_preview": val_indices[:5],
            "train_val_intersection": len(set(tr_indices) & set(val_indices)),
            # Label Hist (Sanitize for JSON)
            "train_label_hist": {int(k): int(v) for k, v in pd.Series([y_tr_list[i] for i in tr_indices]).value_counts().sort_index().items()},
            "val_label_hist": {int(k): int(v) for k, v in pd.Series([y_tr_list[i] for i in val_indices]).value_counts().sort_index().items()}
        }
        
        # Save to diagnostics
        diag_path = f"promoted_results/{fold_name}_diagnostics.json"
        
        # Function to safe update
        def update_json(path, new_data):
            data = {}
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                except:
                    pass
            data.update(new_data)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        update_json(diag_path, {"split_stats": split_stats})
        print(f"[Split] Stats saved to {diag_path}")
        
        X_train_sub = [X_tr_list[i] for i in tr_indices]
        y_train_sub = [y_tr_list[i] for i in tr_indices]
        X_val_sub = [X_tr_list[i] for i in val_indices]
        y_val_sub = [y_tr_list[i] for i in val_indices]
        
        print(f"[{fold_name}] Split Train({n_train_total}) -> TrainSub({len(tr_indices)}) + ValSub({len(val_indices)})")
        
        # Datasets
        window_len = 24
        stride = 12
        # return_5band should be true if guided OR if we need bands for timecat/gating
        return_5band = self.mvp1_guided_cov or (self.bands_mode == "all5_timecat")
        bands_mode = self.bands_mode
        band_norm_mode = self.band_norm_mode
        
        # Phase 13D: Manual Norm handling
        # If manual mode selected in runner, tell dataset to do 'none'
        if band_norm_mode in ["manual_per_band_time", "per_band_global_z"]:
            dataset_norm_mode = "none"
        else:
            dataset_norm_mode = band_norm_mode
            
        # Step 3: Split Audit (Phase 11)
        # Use explicit Trial IDs
        tr_ids = []
        val_ids = []
        te_ids = []
        
        if hasattr(fold, 'trial_id_train') and fold.trial_id_train is not None:
             # Assume numpy array or list
             all_train_ids = fold.trial_id_train
             tr_ids = [all_train_ids[i] for i in tr_indices]
             val_ids = [all_train_ids[i] for i in val_indices]
        
        if hasattr(fold, 'trial_id_test') and fold.trial_id_test is not None:
             te_ids = list(fold.trial_id_test)
        
        train_dset = TrialDataset(X_train_sub, y_train_sub, trial_ids=tr_ids, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=bands_mode, band_norm_mode=dataset_norm_mode)
        val_dset = TrialDataset(X_val_sub, y_val_sub, trial_ids=val_ids, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=bands_mode, band_norm_mode=dataset_norm_mode)
        test_dset = TrialDataset(X_te_list, y_te_list, trial_ids=te_ids, band_idx=4, window_len=window_len, stride=stride, return_5band=return_5band, bands_mode=bands_mode, band_norm_mode=dataset_norm_mode)
        
        batch_size = getattr(self.args, 'batch_size', 32)
        batch_size_val = batch_size
        
        test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dset, batch_size=batch_size_val, shuffle=False)
        
        # Phase 13E Step 3: Compute Subject Means if Centering
        self.subject_means = {}
        if self.subject_centering:
            self.subject_means, diag_path = self._compute_log_euclidean_mean(train_loader, fold_name)
            
        # Phase 13E Step 4: Compute Global Mean if Centering
        self.log_ref_global = None
        if self.global_centering:
            self.log_ref_global = self._compute_global_log_euclidean_mean(train_loader, fold_name)

        
        # Old audit block removed (moved up)
             
        # Check Intersection (Train vs Test)
        # Convert to set for O(1)
        tr_id_set = set(tr_ids)
        te_id_set = set(te_ids)
        intersect = tr_id_set.intersection(te_id_set)
        
        audit_res = {
            "audit_key_definition": "subject_session_trial (e.g., '1_s1_t0')",
            "train_keys_count": len(tr_id_set),
            "test_keys_count": len(te_id_set),
            "intersection_count": len(intersect),
            "intersection_sample": list(intersect)[:5],
            "val_keys_count": len(set(val_ids)) # Info only
        }
        
        with open(f"promoted_results/{fold_name}_split.json", "w") as f:
            json.dump(audit_res, f, indent=2)
            
        if len(intersect) > 0:
            print(f"CRITICAL AUDIT FAILURE: Found {len(intersect)} overlapping IDs between Train and Test!")
            
        # Status
        with open(f"promoted_results/{fold_name}_status.txt", "w") as f:
            f.write(f"Status: TRAINING\nStart Time: {pd.Timestamp.now()}\n")
        
        # Model
        # Phase 7: Optimized Config
        cov_alpha = getattr(self.args, 'cov_alpha', 0.01)
        hidden_dim = getattr(self.args, 'hidden_dim', 96)
        
        # input_dim for ROI vs 62
        input_dim = 62
        pool_type = 'conv'
        
        if self.use_roi_pooling:
            input_dim = 13 # K = 13 ROIs
            print(f"Using ROI Pooling. Manifold Input Dim: {input_dim}")
            pool_type = 'conv' # We will compute cov on (B, 13, T), so 'conv' pool (mean cov) handles this fine?
            # Actually cov_pool_type='all5_timecat' expects 5 bands concat.
            # But if ROI pooling happens first, we are just passing T dim.
            # If bands_mode=all5_timecat, T is effectively 5*Win. 
            # Our ROI Pooling will return (B, 13, T_eff).
            # So DeepSPDClassifier just sees a standard trace of 13 channels.
            # Standard CovPool (based on cov(X)) is fine.
            # The 'all5_timecat' special pool type was for weird reshaping if needed?
            # It seems 'all5_timecat' logic just reshapes. If we are already correct shape, 'conv' or 'mean' is fine.
            # Let's verify 'conv' meaning in spdnet.
            
        elif self.bands_mode == 'all5_timecat':
             pool_type = 'all5_timecat'
        

        self.model = DeepSPDClassifier(
            n_channels=input_dim,
            deep_layers=0, 
            n_classes=self.num_classes, 
            output_dim=32, 
            cov_eps=self.spd_eps, 
            cov_alpha=cov_alpha,
            init_identity=True,
            cov_pool_type=pool_type,
            hidden_dim=hidden_dim
        )
        self.model.to(self.device).double()
        
        self.model.to(self.device).double()
        
        # Optimizer: Include BandGate params if active
        params = list(self.model.parameters())
        if self.use_band_gate:
            params += list(self.band_gate_model.parameters())
        
        # Lower LR for manifold stability
        optimizer = optim.Adam(params, lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_val_wrong_conf = 1.0
        best_epoch = 0
        best_state = None
        
        
        # Diagnostics Flag
        diag_done = False
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            A_list = [] # Stats
            
            for Xb, yb, tib in train_loader: 
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                
                logits = None
                
                # MVP1 Weighting Logic
                if self.mvp1_guided_cov:
                    A62 = self._compute_saliency(Xb.float())
                    weights = A62.unsqueeze(1).pow(self.mvp1_attn_power).double()
                    
        
                if self.bands_mode == "all5_timecat":
                    # model_in dim check
                    # Dataset now returns (B, 5, 62, T)
                    # If using BandGate:
                    if self.use_band_gate:
                        # Xb is (B, Win, 62, 5) due to return_5band=True via guided flag
                        # Gate needs (B, 5, 62, Win)
                        Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()
                        
                        # Phase 13D: Manual Normalization
                        if self.band_norm_mode == "manual_per_band_time":
                            # Mean/Std over Time (dim 3)
                            # Xb_perm: (B, 5, 62, T)
                            mean = Xb_perm.mean(dim=3, keepdim=True)
                            std = Xb_perm.std(dim=3, keepdim=True) + 1e-6
                            Xb_perm = (Xb_perm - mean) / std
                        elif self.band_norm_mode == "per_band_global_z":
                            # Phase 13D Step 1.5: Global across Chan+Time (dim 2, 3)
                            # Xb_perm: (B, 5, 62, T)
                            mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                            std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                            Xb_perm = (Xb_perm - mean) / std
                            
                        # If gate module exists (BandGateV1), use it. 
                        # But for Step 1.5 Manifold-Only, we might not have a gate model if use_band_gate=False?
                        # The block we are in is `if self.use_band_gate:`.
                        # But user request says "Manifold-Only (No DCNet)". 
                        # It doesn't explicitly say "No BandGate", but implies it: "Eliminate DCNet guidance/teacher/gate".
                        # So Step 1.5 should run in the `else` block of `if self.use_band_gate`.
                        # However, the user also says: "In the batch path right before CovPool/Cov computation... Apply per-band GLOBAL z-score".
                        # And "Re-pack back to [B, 62, 120]... in fixed band order".
                        # This logic needs to exist in the `else` block too (Control/ManifoldOnly path).
                        
                        w, feat = self.band_gate_model(Xb_perm) # (B, 5)
                        
                        # Log Stats
                        if epoch == 0 or epoch == self.epochs - 1:
                             self.gate_stats.append({
                                 "epoch": epoch,
                                 "w_mean": w.mean(0).detach().cpu().numpy().tolist(),
                                 "w_std": w.std(0).detach().cpu().numpy().tolist(),
                                 "w_min": w.min(0)[0].detach().cpu().numpy().tolist(),
                                 "w_max": w.max(0)[0].detach().cpu().numpy().tolist()
                             })

                        if self.band_activation_mode == "input_gate_v1":
                            # Phase 13D: Input Gating
                            # w: (B, 5) -> (B, 5, 1, 1)
                            w_sqrt = torch.sqrt(w + 1e-12).view(w.size(0), 5, 1, 1)
                            X_gated = Xb_perm * w_sqrt # (B, 5, 62, Win)
                            
                            # Time Concatenation (T_eff = 120)
                            # (B, 5, 62, Win) -> (B, 62, 5, Win) -> (B, 62, 5*Win)
                            model_in = X_gated.permute(0, 2, 1, 3).reshape(X_gated.size(0), 62, -1)
                            
                            # Diagnostics (First epoch, first batch)
                            if epoch == 0 and logits is None:
                                self._log_spectrum_diagnostics(model_in, fold_name)
                                
                            logits = self.model(model_in)

                        else:
                            # Forward SPDNet (Skip CovPool)
                            # Original Phase 13C-4 logic
                            cov_w = self.cov_pool_weighted(Xb_perm, w) # (B, 62, 62) SPD
                            logits = self.model.forward_from_cov(cov_w)
                    else:
                        # Control Mode: Manually Concatenate Time
                        # (B, 5, 62, T) -> (B, 62, T*5)
                        
                        # In all5_timecat mode, Xb is (B, 5, 62, T) ? 
                        # Wait, trial_dataset returns (B, 62, Win*5) if unguided? No.
                        # Checked lines 723-726 roughly below.
                        # If unguided/no-return-5band, dataset returns 3D tensor?
                        # But we forced return_5band=True for all5_timecat.
                        # So Xb is (B, Win, 62, 5).
                        # Let's verify shape assumption. 
                        # TrialDataset returns X_win_5band: (Win, 62, 5) -> (Win, 62, 5).
                        # DataLoader stacks to (B, Win, 62, 5).
                        
                        # So for Control mode:
                        Xb_perm = Xb.permute(0, 3, 2, 1).contiguous() # (B, 5, 62, Win)
                        
                        # Phase 13D: Manual Normalization (Control needs this too for Step 1.5)
                        if self.band_norm_mode == "manual_per_band_time":
                             mean = Xb_perm.mean(dim=3, keepdim=True)
                             std = Xb_perm.std(dim=3, keepdim=True) + 1e-6
                             Xb_perm = (Xb_perm - mean) / std
                        elif self.band_norm_mode == "per_band_global_z":
                             mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                             std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                             Xb_perm = (Xb_perm - mean) / std
                        
                        # Model Input: (B, 62, 5*Win)
                        # Permute (B, 5, 62, Win) -> (B, 62, 5, Win) -> Reshape
                        model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.shape[0], 62, -1)
                        
                        if self.use_roi_pooling:
                            # Apply ROI Pooling: (B, 62, T) -> (B, 13, T)
                            model_in = self.roi_pool(model_in)
                            
                            # Probe Post-ROI
                            if epoch == 0 and logits is None:
                                with open(f"promoted_results/{fold_name}_probe_post_roi.json", "w") as f:
                                    json.dump({
                                        "probe_point": "post_roi",
                                        "x_shape": list(model_in.shape),
                                        "roi_enabled": True
                                    }, f, indent=2)
                        
                        # Probe Pre-Cov
                        if epoch == 0 and logits is None:
                             with open(f"promoted_results/{fold_name}_probe_pre_cov.json", "w") as f:
                                 json.dump({
                                     "probe_point": "pre_cov",
                                     "x_shape": list(model_in.shape),
                                     "roi_enabled": self.use_roi_pooling,
                                     "bands_mode": self.bands_mode,
                                     "band_norm_mode": self.band_norm_mode,
                                     "expected_cov_dim": model_in.shape[1]
                                 }, f, indent=2)
                        
                        # Diagnostics (First epoch, first batch)
                        if epoch == 0 and logits is None:
                            self._log_spectrum_diagnostics(model_in, fold_name)

                        # Unified Matrix/Centering Logic
                        need_matrix = (self.matrix_mode == 'corr') or self.subject_centering or self.global_centering
                        
                        if need_matrix:
                             mat = self._compute_matrix_from_input(model_in)
                             
                             if self.subject_centering:
                                 mat = self._apply_centering(mat, tib, train_loader.dataset)
                             elif self.global_centering and self.log_ref_global is not None:
                                 # Phase 13E Step 4
                                 mat = self._apply_global_centering(mat, self.log_ref_global)

                             # Post-regularize (ensure strict SPD after operations)
                             mat = mat + torch.eye(mat.size(1), device=self.device).double() * self.spd_eps
                             
                             # Diagnostics (First Batch)
                             if epoch == 0 and not diag_done:
                                 with torch.no_grad():
                                     diff = torch.norm(mat.mean(0) - torch.eye(mat.size(1), device=self.device).double())
                                     print(f"[Centering Check] ||Mean(Batch) - I|| = {diff:.6f}")
                                     
                                     # Save to diagnostics
                                     diag_path = f"promoted_results/{fold_name}_diagnostics.json"
                                     update_json(diag_path, {"centering_identity_check": diff.item()})
                                     diag_done = True

                             logits = self.model.forward_from_cov(mat)
                        else:
                             logits = self.model(model_in)
                
                elif self.mvp1_guided_cov: # This block is for guided, but not all5_timecat
                    # Xb: (B, Win, 62, 5)
                    # cw = weights.unsqueeze(3) # (B, 1, 62, 1)
                    # x_w = Xb * cw
                    # Permute (B, Win, 62, 5) -> (B, 62, Win, 5) -> Reshape
                    # x_w_p = x_w.permute(0, 2, 1, 3)
                    # model_in = x_w_p.reshape(Xb.size(0), 62, -1)
                    
                    # If guided, and not all5_timecat, it means we are using a single band (band_idx=4)
                    # Xb is (B, Win, 62) in this case (from dataset)
                    x_gamma = Xb # Xb is already the single band data
                    x_w = x_gamma * weights
                    model_in = x_w.permute(0, 2, 1).double()
                    logits = self.model(model_in)
                elif self.bands_mode == "all5_timecat":
                    # Xb is already (B, 62, Win*5) from Dataset if unguided
                    model_in = Xb
                    logits = self.model(model_in)
                else:
                    model_in = Xb
                    logits = self.model(model_in)
                    
                # optimizer.zero_grad() # Moved up
                # logits = self.model(model_in) # Handled in branches
                loss = criterion(logits, yb)
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += Xb.size(0)
            
            train_loss = total_loss / total
            train_acc = correct / total
            
            # Validation on Sub-Split
            val_metrics, _, _ = self.evaluate(val_loader, f"{fold_name}_val", silent=True)
            val_acc = val_metrics['trial_agg_acc'] 
            val_wc = val_metrics.get('wrong_conf', 1.0)
            
            # Check Best
            is_best = False
            if val_acc > best_val_acc:
                is_best = True
            elif abs(val_acc - best_val_acc) < 1e-4 and val_wc < best_val_wrong_conf:
                is_best = True
                
            if is_best:
                best_val_acc = val_acc
                best_val_wrong_conf = val_wc
                best_epoch = epoch + 1
                best_state = self.model.state_dict()
                # Save Best Checkpoint
                torch.save(best_state, f"experiments/checkpoints/{fold_name}_best.pt")
                
            # Log
            a_stat = ""
            if len(A_list) > 0:
                 a_stat = f"| A: {np.mean(A_list):.2f}"
            
            print(f"Ep {epoch+1:02d} | Loss: {train_loss:.4f} TrAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} ValWC: {val_wc:.4f} {a_stat} | {'*' if is_best else ''}")
            
            # Diagnostics Export
            self._log_diagnostics(epoch, train_loader, fold_name, val_acc, val_wc)
            
            # Lambda Min Stats (Phase 13C-1)
            # Sample first batch
            if True:
                 try:
                     X_samp, _, _ = next(iter(train_loader))
                     X_samp = X_samp.to(self.device).double()
                     # If guided, process it
                     if self.mvp1_guided_cov:
                         # ... (Skipping guided logic for lambda check for now, unless complex)
                         # Guided reshapes input.
                         # For lambda min, we care about the input to CovPool.
                         # If guided, X_samp is (B, Win, 62, 5).
                         pass 
                     else:
                         # X_samp is (B, 62, T_eff)
                         B, C, T = X_samp.shape
                         # Compute Cov
                         # Centering
                         mean = X_samp.mean(dim=2, keepdim=True)
                         X_c = X_samp - mean
                         # Cov
                         cov = torch.matmul(X_c, X_c.transpose(1, 2)) / (T - 1)
                         # Add eps
                         cov = cov + torch.eye(C, device=self.device).double() * self.spd_eps
                         # Eigvals
                         eigs = torch.linalg.eigvalsh(cov)
                         min_eigs = eigs[:, 0]
                         
                         l_min = min_eigs.min().item()
                         l_p05 = torch.quantile(min_eigs, 0.05).item()
                         l_med = min_eigs.median().item()
                         
                         # Log
                         with open(f"promoted_results/{fold_name}_lambda_stats.csv", "a") as f:
                              if epoch == 0: f.write("epoch,l_min,l_p05,l_med\n")
                              f.write(f"{epoch+1},{l_min:.6e},{l_p05:.6e},{l_med:.6e}\n")
                 except Exception as e:
                     print(f"Warning: Lambda Stat failed: {e}")

        print(f"Training Done. Best Epoch: {best_epoch} (ValAcc: {best_val_acc:.4f})")
        
        # Save Last
        last_state = self.model.state_dict()
        torch.save(last_state, f"experiments/checkpoints/{fold_name}_last.pt")
        
        results = {}
        
        # Eval Test with Best
        if best_state is not None:
            self.model.load_state_dict(best_state)
        # Else use last, but best_state should be set unless validation fails completely (0%)
        
        test_metrics_best, test_preds_best, test_agg_best = self.evaluate(test_loader, f"{fold_name}_test_best")
        
        test_agg_best['seed'] = seed_val
        test_agg_best.to_csv(f"promoted_results/{fold_name}_preds_test_best.csv", index=False)
        test_preds_best.to_csv(f"promoted_results/{fold_name}_preds_test_best_raw.csv", index=False)
        
        results['best'] = {
            'epoch': best_epoch,
            'val_acc': best_val_acc,
            'test_win_acc': test_metrics_best['win_acc'],
            'test_trial_acc': test_metrics_best['trial_agg_acc'],
            'test_wrong_conf': test_metrics_best.get('wrong_conf', 0.0)
        }
        
        # Eval Test with Last
        self.model.load_state_dict(last_state)
        test_metrics_last, test_preds_last, test_agg_last = self.evaluate(test_loader, f"{fold_name}_test_last")
        
        # Save Window-Level
        test_preds_last.to_csv(f"promoted_results/{fold_name}_preds_test_last_window.csv", index=False)
        
        # Save Trial-Level (Standardized for Fusion)
        # Add seed info locally if possible, or leave for post-processing
        # Standard columns: trial_id, true_label, prob_0, prob_1, prob_2
        test_agg_last['seed'] = seed_val
        test_agg_last.to_csv(f"promoted_results/{fold_name}_preds_test_last_trial.csv", index=False)
        
        results['last'] = {
            'epoch': self.epochs,
            'test_win_acc': test_metrics_last['win_acc'],
            'test_trial_acc': test_metrics_last['trial_agg_acc'],
            'test_wrong_conf': test_metrics_last.get('wrong_conf', 0.0)
        }
        
        # Save JSON
        # Metadata
        results['metadata'] = {
            'dataset': 'seed1', # infer from somewhere? hardcoded for now or use adapter
            'seed': seed_val,
            'epochs': self.epochs,
            'batch_size': batch_size,
            'lr': 1e-4, # hardcoded in runner
            'weight_decay': 0.0,
            'split_mode': "trial_80_20", 
            'audit_key': audit_res['audit_key_definition'],
            'guided': self.mvp1_guided_cov,
            'attn_power': self.mvp1_attn_power,
            'teacher_ckpt_path': self.dcnet_ckpt,
            'bands_mode': self.bands_mode,
            'band_norm_mode': self.band_norm_mode,
            'use_roi_pooling': getattr(self, 'use_roi_pooling', False),
            'roi_k': 13 if getattr(self, 'use_roi_pooling', False) else 62, 
            'band_idx': 4,
            'T': window_len,
            'T_eff': window_len * 5 if self.bands_mode == 'all5_timecat' else window_len,
            'timecat_order': 'b0b1b2b3b4',
            'spd_eps': self.spd_eps,
            'aggregation_method': 'mean',
            'counts': {
                'train_trials': len(tr_ids),
                'test_trials': len(te_ids),
                'train_windows': len(train_dset),
                'test_windows': len(test_dset)
            },
            'audit_passed': bool(len(intersect) == 0),
            'device': str(self.device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            'code_version': "phase13c1b"
        }
        
        # Calculate SPD Stats on Sample (Phase 13C-1b)
        print("Computing SPD Stats for Report...")
        spd_train = self._compute_spd_stats(train_loader, n_samples=256)
        spd_test = self._compute_spd_stats(test_loader, n_samples=256) # Use consistent loader
        
        results['metadata']['spd_stats'] = {
            'train': spd_train,
            'test': spd_test
        }
        
        # Save Dictionary of Results (metrics already inside 'best'/'last')
        with open(f"promoted_results/{fold_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Save Gate Stats if exists
        if self.use_band_gate and self.gate_stats:
            with open(f"promoted_results/{fold_name}_gate_stats.json", "w") as f:
                 json.dump(self.gate_stats, f, indent=2)
            
        # Save Metadata (already part of results)
        # with open(f"promoted_results/{fold_name}_metadata.json", "w") as f:
        #     json.dump(metadata, f, indent=2)
            
        # Status Done
        with open(f"promoted_results/{fold_name}_status.txt", "a") as f:
            f.write(f"Status: COMPLETED\nEnd Time: {pd.Timestamp.now()}\nResult: Success\n")
            
        return results

    def _log_diagnostics(self, epoch, loader, fold_name, val_acc, val_wc):
        # ... (Similar to before but appending to file)
        # For brevity, implementing a minimal version calling the previous logic block
        # Or I can just paste the previous block logic here.
        model = self.model
        model.diagnostics_enabled = True
        with torch.no_grad():
             try:
                 Xb_diag, _, _ = next(iter(loader))
                 Xb_diag = Xb_diag.to(self.device).double()
                 
                 delta_cov = 0.0
                 delta_eig = 0.0
                 A_mean = 0.0
                 A_std = 0.0
                 
                 if self.bands_mode == "all5_timecat":
                     # Xb_diag: (B, Win, 62, 5)
                     Xb_perm = Xb_diag.permute(0, 3, 2, 1).contiguous()
                     if self.band_norm_mode == "manual_per_band_time":
                        mean = Xb_perm.mean(dim=3, keepdim=True)
                        std = Xb_perm.std(dim=3, keepdim=True) + 1e-6
                        Xb_perm = (Xb_perm - mean) / std
                     elif self.band_norm_mode == "per_band_global_z":
                        mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                        std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                        Xb_perm = (Xb_perm - mean) / std
                     
                     if self.use_band_gate:
                         w, _ = self.band_gate_model(Xb_perm)
                         if self.band_activation_mode == "input_gate_v1":
                             w_sqrt = torch.sqrt(w + 1e-12).view(w.size(0), 5, 1, 1)
                             X_gated = Xb_perm * w_sqrt
                             model_in_diag = X_gated.permute(0, 2, 1, 3).reshape(Xb_diag.size(0), 62, -1)
                         else:
                             cov_w = self.cov_pool_weighted(Xb_perm, w)
                             # Special handling: forward_from_cov does not register hooks?
                             # self.model.forward() registers hooks for diagnostics usually.
                             # If we bypass, we might miss diagnostics if hook is on forward.
                             # But let's assume forward_from_cov calls things that trigger hooks if they are on submodules?
                             # Actually DeepSPDClassifier doesn't utilize standard forward hooks for diag, 
                             # it has internal logic. 
                             # `model(model_in_diag)` calls forward().
                             # If I call `forward_from_cov`, I need to make sure diagnostics capture it.
                             # But wait, `_log_diagnostics` calls model(model_in_diag) to trigger the hook?
                             # No, line 1000 `_ = model(model_in_diag)`.
                             # If I can't construct `model_in_diag` (input tensor), I can't call `model(...)`.
                             # `forward_from_cov` takes covariance.
                             # For now, let's just NOT crash. If BandGate+CovPool is used, diagnostics might be tricky here.
                             # But this is "Manifold Only" task (Step 1.5). use_band_gate=False.
                             pass
                         pass
                     else:
                         model_in_diag = Xb_perm.permute(0, 2, 1, 3).reshape(Xb_diag.size(0), 62, -1)
                         
                         if self.use_roi_pooling:
                             model_in_diag = self.roi_pool(model_in_diag)
                 
                 elif self.mvp1_guided_cov:
                      A62 = self._compute_saliency(Xb_diag.float())
                      x_g = Xb_diag[:, :, :, 4] 
                      weights = A62.unsqueeze(1).pow(self.mvp1_attn_power).double()
                      x_w_time = x_g * weights
                      x_orig_p = x_g.permute(0, 2, 1)
                      x_w_p = x_w_time.permute(0, 2, 1)
                      delta_cov, delta_eig = self._compute_effect_size(x_orig_p, x_w_p)
                      model_in_diag = x_w_p
                      A_mean = A62.mean().item()
                      A_std = A62.std().item()
                 else:
                      model_in_diag = Xb_diag
                      
                 if 'model_in_diag' in locals():
                     _ = model(model_in_diag)
             except StopIteration:
                 pass
        model.diagnostics_enabled = False
        
        stats = model.last_diagnostics
        if stats:
            stats['epoch'] = epoch + 1
            stats['val_acc'] = val_acc
            stats['val_wc'] = val_wc
            stats['A_mean'] = A_mean
            stats['A_std'] = A_std
            stats['delta_cov_fro'] = delta_cov
            stats['delta_eig_l1'] = delta_eig
            
            # Append to CSV
            log_path = f"promoted_results/{fold_name}_diag_effect.csv"
            df = pd.DataFrame([stats])
            if not os.path.exists(log_path):
                df.to_csv(log_path, index=False)
            else:
                df.to_csv(log_path, mode='a', header=False, index=False)

    def evaluate(self, loader, fold_name, silent=False):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_probs = []
        all_labels = []
        all_trials = []
        
        with torch.no_grad():
            for Xb, yb, tib in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                
                logits = None
                
                if self.bands_mode == "all5_timecat":
                    # Xb: (B, Win, 62, 5) - From forced return_5band=True
                    Xb_perm = Xb.permute(0, 3, 2, 1).contiguous() # (B, 5, 62, Win)
                    
                    # Norm
                    if self.band_norm_mode == "manual_per_band_time":
                        mean = Xb_perm.mean(dim=3, keepdim=True)
                        std = Xb_perm.std(dim=3, keepdim=True) + 1e-6
                        Xb_perm = (Xb_perm - mean) / std
                    elif self.band_norm_mode == "per_band_global_z":
                        mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                        std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                        Xb_perm = (Xb_perm - mean) / std
                    
                    if self.use_band_gate:
                         w, _ = self.band_gate_model(Xb_perm)
                         if self.band_activation_mode == "input_gate_v1":
                             w_sqrt = torch.sqrt(w + 1e-12).view(w.size(0), 5, 1, 1)
                             X_gated = Xb_perm * w_sqrt
                             model_in = X_gated.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                         else:
                             cov_w = self.cov_pool_weighted(Xb_perm, w)
                             logits = self.model.forward_from_cov(cov_w)
                    else:
                        # Manifold Only / Control
                        Xb_perm = Xb.permute(0, 3, 2, 1).contiguous()
                        
                        if self.band_norm_mode == "manual_per_band_time":
                             mean = Xb_perm.mean(dim=3, keepdim=True)
                             std = Xb_perm.std(dim=3, keepdim=True) + 1e-6
                             Xb_perm = (Xb_perm - mean) / std
                        elif self.band_norm_mode == "per_band_global_z":
                             mean = Xb_perm.mean(dim=(2, 3), keepdim=True)
                             std = Xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
                             Xb_perm = (Xb_perm - mean) / std
                        
                        model_in = Xb_perm.permute(0, 2, 1, 3).reshape(Xb.size(0), 62, -1)
                        
                        if self.use_roi_pooling:
                            model_in = self.roi_pool(model_in)
                        
                        # Unified Matrix/Centering Logic
                        need_matrix = (self.matrix_mode == 'corr') or self.subject_centering or self.global_centering
                        
                        if need_matrix:
                             mat = self._compute_matrix_from_input(model_in)
                             
                             if self.subject_centering:
                                 mat = self._apply_centering(mat, tib, loader.dataset)
                             elif self.global_centering and self.log_ref_global is not None:
                                 mat = self._apply_global_centering(mat, self.log_ref_global)

                             # Post-regularize
                             mat = mat + torch.eye(mat.size(1), device=self.device).double() * self.spd_eps
                                 
                             logits = self.model.forward_from_cov(mat)
                        else:
                             logits = self.model(model_in)
                        
                elif self.mvp1_guided_cov:
                    A62 = self._compute_saliency(Xb.float())
                    weights = A62.unsqueeze(1).pow(self.mvp1_attn_power).double()
                    # Single band (band_idx=4)
                    x_gamma = Xb[:, :, :, 4]
                    x_w = x_gamma * weights
                    model_in = x_w.permute(0, 2, 1).double()
                else:
                    model_in = Xb
                
                if logits is None:
                    logits = self.model(model_in)
                preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)
                
                correct += (preds == yb).sum().item()
                total += Xb.size(0)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
                all_trials.append(tib.cpu().numpy())
                
        if total == 0:
            return {'win_acc': 0.0, 'trial_agg_acc': 0.0}, None
            
        win_acc = correct / total
        
        # Trial Aggregation
        final_probs = np.concatenate(all_probs, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        final_trials = np.concatenate(all_trials, axis=0)
        
        unique_trials = np.unique(final_trials)
        agg_correct = 0
        agg_total = 0
        
        wrong_conf_sum = 0.0
        wrong_count = 0
        
        for t_idx in unique_trials:
             mask = (final_trials == t_idx)
             trial_probs = final_probs[mask]
             trial_label = final_labels[mask][0]
             
             mean_prob = np.mean(trial_probs, axis=0)
             pred_label = np.argmax(mean_prob)
             
             if pred_label == trial_label:
                 agg_correct += 1
             else:
                 # Wrong Confidence
                 wrong_conf_sum += np.max(mean_prob)
                 wrong_count += 1
                 
             agg_total += 1
             
        agg_acc = agg_correct / agg_total if agg_total > 0 else 0.0
        wrong_conf = wrong_conf_sum / wrong_count if wrong_count > 0 else 0.0
        
        if not silent:
            print(f"[{fold_name}] Final Results -> Window Acc: {win_acc:.4f} | Trial Agg Acc: {agg_acc:.4f}")
            
        metrics = {
            "win_acc": win_acc,
            "trial_agg_acc": agg_acc,
            "wrong_conf": wrong_conf
        }
        
        # Window-DF
        df_win = pd.DataFrame(final_probs, columns=["prob_0", "prob_1", "prob_2"])
        df_win["true_label"] = final_labels
        
        # Map numeric index to String ID using Dataset Metadata
        if hasattr(loader.dataset, 'trial_ids_map'):
             ids_map = loader.dataset.trial_ids_map
             # final_trials are indices (int)
             real_ids = [ids_map[idx] for idx in final_trials]
             df_win["trial_id"] = real_ids
        else:
             df_win["trial_id"] = final_trials
        
        df_agg = df_win.groupby("trial_id").agg({
             "true_label": "first",
             "prob_0": "mean",
             "prob_1": "mean", 
             "prob_2": "mean"
        }).reset_index()
        df_agg["pred_label"] = np.argmax(df_agg[["prob_0", "prob_1", "prob_2"]].values, axis=1)
        
        # Phase 13A: Add Entropy and Margin
        probs = df_agg[["prob_0", "prob_1", "prob_2"]].values
        # Entropy: -sum(p * log(p))
        # Add epsilon to avoid log(0)
        p_safe = np.clip(probs, 1e-9, 1.0)
        df_agg["entropy"] = -np.sum(p_safe * np.log(p_safe), axis=1)
        
        # Margin: Top1 - Top2
        p_sorted = np.sort(probs, axis=1)
        df_agg["margin"] = p_sorted[:, -1] - p_sorted[:, -2]
        
        return metrics, df_win, df_agg
