
import torch
import torch.nn as nn

class CovPool(nn.Module):
    """
    Differentiable Covariance Pooling.
    Input: (B, C, T) - temporal features
    Output: (B, C, C) - SPD matrices
    """
    def __init__(self, eps=1e-4, alpha=None):
        """
        Args:
            eps (float): Diagonal regularization (default 1e-4).
            alpha (float, optional): Linear shrinkage coefficient (0 to 1).
                                     If None, no shrinkage is applied (only eps).
                                     Formula: C = (1-a)*C + a*(trace(C)/n)*I
        """
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        # x: (Batch, Channels, Time)
        B, C, T = x.shape
        # Center the data
        mean = x.mean(dim=2, keepdim=True)
        xm = x - mean
        # Unbiased estimate: 1/(T-1) * X @ X.T
        cov = torch.bmm(xm, xm.transpose(1, 2)) / (T - 1)
        
        ident = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
        
        # Shrinkage (Ledoit-Wolf style target: Scaled Identity)
        if self.alpha is not None and 0 < self.alpha < 1:
            trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1) # (B,)
            avg_var = trace / C
            target = avg_var.view(B, 1, 1) * ident
            cov = (1 - self.alpha) * cov + self.alpha * target
            
        # Diagonal Regularization for numerical stability (ensure strictly positive definite)
        # Applied after shrinkage or alone
        return cov + self.eps * ident


def perturb_diagonal(x, eps=1e-6):
    """
    Add small perturbation to diagonal to ensure distinct eigenvalues.
    x: (B, C, C)
    """
    B, C, _ = x.shape
    device = x.device
    # Add varying noise to diagonal: diag[i] += eps * i
    # This separates clustered eigenvalues (like noise floor)
    steps = torch.arange(C, device=device, dtype=x.dtype).unsqueeze(0) # (1, C)
    noise = eps * steps
    
    # torch.diagonal returns a view, but we can add directly
    # Or use eye construction for batch safety
    ident_noise = torch.diag_embed(noise).expand(B, -1, -1)
    return x + ident_noise


def compute_spectrum_stats(x, name="layer", tau=1e-5):
    """
    Compute spectral statistics for a batch of SPD matrices.
    x: (B, C, C)
    """
    with torch.no_grad():
        vals = torch.linalg.eigvalsh(x) # (B, C)
        # Avoid zero/negative for conditioning if possible, or just take raw
        # Typically we care about max/min ratio
        
        eig_min = vals.min(dim=1).values.mean().item()
        eig_max = vals.max(dim=1).values.mean().item()
        eig_median = vals.median(dim=1).values.mean().item()
        
        # Conditioning
        # Filter out very small values for conditioning calculation to avoid inf
        v_clamped = torch.clamp(vals, min=1e-9)
        cond = (v_clamped.max(dim=1).values / v_clamped.min(dim=1).values).mean().item()
        
        # Rank tau
        rank_tau = (vals > tau).float().sum(dim=1).mean().item()
        
        # Effective Rank (Shannon Entropy)
        # p_i = lambda_i / sum(lambda)
        # H = -sum(p_i * log(p_i))
        # eff_rank = exp(H)
        v_abs = vals.abs()
        p = v_abs / v_abs.sum(dim=1, keepdim=True).clamp(min=1e-9)
        entropy = -(p * torch.log(p.clamp(min=1e-9))).sum(dim=1)
        eff_rank = torch.exp(entropy).mean().item()
        
        # Effective Rank (Participation Ratio)
        # R = (sum(lambda))^2 / sum(lambda^2)
        sum_eig = v_abs.sum(dim=1)
        sum_eig_sq = (v_abs ** 2).sum(dim=1)
        eff_rank_p = (sum_eig ** 2 / sum_eig_sq.clamp(min=1e-9)).mean().item()
        
        return {
            f"{name}_eig_min": eig_min,
            f"{name}_eig_max": eig_max,
            f"{name}_eig_median": eig_median,
            f"{name}_cond": cond,
            f"{name}_rank_tau": rank_tau,
            f"{name}_eff_rank": eff_rank,
            f"{name}_eff_rank_p": eff_rank_p
        }

class BiMap(nn.Module):
    """
    Bilinear Mapping Layer (Manifold FC).
    Map SPD(C_in) -> SPD(C_out) via W.T @ X @ W
    """
    def __init__(self, c_in, c_out, init_identity=True):
        super().__init__()
        # W shape: (C_in, C_out) - simplified geometric interpretation
        self.W = nn.Parameter(torch.empty(c_in, c_out))
        if init_identity:
            # W = [I, 0] + noise
            # Create identity block
            with torch.no_grad():
                nn.init.orthogonal_(self.W) # Start with orthogonal base
                # If reducing, we want to look like PCA selection roughly
                # Identity-like means W_ij = delta_ij
                # But we want to respect the manifold structure
                # Orthogonal init is close to "Identity on Stiefel"
                # But user asked for [I, 0] explicitly.
                # Let's try to bias it towards Identity.
                self.W.data.fill_(0.0)
                min_dim = min(c_in, c_out)
                for i in range(min_dim):
                    self.W.data[i, i] = 1.0
                # Add noise
                self.W.data += 0.01 * torch.randn_like(self.W.data)
                # Re-project to Stiefel
                U, _, V = torch.linalg.svd(self.W.data, full_matrices=False)
                self.W.data = U @ V
        else:
            nn.init.orthogonal_(self.W) # Orthogonal init is usually better for manifold weights

    def forward(self, x):
        stats = {}
        # x: (B, C_in, C_in)
        # Standard bilinearform: y = W^T * x * W
        
        tx = torch.matmul(x, self.W) # (B, n, m)
        ty = torch.matmul(self.W.t(), tx) # (B, m, m) broadcast W.t() (m, n) across batch
        return ty

class ReEig(nn.Module):
    """
    Rectified Eigenvalues (Manifold ReLU).
    U @ max(S, eps) @ U.T
    """
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, C, C) - symmetric
        
        # Perturb for gradient stability if eigenvalues clustered
        x_p = perturb_diagonal(x, eps=1e-6)
        
        # Eigen decomposition
        # NOTE: eigh is for symmetric matrices. 
        vals, vecs = torch.linalg.eigh(x_p) 
        
        # Rectify: max(val, eps)
        vals = torch.clamp(vals, min=self.eps)
        
        # Reconstruct: U @ diag(vals) @ U.T
        # diag_vals: (B, C, C)
        diag_vals = torch.diag_embed(vals)
        
        output = torch.bmm(torch.bmm(vecs, diag_vals), vecs.transpose(1, 2))
        return output

class LogEig(nn.Module):
    """
    Matrix Logarithm Map (Riemannian -> Euclidean Tangent Space).
    U @ log(S) @ U.T
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, C, C)
        
        # Perturb for gradient stability
        x_p = perturb_diagonal(x, eps=1e-6)
        
        vals, vecs = torch.linalg.eigh(x_p)
        # For Log map, strictly positive eigenvalues are required.
        vals = torch.clamp(vals, min=self.eps)
        vals_log = torch.log(vals)
        
        # Reconstruct
        diag_log = torch.diag_embed(vals_log)
        out_mat = torch.bmm(torch.bmm(vecs, diag_log), vecs.transpose(1, 2))
        return out_mat

class UpperTriVectorize(nn.Module):
    """
    Flatten upper triangle of symmetric matrix.
    Input: (B, C, C)
    Output: (B, C*(C+1)/2)
    """
    def forward(self, x):
        B, C, _ = x.shape
        # Get upper triangle indices
        # We can compute once if size fixed, but dynamic is safer
        idx = torch.triu_indices(C, C, device=x.device)
        # x[:, idx[0], idx[1]] -> (B, num_entries)
        return x[:, idx[0], idx[1]]

class DeepSPDClassifier(nn.Module):
    """
    Initial version: Cov -> [BiMap->ReEig]*N -> LogEig -> Vec -> MLP
    According to request:
    'Initial version... Cov -> LogMap -> MLP'
    'Once 65%+ ... add BiMap'
    So, defaults will be minimal, but structure supports stacking.
    """
    def __init__(self, n_channels=62, deep_layers=0, n_classes=3, hidden_dim=64, cov_eps=1e-4, cov_alpha=None, output_dim=32, init_identity=True, cov_pool_type='conv'):
        super().__init__()
        
        # Select CovPool based on type
        if cov_pool_type == 'conv':
            self.cov_pool = CovPool(eps=cov_eps, alpha=cov_alpha)
        elif cov_pool_type == 'all5_timecat':
            # Assuming input is reshaped outside, so standard CovPool works?
            # Or does it need special handling?
            # Runner reshapes (B, 5, 62, T) -> (B, 62, 5*T)
            # So standard CovPool works on the 2D trace.
            self.cov_pool = CovPool(eps=cov_eps, alpha=cov_alpha)
        else:
            raise ValueError(f"Unknown cov_pool_type: {cov_pool_type}")
            
        # Manifold Layers (BiMap + ReEig blocks)
        self.manifold_layers = nn.ModuleList()
        current_dim = n_channels
        
        # If output_dim < n_channels, we use BiMap to reduce dim
        # We can simply add one BiMap layer 62 -> 32
        if output_dim < n_channels:
             # Just one layer for now as requested (62 -> 32)
             self.manifold_layers.append(BiMap(current_dim, output_dim, init_identity=init_identity))
             self.manifold_layers.append(ReEig(eps=cov_eps))
             current_dim = output_dim
        
        self.log_eig = LogEig(eps=cov_eps)
        self.vectorize = UpperTriVectorize()
        
        # MLP Classifier
        # Input dim is based on the tangent space size (upper triangular for output_dim)
        vec_dim = current_dim * (current_dim + 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # Diagnostics buffer
        self.diagnostics_enabled = False
        self.last_diagnostics = {}

    def get_embedding(self, x):
        """
        Returns the vector embedding (UpperTri) before the MLP classifier.
        Output: (B, vec_dim)
        """
        stats = {}
        if x.ndim == 4: x = x.squeeze(1)
        if x.shape[1] != 62 and x.shape[2] == 62: x = x.transpose(1, 2)
            
        # Cov Pool
        cov = self.cov_pool(x)
        if self.diagnostics_enabled:
            stats.update(compute_spectrum_stats(cov, "CovPool"))
        
        # Manifold layers
        for i, layer in enumerate(self.manifold_layers):
            cov = layer(cov)
            if self.diagnostics_enabled:
                if isinstance(layer, BiMap):
                    name = f"BiMap_{i}"
                    stats.update(compute_spectrum_stats(cov, name))
                    W = layer.W
                    WtW = torch.matmul(W.t(), W)
                    I = torch.eye(WtW.shape[0], device=W.device, dtype=W.dtype)
                    orth_err = torch.norm(WtW - I, p='fro').item()
                    stats[f"{name}_orth_err"] = orth_err
                elif isinstance(layer, ReEig):
                    stats.update(compute_spectrum_stats(cov, f"ReEig_{i}"))
            
        # Tangent Space
        log_cov = self.log_eig(cov)
        
        # Vectorize
        vec = self.vectorize(log_cov)
        
        if self.diagnostics_enabled:
            self.last_diagnostics = stats
            
        return vec

    def forward(self, x):
        vec = self.get_embedding(x)
        logits = self.mlp(vec)
        return logits

    def forward_from_cov(self, cov):
        """
        Bushes a pre-computed covariance matrix through the manifold layers and MLP.
        Used for BandGate integration where covariance is computed externally.
        cov: (B, C, C) SPD
        """
        stats = {}
        
        # Manifold layers
        for i, layer in enumerate(self.manifold_layers):
            cov = layer(cov)
            if self.diagnostics_enabled:
                if isinstance(layer, BiMap):
                    name = f"BiMap_{i}"
                    stats.update(compute_spectrum_stats(cov, name))
                elif isinstance(layer, ReEig):
                    stats.update(compute_spectrum_stats(cov, f"ReEig_{i}"))
            
        # Tangent Space
        log_cov = self.log_eig(cov)
        
        # Vectorize
        vec = self.vectorize(log_cov)
        
        if self.diagnostics_enabled:
            self.last_diagnostics = stats
            
        logits = self.mlp(vec)
        return logits

class CovPoolWeightedBandsV1(nn.Module):
    """
    Weighted Sum of Covariances from multiple bands.
    Input: x (B, 5, 62, T), w (B, 5)
    Output: C (B, 62, 62) SPD Matrix
    
    Logic:
      For each band b:
        Cb = Cov(x[:, b, :, :])  (Centered, 1/(T-1))
      C_out = Sum_b (w_b * Cb)
      Apply eps regularization.
    """
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
        
    def forward(self, x, w):
        # x: (B, 5, C, T)
        # w: (B, 5)
        B, n_bands, C, T = x.shape
        
        # Center x per band: (B, 5, C, T) - (B, 5, C, 1)
        mean = x.mean(dim=3, keepdim=True)
        xm = x - mean
        
        # Compute Cov per band: (B, 5, C, C)
        # Batch Matmul: (B*5, C, T) @ (B*5, T, C) -> (B*5, C, C)
        xm_flat = xm.reshape(B * n_bands, C, T)
        cov_bands = torch.bmm(xm_flat, xm_flat.transpose(1, 2)) / (T - 1)
        cov_bands = cov_bands.view(B, n_bands, C, C)
        
        # Weighted Sum
        # w: (B, 5) -> (B, 5, 1, 1)
        w_exp = w.view(B, n_bands, 1, 1)
        
        # Sum_b (w_b * Cb)
        cov_weighted = (cov_bands * w_exp).sum(dim=1) # (B, C, C)
        
        # Regularization
        ident = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
        return cov_weighted + self.eps * ident
