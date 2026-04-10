
import torch
import torch.nn as nn
import torch.nn.functional as F

class BandScalarGateV1(nn.Module):
    """
    Learns a scalar weight for each of the 5 frequency bands.
    Input: X_bands (B, 5, 62, T)
    Feature: Log-Variance of each band (averaged over channel/time) -> (B, 5)
    Gate: MLP (5 -> 16 -> 5) + Softmax
    Output: w (B, 5)
    """
    def __init__(self, n_bands=5, hidden_dim=16):
        super().__init__()
        self.n_bands = n_bands
        
        self.mlp = nn.Sequential(
            nn.Linear(n_bands, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bands)
        )
        
    def forward(self, x):
        # x: (B, 5, 62, T)
        B, n_b, n_ch, T = x.shape
        
        # 1. Feature Extraction: Log Variance
        # Var over (Channel, Time) -> (B, 5)
        # Add eps for log stability
        var = x.var(dim=(2, 3)) + 1e-6 
        feat = torch.log(var) # (B, 5)
        
        # 2. MLP
        logits = self.mlp(feat) # (B, 5)
        
        # 3. Softmax
        w = F.softmax(logits, dim=1) # (B, 5)
        
        return w, feat
