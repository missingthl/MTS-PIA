
import torch
import torch.nn as nn
from .spdnet import DeepSPDClassifier

class SharedMultiBandSPDNet(nn.Module):
    def __init__(self, n_classes=3, hidden_dim=96, input_dim=62, dropout=0.5):
        """
        Shared Backbone 5-Band Manifold Model.
        Args:
            n_classes: Number of output classes.
            hidden_dim: Dim for DeepSPDClassifier internals.
            input_dim: Input channel count (62).
            dropout: Dropout probability for fusion head.
        """
        super().__init__()
        
        # 1. Shared Backbone
        # Note: We assume DeepSPDClassifier exposes get_embedding()
        # n_channels replaces input_dim
        # Remove BiMap (output_dim=62) to debug signal flow
        # Reduce alpha to 1e-2
        self.backbone = DeepSPDClassifier(
            n_classes=n_classes, 
            n_channels=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=62, # No BiMap reduction
            cov_alpha=0.01,
            cov_eps=1e-4
        )
        
        # Determine embedding dimension
        # If output_dim=62 (No BiMap), vector dim is 62*63/2 = 1953
        # If output_dim=32, vector dim is 32*33/2 = 528
        # We need to calculate dynamically based on backbone config or hardcode for now
        # For output_dim=62:
        self.embed_dim = 62 * 63 // 2 # 1953
        
        # 2. Per-Band Adapters (LayerNorm)
        # 5 bands
        self.band_adapters = nn.ModuleList([
            nn.LayerNorm(self.embed_dim) for _ in range(5)
        ])
        
        # 3. Fusion Head
        # Concat 5 * 528 = 2640 -> 128 -> 3
        # Strongly regularized: Dropout(0.5)
        input_fusion = 5 * self.embed_dim
        bottleneck = 128
        
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_fusion, bottleneck),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(bottleneck, n_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 62, T, 5) or (B, T, 62, 5).
               We expect (B, 62, T, 5) based on Runner logic, but let's be safe.
               Actually, DeepSPDClassifier expects (B, C, T) where it computes cov per trial.
               Wait, DeepSPDClassifier input is (N, C, T).
               
               If input is DE features (B, 62, T, 5).
        """
        # Ensure (B, 62, T, 5)
        if x.ndim == 4:
            if x.shape[3] != 5 and x.shape[1] == 5:
                # Permute if necessary, but runner usually gives (B, 62, T, 5)
                x = x.permute(0, 2, 3, 1) # Assuming (B, 5, 62, T) -> ? No.
                pass
        
        # Extract Batch Size
        B = x.shape[0]
        
        embeddings = []
        
        for i in range(5):
            # Extract band i: (B, 62, T)
            # Assuming format (B, 62, T, 5) -> x[..., i] -> (B, 62, T)
            xb = x[..., i] 
            
            # Forward Backbone
            # DeepSPDClassifier.get_embedding(x) returns (B, 528)
            emb = self.backbone.get_embedding(xb)
            
            # Adapter
            emb = self.band_adapters[i](emb)
            
            embeddings.append(emb)
            
        # Concat: (B, 2640)
        concat = torch.cat(embeddings, dim=1)
        
        # Fusion
        logits = self.fusion_mlp(concat)
        
        return logits
