
import torch
import torch.nn as nn
from .spdnet import DeepSPDClassifier

class MultiBandDeepSPDClassifier(nn.Module):
    """
    5-Band SPDNet Wrapper:
    - 5 separate DeepSPDClassifier branches (one per band).
    - Per-branch output embedding (dim=528).
    - Per-branch LayerNorm(528).
    - Concatenation -> MLP -> Classification.
    """
    def __init__(self, 
                 num_classes=3, 
                 bands=5, 
                 shared_weights=False,
                 input_dim=62, 
                 output_dim=32, 
                 hidden_dim=96,
                 dropout_p=0.5):
        super().__init__()
        self.bands = bands
        self.shared_weights = shared_weights
        
        # 1. Branches
        if shared_weights:
            self.branch = DeepSPDClassifier(input_dim, num_classes, output_dim, hidden_dim, dropout_p)
            self.branches = nn.ModuleList([self.branch for _ in range(bands)])
        else:
            self.branches = nn.ModuleList([
                DeepSPDClassifier(input_dim, num_classes, output_dim, hidden_dim, dropout_p)
                for _ in range(bands)
            ])
            
        # 2. Normalization & Fusion
        # Assumes DeepSPDClassifier.get_embedding returns vector of size 528 (Flatten(32*33/2)=528)
        # We need to verify the exact embedding size. 
        # DeepSPDClassifier uses LogEig -> Flatten -> MLP. 
        # Actually DeepSPDClassifier doesn't expose `get_embedding` publicly in the previous file.
        # We will need to check/modify spdnet.py or replicate the logic here.
        # Let's check spdnet.py first.
        # Assuming embedding dim is 528 (triangular part of 32x32 matrix -> 32*33/2 = 528).
        
        self.embedding_dim = output_dim * (output_dim + 1) // 2
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(bands)
        ])
        
        # 3. Fusion MLP
        concat_dim = self.embedding_dim * bands
        fusion_hidden = 512
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_hidden, self.embedding_dim), # Project back to embedding size or keep separate
            nn.ReLU(inplace=True)
        )
        
        # 4. Classifier
        self.classifier = nn.Linear(self.embedding_dim, num_classes)


    def forward(self, x):
        # x: (B, 62, T, Bands)
        # We need to standardize input.
        # DeepSPD embedding expects (B, 62, T).
        
        # Check input dim
        if x.ndim != 4:
            raise ValueError(f"MultiBandDeepSPDClassifier expects (B, 62, T, Bands), got {x.shape}")
            
        B, C, T, Bands = x.shape
        if Bands != self.bands:
             # Try permutation if bands is not last
             if x.shape[1] == self.bands: # (B, Bands, 62, T)
                 x = x.permute(0, 2, 3, 1) # -> (B, 62, T, Bands)
             else:
                 raise ValueError(f"Expected {self.bands} bands, got {x.shape}")
        
        embeddings = []
        for b in range(self.bands):
            # Extract band b -> (B, 62, T)
            x_b = x[..., b] 
            
            # Get embedding -> (B, 528)
            # Use branch[0] if shared, else branches[b]
            if self.shared_weights:
                branch = self.branches[0]
            else:
                branch = self.branches[b]
            
            emb = branch.get_embedding(x_b)
            
            # Normalize
            emb = self.layer_norms[b](emb)
            embeddings.append(emb)
            
        # Concat -> (B, 528*5)
        fused = torch.cat(embeddings, dim=1)
        
        # Fusion MLP -> (B, 528)
        features = self.fusion_mlp(fused)
        
        # Classifier -> (B, K)
        logits = self.classifier(features)
        
        return logits
