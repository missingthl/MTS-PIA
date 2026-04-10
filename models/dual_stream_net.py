import torch
from torch import nn


class DualStreamNet(nn.Module):
    def __init__(
        self,
        spatial_backbone: nn.Module,
        manifold_backbone: nn.Module,
        spatial_feat_dim: int,
        manifold_feat_dim: int,
        num_classes: int = 3,
        fusion_hidden: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.spatial_backbone = spatial_backbone
        self.manifold_backbone = manifold_backbone
        self.spatial_feat_dim = int(spatial_feat_dim)
        self.manifold_feat_dim = int(manifold_feat_dim)
        fusion_in = self.spatial_feat_dim + self.manifold_feat_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def extract_spatial_features(self, x: torch.Tensor):
        logits, feats = self.spatial_backbone(x, return_features=True)
        return logits, feats

    def extract_manifold_embedding_from_cov(self, cov: torch.Tensor):
        m = self.manifold_backbone
        for layer in m.manifold_layers:
            cov = layer(cov)
        log_cov = m.log_eig(cov)
        vec = m.vectorize(log_cov)
        return vec

    def forward_fusion(self, feat_spatial: torch.Tensor, feat_manifold: torch.Tensor):
        fused = torch.cat([feat_spatial, feat_manifold], dim=1)
        logits = self.fusion_head(fused)
        return logits

    def forward(self, feat_spatial: torch.Tensor, feat_manifold: torch.Tensor):
        logits_fusion = self.forward_fusion(feat_spatial, feat_manifold)
        return feat_spatial, feat_manifold, logits_fusion
