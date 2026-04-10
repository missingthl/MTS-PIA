from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from models.raw_dcnet_adapter import RawDCNetTemporalAdapter


@dataclass(frozen=True)
class DualStreamModelConfig:
    channels: int
    seq_len: int
    z_dim: int
    num_classes: int
    spatial_proj_channels: int = 16
    spatial_proj_bins: int = 8
    manifold_hidden_dim: int = 128
    manifold_feature_dim: int = 64
    fusion_hidden_dim: int = 128
    dropout: float = 0.3
    dual_aux_weight: float = 0.3


class ManifoldMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(feature_dim)),
            nn.BatchNorm1d(int(feature_dim)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


class SpatialOnlyClassifier(nn.Module):
    def __init__(self, cfg: DualStreamModelConfig) -> None:
        super().__init__()
        self.backbone = RawDCNetTemporalAdapter(
            in_channels=int(cfg.channels),
            seq_len=int(cfg.seq_len),
            num_classes=int(cfg.num_classes),
            proj_channels=int(cfg.spatial_proj_channels),
            proj_bins=int(cfg.spatial_proj_bins),
            allow_variable_length=True,
        )

    def forward(self, raw_x: torch.Tensor, z: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        logits, feats = self.backbone(raw_x, return_features=True)
        return {
            "logits": logits,
            "spatial_logits": logits,
            "spatial_features": feats,
        }


class ManifoldOnlyClassifier(nn.Module):
    def __init__(self, cfg: DualStreamModelConfig) -> None:
        super().__init__()
        self.backbone = ManifoldMLPClassifier(
            input_dim=int(cfg.z_dim),
            feature_dim=int(cfg.manifold_feature_dim),
            hidden_dim=int(cfg.manifold_hidden_dim),
            num_classes=int(cfg.num_classes),
            dropout=float(cfg.dropout),
        )

    def forward(self, raw_x: torch.Tensor | None, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, feats = self.backbone(z, return_features=True)
        return {
            "logits": logits,
            "manifold_logits": logits,
            "manifold_features": feats,
        }


class DualStreamClassifier(nn.Module):
    def __init__(self, cfg: DualStreamModelConfig) -> None:
        super().__init__()
        self.aux_weight = float(cfg.dual_aux_weight)
        self.spatial_backbone = RawDCNetTemporalAdapter(
            in_channels=int(cfg.channels),
            seq_len=int(cfg.seq_len),
            num_classes=int(cfg.num_classes),
            proj_channels=int(cfg.spatial_proj_channels),
            proj_bins=int(cfg.spatial_proj_bins),
            allow_variable_length=True,
        )
        self.manifold_backbone = ManifoldMLPClassifier(
            input_dim=int(cfg.z_dim),
            feature_dim=int(cfg.manifold_feature_dim),
            hidden_dim=int(cfg.manifold_hidden_dim),
            num_classes=int(cfg.num_classes),
            dropout=float(cfg.dropout),
        )
        spatial_feat_dim = 500
        manifold_feat_dim = int(cfg.manifold_feature_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(spatial_feat_dim + manifold_feat_dim, int(cfg.fusion_hidden_dim)),
            nn.BatchNorm1d(int(cfg.fusion_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.fusion_hidden_dim), int(cfg.num_classes)),
        )

    def forward(self, raw_x: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial_logits, spatial_feat = self.spatial_backbone(raw_x, return_features=True)
        manifold_logits, manifold_feat = self.manifold_backbone(z, return_features=True)
        fusion_logits = self.fusion_head(torch.cat([spatial_feat, manifold_feat], dim=1))
        return {
            "logits": fusion_logits,
            "fusion_logits": fusion_logits,
            "spatial_logits": spatial_logits,
            "manifold_logits": manifold_logits,
            "spatial_features": spatial_feat,
            "manifold_features": manifold_feat,
        }
