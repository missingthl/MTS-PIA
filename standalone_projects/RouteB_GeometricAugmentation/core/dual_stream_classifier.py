from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

# from models.raw_dcnet_adapter import RawDCNetTemporalAdapter  # Pruning Phase 2 dependency


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
        raise NotImplementedError("SpatialOnlyClassifier depends on models.RawDCNetTemporalAdapter which was pruned for standalone Route B.")

    def forward(self, raw_x: torch.Tensor, z: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


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
        raise NotImplementedError("DualStreamClassifier depends on models.RawDCNetTemporalAdapter which was pruned for standalone Route B.")

    def forward(self, raw_x: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
