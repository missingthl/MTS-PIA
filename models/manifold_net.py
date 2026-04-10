from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from torch import nn

from models.encoders import SpatialGraphEncoder


def _group_count(channels: int) -> int:
    for g in (16, 8, 4, 2, 1):
        if channels % g == 0 and g <= channels:
            return g
    return 1


class SpatialGridEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        g1 = _group_count(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g1, in_channels)
        self.act2 = nn.GELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pool2(x)
        return self.head(x)


class ManifoldNetwork(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        embed_dim: int,
        spatial_encoder: str = "grid",
        *,
        graph_use_max: bool = False,
        graph_num_layers: int = 1,
        graph_num_heads: int = 2,
        graph_ffn_mult: float = 2.0,
        graph_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion_dim = int(fusion_dim)
        self.embed_dim = int(embed_dim)
        self.spatial_encoder = (spatial_encoder or "grid").lower()
        self.graph_num_layers = int(graph_num_layers)
        self.graph_num_heads = int(graph_num_heads)
        self.graph_ffn_mult = float(graph_ffn_mult)
        self.graph_dropout = float(graph_dropout)
        self.graph_use_max = bool(graph_use_max)

        g1 = _group_count(self.fusion_dim)
        self.band_fusion = nn.Sequential(
            nn.Conv2d(5, self.fusion_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g1, self.fusion_dim),
            nn.GELU(),
        )
        if self.spatial_encoder == "grid":
            self.spatial_encoder_module = SpatialGridEncoder(self.fusion_dim, self.embed_dim)
        elif self.spatial_encoder == "graph":
            self.spatial_encoder_module = SpatialGraphEncoder(
                self.fusion_dim,
                self.embed_dim,
                num_layers=self.graph_num_layers,
                nhead=self.graph_num_heads,
                ffn_mult=self.graph_ffn_mult,
                dropout=self.graph_dropout,
                use_max=self.graph_use_max,
            )
        else:
            raise ValueError(
                f"spatial_encoder must be 'grid' or 'graph', got {self.spatial_encoder}"
            )

        self.total_params = sum(p.numel() for p in self.parameters())
        self.spatial_encoder_params = sum(p.numel() for p in self.spatial_encoder_module.parameters())
        print(
            f"[manifold_net] total_params={self.total_params} "
            f"spatial_encoder_params={self.spatial_encoder_params} "
            f"encoder={self.spatial_encoder}",
            flush=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        return_taps: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5D input [B,T,5,62,62], got {x.shape}")
        if tuple(x.shape[-3:]) != (5, 62, 62):
            raise ValueError(f"expected last dims (5,62,62), got {x.shape[-3:]}")

        b, t, _, _, _ = x.shape
        x = x.reshape(b * t, 5, 62, 62)
        band_map = self.band_fusion(x)
        band_vec = band_map.mean(dim=(2, 3)).reshape(b, t, self.fusion_dim)
        spatial = self.spatial_encoder_module(band_map)
        spatial = spatial.reshape(b, t, self.embed_dim)

        if src_key_padding_mask is not None:
            feats = self.masked_mean(spatial, src_key_padding_mask, dim=1, debug=debug)
        else:
            feats = spatial.mean(dim=1)

        if not return_taps:
            return feats
        taps = {
            "z_band": band_vec,
            "z_spatial": spatial,
            "z_pooled": feats,
        }
        return feats, taps

    @staticmethod
    def masked_mean(
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        dim: int = 1,
        debug: bool = False,
    ) -> torch.Tensor:
        mask = padding_mask.bool()
        valid = (~mask).unsqueeze(-1).type_as(x)
        valid_counts = (~mask).sum(dim=dim)
        denom = valid_counts.unsqueeze(-1).clamp(min=1.0)
        out = (x * valid).sum(dim=dim) / denom

        if debug:
            padding_true = int(mask.sum().item())
            if padding_true == 0:
                baseline = x.mean(dim=dim)
                diff = float((out - baseline).abs().max().item())
                print(
                    f"[manifold_net][mask] padding_true=0 masked_mean_diff={diff:.6e}",
                    flush=True,
                )
                if diff > 1e-6:
                    print(
                        "[manifold_net][mask] WARNING: masked_mean mismatch when padding_true=0",
                        flush=True,
                    )
            else:
                counts = valid_counts.detach().float().cpu().numpy()
                min_v = int(counts.min()) if counts.size else 0
                p50 = float(np.percentile(counts, 50)) if counts.size else 0.0
                p95 = float(np.percentile(counts, 95)) if counts.size else 0.0
                max_v = int(counts.max()) if counts.size else 0
                print(
                    "[manifold_net][mask] "
                    f"valid_counts_stats={{'min':{min_v},'p50':{p50:.3f},'p95':{p95:.3f},'max':{max_v}}}",
                    flush=True,
                )
        return out
