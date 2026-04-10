from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.layers import StructureAwareProjector


def _pick_nhead(embed_dim: int, preferred: int = 4) -> int:
    for nhead in (preferred, 2, 1):
        if embed_dim % nhead == 0:
            return nhead
    return 1


class SpatialGraphEncoder(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        embed_dim: int,
        *,
        num_layers: int = 2,
        nhead: int = 4,
        ffn_mult: float = 2.0,
        dropout: float = 0.1,
        use_max: bool = False,
    ) -> None:
        super().__init__()
        self.projector = StructureAwareProjector(use_max=use_max)
        node_in_dim = (3 if use_max else 2) * int(fusion_dim)
        self.input_proj = nn.Linear(node_in_dim, int(embed_dim))
        nhead = _pick_nhead(int(embed_dim), preferred=int(nhead))
        dim_ff = max(int(embed_dim * float(ffn_mult)), int(embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=nhead,
            dropout=float(dropout),
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        nodes = self.projector(feat_map)
        nodes = self.input_proj(nodes)
        nodes = self.encoder(nodes)
        pooled = nodes.mean(dim=1)
        return self.norm(pooled)
