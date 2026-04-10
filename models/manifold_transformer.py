from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ManifoldTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 9765,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if ff_dim is None:
            ff_dim = model_dim * 4

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, model_dim)) if max_len > 0 else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        if self.pos_embed is not None:
            if x.size(1) > self.pos_embed.size(1):
                raise ValueError("sequence length exceeds max_len for pos_embed")
            x = x + self.pos_embed[:, : x.size(1), :]
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
