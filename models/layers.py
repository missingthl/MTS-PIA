from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class StructureAwareProjector(nn.Module):
    def __init__(self, use_max: bool = False) -> None:
        super().__init__()
        self.use_max = bool(use_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input [N,D,62,62], got {x.shape}")
        if x.shape[-1] != 62 or x.shape[-2] != 62:
            raise ValueError(f"expected spatial dims 62x62, got {x.shape[-2:]}")

        row_mean = x.mean(dim=-1)
        diag = torch.diagonal(x, dim1=-2, dim2=-1)
        parts = [row_mean, diag]
        if self.use_max:
            row_max = x.max(dim=-1).values
            parts.append(row_max)

        nodes = torch.cat(parts, dim=1)
        nodes = nodes.permute(0, 2, 1)
        expected_dim = (3 if self.use_max else 2) * x.shape[1]
        if nodes.shape[1] != 62 or nodes.shape[2] != expected_dim:
            raise ValueError(
                f"unexpected nodes shape {nodes.shape}, expected (*,62,{expected_dim})"
            )
        return nodes
