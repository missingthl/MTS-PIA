from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass(frozen=True)
class TrajectoryModelConfig:
    z_dim: int
    num_classes: int
    gru_hidden_dim: int = 128
    dropout: float = 0.3


class StaticLinearClassifier(nn.Module):
    def __init__(self, cfg: TrajectoryModelConfig) -> None:
        super().__init__()
        self.classifier = nn.Linear(int(cfg.z_dim), int(cfg.num_classes))

    def forward(
        self,
        z_static: torch.Tensor,
        z_seq: torch.Tensor | None = None,
        seq_lengths: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.classifier(z_static)
        return {"logits": logits, "features": z_static}


class DynamicMeanPoolClassifier(nn.Module):
    def __init__(self, cfg: TrajectoryModelConfig) -> None:
        super().__init__()
        self.classifier = nn.Linear(int(cfg.z_dim), int(cfg.num_classes))

    def forward(
        self,
        z_static: torch.Tensor | None,
        z_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, steps, feat_dim = z_seq.shape
        mask = (
            torch.arange(int(steps), device=z_seq.device)[None, :]
            < seq_lengths.to(z_seq.device)[:, None]
        ).to(z_seq.dtype)
        denom = torch.clamp(seq_lengths.to(z_seq.device).unsqueeze(1).to(z_seq.dtype), min=1.0)
        pooled = (z_seq * mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.classifier(pooled)
        return {"logits": logits, "features": pooled}


class DynamicGRUClassifier(nn.Module):
    def __init__(self, cfg: TrajectoryModelConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(cfg.z_dim),
            hidden_size=int(cfg.gru_hidden_dim),
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.classifier = nn.Linear(int(cfg.gru_hidden_dim), int(cfg.num_classes))

    def forward(
        self,
        z_static: torch.Tensor | None,
        z_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            z_seq,
            lengths=seq_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _out, h_n = self.gru(packed)
        feat = self.dropout(h_n[-1])
        logits = self.classifier(feat)
        return {"logits": logits, "features": feat}
