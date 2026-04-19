from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.time_series_library_reference import (
    build_timesnet_classification_config,
    get_timesnet_model_class,
)


@dataclass
class TimesNetForwardOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    final_logit: torch.Tensor


class TimesNetAdapter(nn.Module):
    """TimesNet classification host exposed through the same E0/E1/E2 interface."""

    def __init__(
        self,
        *,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 32,
        d_ff: int = 64,
        e_layers: int = 2,
        top_k: int = 3,
        num_kernels: int = 4,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        base_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.num_classes = int(num_classes)
        self.config = build_timesnet_classification_config(
            seq_len=int(seq_len),
            enc_in=int(in_channels),
            num_class=int(num_classes),
            d_model=int(d_model),
            d_ff=int(d_ff),
            e_layers=int(e_layers),
            top_k=int(top_k),
            num_kernels=int(num_kernels),
            dropout=float(dropout),
            embed=str(embed),
            freq=str(freq),
        )
        model_cls = get_timesnet_model_class()
        self.base_model = base_model or model_cls(self.config)
        self.feature_dim = int(self.base_model.projection.in_features)

    def forward_features(self, x: torch.Tensor) -> TimesNetForwardOutputs:
        if x.ndim != 3:
            raise ValueError(f"TimesNetAdapter expects [B,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"TimesNetAdapter expected channel dim {self.in_channels}, got {tuple(x.shape)}"
            )
        if int(x.shape[2]) != self.seq_len:
            raise ValueError(
                f"TimesNetAdapter expected seq_len {self.seq_len}, got {tuple(x.shape)}"
            )

        x_enc = x.float().transpose(1, 2)  # [B,T,C]
        padding_mask = torch.ones(
            x_enc.shape[0],
            x_enc.shape[1],
            dtype=x_enc.dtype,
            device=x_enc.device,
        )
        enc_out = self.base_model.enc_embedding(x_enc, None)
        for i in range(self.base_model.layer):
            enc_out = self.base_model.layer_norm(self.base_model.model[i](enc_out))
        output = self.base_model.act(enc_out)
        output = self.base_model.dropout(output)
        output = output * padding_mask.unsqueeze(-1)
        latent = output.reshape(output.shape[0], -1)
        base_logit = self.base_model.projection(latent)
        sequence_features = output.transpose(1, 2).contiguous()
        return TimesNetForwardOutputs(
            sequence_features=sequence_features,
            latent=latent,
            base_logit=base_logit,
            final_logit=base_logit,
        )

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        outputs = self.forward_features(x)
        if return_features:
            return outputs
        return outputs.final_logit
