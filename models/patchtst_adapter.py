from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.time_series_library_reference import (
    build_patchtst_classification_config,
    get_patchtst_model_class,
)


@dataclass
class PatchTSTForwardOutputs:
    sequence_features: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    final_logit: torch.Tensor


class PatchTSTAdapter(nn.Module):
    """PatchTST classification host exposed through the same E0/E1/E2 interface."""

    def __init__(
        self,
        *,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 3,
        n_heads: int = 8,
        factor: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        patch_len: int = 16,
        patch_stride: int = 8,
        base_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.num_classes = int(num_classes)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.config = build_patchtst_classification_config(
            seq_len=int(seq_len),
            enc_in=int(in_channels),
            num_class=int(num_classes),
            d_model=int(d_model),
            d_ff=int(d_ff),
            e_layers=int(e_layers),
            n_heads=int(n_heads),
            factor=int(factor),
            dropout=float(dropout),
            activation=str(activation),
        )
        model_cls = get_patchtst_model_class()
        self.base_model = base_model or model_cls(
            self.config,
            patch_len=int(patch_len),
            stride=int(patch_stride),
        )
        self.feature_dim = int(self.base_model.projection.in_features)

    def forward_features(self, x: torch.Tensor) -> PatchTSTForwardOutputs:
        if x.ndim != 3:
            raise ValueError(f"PatchTSTAdapter expects [B,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"PatchTSTAdapter expected channel dim {self.in_channels}, got {tuple(x.shape)}"
            )
        if int(x.shape[2]) != self.seq_len:
            raise ValueError(
                f"PatchTSTAdapter expected seq_len {self.seq_len}, got {tuple(x.shape)}"
            )

        x_enc = x.float().transpose(1, 2)  # [B,T,C]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_patch = x_enc.permute(0, 2, 1)  # [B,C,T]
        enc_out, n_vars = self.base_model.patch_embedding(x_patch)
        enc_out, _ = self.base_model.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B,C,d_model,patch_num]

        sequence_features = enc_out.reshape(enc_out.shape[0], -1, enc_out.shape[-1])
        latent = self.base_model.dropout(self.base_model.flatten(enc_out)).reshape(enc_out.shape[0], -1)
        base_logit = self.base_model.projection(latent)
        return PatchTSTForwardOutputs(
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
