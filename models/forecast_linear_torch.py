from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _moving_avg(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    pad = (kernel_size - 1) // 2
    # x: [B, L, C] -> [B, C, L]
    xc = x.transpose(1, 2)
    xc = torch.nn.functional.pad(xc, (pad, pad), mode="replicate")
    out = torch.nn.functional.avg_pool1d(xc, kernel_size=kernel_size, stride=1)
    return out.transpose(1, 2)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = _moving_avg(x, self.kernel_size)
        seasonal = x - trend
        return seasonal, trend


@dataclass(frozen=True)
class ForecastLinearConfig:
    lookback: int
    horizon: int
    n_features: int
    target_idx: int
    individual: bool = False
    decomp_kernel_size: int = 25


class DLinearMS(nn.Module):
    def __init__(self, cfg: ForecastLinearConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decomp = SeriesDecomp(cfg.decomp_kernel_size)
        if cfg.individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(cfg.lookback, cfg.horizon) for _ in range(cfg.n_features)])
            self.linear_trend = nn.ModuleList([nn.Linear(cfg.lookback, cfg.horizon) for _ in range(cfg.n_features)])
        else:
            self.linear_seasonal = nn.Linear(cfg.lookback, cfg.horizon)
            self.linear_trend = nn.Linear(cfg.lookback, cfg.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.transpose(1, 2)  # [B, C, L]
        trend = trend.transpose(1, 2)        # [B, C, L]

        if self.cfg.individual:
            seasonal_out = []
            trend_out = []
            for i in range(self.cfg.n_features):
                seasonal_out.append(self.linear_seasonal[i](seasonal[:, i, :]))
                trend_out.append(self.linear_trend[i](trend[:, i, :]))
            seasonal_out = torch.stack(seasonal_out, dim=1)
            trend_out = torch.stack(trend_out, dim=1)
        else:
            bsz, n_feat, _ = seasonal.shape
            seasonal_out = self.linear_seasonal(seasonal.reshape(bsz * n_feat, self.cfg.lookback)).reshape(bsz, n_feat, self.cfg.horizon)
            trend_out = self.linear_trend(trend.reshape(bsz * n_feat, self.cfg.lookback)).reshape(bsz, n_feat, self.cfg.horizon)

        out = seasonal_out + trend_out  # [B, C, H]
        return out[:, self.cfg.target_idx, :]  # [B, H]


class NLinearMS(nn.Module):
    def __init__(self, cfg: ForecastLinearConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.individual:
            self.linear = nn.ModuleList([nn.Linear(cfg.lookback, cfg.horizon) for _ in range(cfg.n_features)])
        else:
            self.linear = nn.Linear(cfg.lookback, cfg.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        last = x[:, -1:, :]               # [B, 1, C]
        x_norm = x - last
        x_norm = x_norm.transpose(1, 2)   # [B, C, L]

        if self.cfg.individual:
            out = []
            for i in range(self.cfg.n_features):
                out.append(self.linear[i](x_norm[:, i, :]))
            out = torch.stack(out, dim=1)
        else:
            bsz, n_feat, _ = x_norm.shape
            out = self.linear(x_norm.reshape(bsz * n_feat, self.cfg.lookback)).reshape(bsz, n_feat, self.cfg.horizon)

        anchor = last[:, 0, self.cfg.target_idx].unsqueeze(1).expand(-1, self.cfg.horizon)
        return out[:, self.cfg.target_idx, :] + anchor
