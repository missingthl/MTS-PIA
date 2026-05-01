from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_periods_from_train(
    X_train: np.ndarray,
    *,
    k: int = 5,
    min_period: int = 2,
) -> List[int]:
    """Infer dominant periods from train-only data.

    This mirrors the MPTSNet idea of using FFT amplitudes to choose periodic
    scales, while avoiding any dependency on the official repository data
    pipeline.  Input is project-native [N, C, T].
    """
    x = np.asarray(X_train, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected X_train with shape [N,C,T], got {x.shape}")
    t = int(x.shape[-1])
    if t <= 2:
        return [1]
    series_ntc = np.transpose(x, (0, 2, 1))
    avg = series_ntc.mean(axis=0).mean(axis=-1)
    spectrum = np.fft.rfft(avg)
    power = np.abs(spectrum)
    if power.shape[0] > 0:
        power[0] = 0.0
    periods: List[int] = []
    for freq_idx in np.argsort(power)[::-1]:
        freq_i = int(freq_idx)
        if freq_i <= 0:
            continue
        period = int(round(float(t) / float(freq_i)))
        period = max(int(min_period), min(period, t))
        if period not in periods:
            periods.append(period)
        if len(periods) >= int(k):
            break
    if not periods:
        periods = [max(int(min_period), min(t, t // 2 if t >= 4 else 1))]
    return periods


class _TokenEmbedding(nn.Module):
    def __init__(self, in_channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, D, T]
        return self.dropout(self.proj(x))


class _Inception1D(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, num_kernels: int = 4) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            [
                nn.Conv1d(channels, hidden_channels, kernel_size=2 * i + 1, padding=i)
                for i in range(int(num_kernels))
            ]
        )
        self.out = nn.Conv1d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([conv(x) for conv in self.kernels], dim=-1).mean(dim=-1)
        return self.out(F.gelu(feats))


class _PeriodicBlock(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        d_model: int,
        d_ff: int,
        periods: Sequence[int],
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.periods = [max(1, int(p)) for p in periods]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, int(n_heads)),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.time_transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(n_layers)))
        self.local_cnn = _Inception1D(d_model, d_ff)
        self.norm = nn.LayerNorm(d_model)

    def _periodic_local(self, x: torch.Tensor, period: int) -> torch.Tensor:
        # x: [B, D, T]
        b, d, t = x.shape
        period = max(1, min(int(period), t))
        if t % period != 0:
            length = ((t // period) + 1) * period
            pad = torch.zeros(b, d, length - t, dtype=x.dtype, device=x.device)
            x_pad = torch.cat([x, pad], dim=-1)
        else:
            length = t
            x_pad = x
        n_period = max(1, length // period)
        windows = x_pad.reshape(b, d, n_period, period).permute(0, 2, 1, 3).reshape(b * n_period, d, period)
        local = self.local_cnn(windows) + windows
        local = local.reshape(b, n_period, d, period).permute(0, 2, 1, 3).reshape(b, d, length)
        return local[:, :, :t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T]
        b, d, t = x.shape
        time_feat = self.time_transformer(x.transpose(1, 2)).transpose(1, 2)
        xf = torch.fft.rfft(x.transpose(1, 2), dim=1)
        period_features = []
        weights = []
        for period in self.periods:
            period_features.append(self._periodic_local(x, period))
            freq_idx = max(1, min(xf.shape[1] - 1, int(round(float(t) / float(max(period, 1))))))
            weights.append(torch.abs(xf[:, freq_idx]).mean(dim=-1))
        stacked = torch.stack(period_features, dim=-1)  # [B, D, T, K]
        w = torch.stack(weights, dim=-1)
        w = torch.softmax(w, dim=-1).view(b, 1, 1, -1)
        fused = torch.sum(stacked * w, dim=-1)
        out = fused + time_feat + x
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class MPTSNetClassifier(nn.Module):
    """Project-native MPTSNet backbone adapter.

    This follows the official MPTSNet design intent: FFT-selected multi-period
    scales, local CNN feature extraction on periodic segments, and transformer
    global dependency modeling.  It intentionally stays dependency-light and
    uses the ACT project [B,C,T] tensor convention.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        *,
        periods: Optional[Iterable[int]] = None,
        period_k: int = 5,
        d_model: Optional[int] = None,
        d_ff: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        transformer_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_model_i = int(d_model) if d_model is not None else max(32, min(128, int(in_channels) * 4))
        n_heads_i = max(1, min(int(n_heads), d_model_i))
        while d_model_i % n_heads_i != 0 and n_heads_i > 1:
            n_heads_i -= 1
        if periods is None:
            periods = [max(2, int(seq_len) // 2)]
        self.periods = [max(1, min(int(p), int(seq_len))) for p in periods][: max(1, int(period_k))]
        self.embedding = _TokenEmbedding(int(in_channels), d_model_i, float(dropout))
        self.blocks = nn.ModuleList(
            [
                _PeriodicBlock(
                    seq_len=int(seq_len),
                    d_model=d_model_i,
                    d_ff=int(d_ff),
                    periods=self.periods,
                    n_heads=n_heads_i,
                    n_layers=int(transformer_layers),
                    dropout=float(dropout),
                )
                for _ in range(max(1, int(e_layers)))
            ]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(d_model_i, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        means = x.mean(dim=-1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(F.gelu(h))
        pooled = h.mean(dim=-1)
        return self.head(pooled)
