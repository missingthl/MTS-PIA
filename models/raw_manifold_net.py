import math
from typing import Dict, Optional

import torch
from torch import nn


def _vec_utri(mat: torch.Tensor) -> torch.Tensor:
    idx = torch.triu_indices(mat.shape[-1], mat.shape[-1], device=mat.device)
    return mat[:, idx[0], idx[1]]


class RawCovTSMNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 62,
        conv_channels: int = 62,
        num_classes: int = 3,
        spd_eps: float = 1e-3,
        logmap_eps: float = 1e-6,
        raw_norm_mode: str = "none",
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_classes = num_classes
        self.spd_eps = float(spd_eps)
        self.logmap_eps = float(logmap_eps)
        self.raw_norm_mode = str(raw_norm_mode or "none").lower()

        self.conv = nn.Conv1d(in_channels, conv_channels, kernel_size=1, bias=True)
        self.instancenorm = nn.InstanceNorm1d(conv_channels, affine=True)

        vec_dim = conv_channels * (conv_channels + 1) // 2
        self.tsm_proj = nn.Linear(vec_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def _stats_per_channel(self, x: torch.Tensor) -> Dict[str, list]:
        x_det = x.detach()
        mean = x_det.mean(dim=(0, 2))
        std = x_det.std(dim=(0, 2))
        minv = x_det.amin(dim=(0, 2))
        maxv = x_det.amax(dim=(0, 2))
        return {
            "mean": mean.cpu().tolist(),
            "std": std.cpu().tolist(),
            "min": minv.cpu().tolist(),
            "max": maxv.cpu().tolist(),
        }

    def _nan_inf_flags(self, x: torch.Tensor) -> Dict[str, bool]:
        return {
            "has_nan": bool(torch.isnan(x).any().item()),
            "has_inf": bool(torch.isinf(x).any().item()),
        }

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - x.mean(dim=2, keepdim=True)
        denom = max(1, x.shape[2] - 1)
        cov = torch.matmul(x_c, x_c.transpose(1, 2)) / float(denom)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        return cov

    def _logmap(self, cov: torch.Tensor) -> torch.Tensor:
        vals, vecs = torch.linalg.eigh(cov)
        vals = torch.clamp(vals, min=self.logmap_eps)
        log_vals = torch.log(vals)
        log_cov = vecs @ torch.diag_embed(log_vals) @ vecs.transpose(1, 2)
        return log_cov

    def forward(self, x: torch.Tensor, debug: bool = False):
        debug_out: Optional[Dict[str, object]] = None

        if debug:
            debug_out = {
                "x_in": self._stats_per_channel(x),
                "nan_inf_x_in": self._nan_inf_flags(x),
            }

        x_conv = self.conv(x)
        if self.raw_norm_mode == "instancenorm":
            x_conv = self.instancenorm(x_conv)

        if debug:
            near_zero = (x_conv.detach().abs() < 1e-6).float().mean().item()
            debug_out["x_conv"] = self._stats_per_channel(x_conv)
            debug_out["x_conv_near_zero_pct"] = float(near_zero)
            debug_out["nan_inf_x_conv"] = self._nan_inf_flags(x_conv)

        x_conv = x_conv.double()
        cov_pre = self._covariance(x_conv)
        cov_sym_err = torch.norm(cov_pre - cov_pre.transpose(1, 2), dim=(1, 2)) / (
            torch.norm(cov_pre, dim=(1, 2)) + 1e-12
        )
        trace_pre = torch.diagonal(cov_pre, dim1=1, dim2=2).sum(dim=1)

        eig_pre = torch.linalg.eigvalsh(cov_pre)
        eig_pre_flat = eig_pre.reshape(-1)
        min_eig_pre = float(eig_pre.min().item())
        cond_pre = (eig_pre[:, -1] / eig_pre[:, 0].clamp(min=1e-12))

        cov_post = cov_pre + torch.eye(cov_pre.shape[1], device=cov_pre.device).double() * self.spd_eps
        cov_post = 0.5 * (cov_post + cov_post.transpose(1, 2))
        eig_post = torch.linalg.eigvalsh(cov_post)
        eig_post_flat = eig_post.reshape(-1)
        min_eig_post = float(eig_post.min().item())
        cond_post = (eig_post[:, -1] / eig_post[:, 0].clamp(min=1e-12))

        if debug:
            debug_out["cov_pre_eps"] = {
                "symmetry_error_mean": float(cov_sym_err.mean().item()),
                "trace_mean": float(trace_pre.mean().item()),
            }
            debug_out["eig_pre_eps"] = {
                "p01": float(torch.quantile(eig_pre_flat, 0.01).item()),
                "p05": float(torch.quantile(eig_pre_flat, 0.05).item()),
                "p50": float(torch.quantile(eig_pre_flat, 0.50).item()),
                "p95": float(torch.quantile(eig_pre_flat, 0.95).item()),
                "p99": float(torch.quantile(eig_pre_flat, 0.99).item()),
                "min": min_eig_pre,
            }
            debug_out["cov_post_eps"] = {
                "symmetry_error_mean": float(
                    torch.norm(cov_post - cov_post.transpose(1, 2), dim=(1, 2)).mean().item()
                ),
                "trace_mean": float(torch.diagonal(cov_post, dim1=1, dim2=2).sum(dim=1).mean().item()),
            }
            debug_out["eig_post_eps"] = {
                "p01": float(torch.quantile(eig_post_flat, 0.01).item()),
                "p05": float(torch.quantile(eig_post_flat, 0.05).item()),
                "p50": float(torch.quantile(eig_post_flat, 0.50).item()),
                "p95": float(torch.quantile(eig_post_flat, 0.95).item()),
                "p99": float(torch.quantile(eig_post_flat, 0.99).item()),
                "min": min_eig_post,
            }
            debug_out["cond_pre_eps_p95"] = float(torch.quantile(cond_pre, 0.95).item())
            debug_out["cond_post_eps_p95"] = float(torch.quantile(cond_post, 0.95).item())
            debug_out["nan_inf_cov_pre"] = self._nan_inf_flags(cov_pre)
            debug_out["nan_inf_cov_post"] = self._nan_inf_flags(cov_post)
            debug_out["nan_inf_eig_pre"] = {
                "has_nan": bool(torch.isnan(eig_pre).any().item()),
                "has_inf": bool(torch.isinf(eig_pre).any().item()),
            }
            debug_out["nan_inf_eig_post"] = {
                "has_nan": bool(torch.isnan(eig_post).any().item()),
                "has_inf": bool(torch.isinf(eig_post).any().item()),
            }

        log_cov = self._logmap(cov_post)
        vec = _vec_utri(log_cov)
        vec = vec.float()
        feat = torch.relu(self.tsm_proj(vec))
        logits = self.head(feat)

        if debug:
            return logits, debug_out
        return logits
