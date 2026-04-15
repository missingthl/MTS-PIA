from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

from models.local_closed_form_residual_head import LocalClosedFormResidualHead

@dataclass
class MiniRocketDLCRForwardOutputs:
    latent: torch.Tensor
    base_logit: torch.Tensor
    local_closed_form_logit: torch.Tensor
    readout_gate: torch.Tensor | None
    final_logit: torch.Tensor
    beta: torch.Tensor

class MiniRocketDLCRAdapter(nn.Module):
    """
    Adapter for MiniRocket features (frozen) with a Linear Base Head and DLCR Local Head.
    
    Architecture:
    Input Z (MiniRocket Features, e.g. 9996 dim)
    -> Base Head: nn.Linear(9996, num_classes)
    -> Local Head: LocalClosedFormResidualHead(feature_dim=9996)
    -> Fusion: Final = Base + Beta * Local
    """
    def __init__(
        self,
        *,
        feature_dim: int = 9996,
        num_classes: int,
        prototypes_per_class: int = 4,
        routing_temperature: float = 1.0,
        class_prior_temperature: float | None = None,
        subproto_temperature: float | None = None,
        ridge: float = 1e-2,
        ridge_mode: str = "fixed",
        ridge_trace_eps: float = 1e-8,
        solve_mode: str = "ridge_solve",
        pinv_rcond: float = 1e-4,
        input_norm_mode: str = "none",
        input_norm_eps: float = 1e-8,
        enable_probe: bool = False,
        init_beta: float = 0.1,
        support_mode: str = "same_only",
        prototype_aggregation: str = "pooled",
        prototype_geometry_mode: str = "flat",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        
        # Base classifier head (replaces the backbone's built-in head)
        self.base_head = nn.Linear(self.feature_dim, self.num_classes)
        
        # Local DLCR head
        self.local_head = LocalClosedFormResidualHead(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            prototypes_per_class=int(prototypes_per_class),
            routing_temperature=float(routing_temperature),
            class_prior_temperature=class_prior_temperature,
            subproto_temperature=subproto_temperature,
            ridge=float(ridge),
            ridge_mode=str(ridge_mode),
            ridge_trace_eps=float(ridge_trace_eps),
            solve_mode=str(solve_mode),
            pinv_rcond=float(pinv_rcond),
            input_norm_mode=str(input_norm_mode),
            input_norm_eps=float(input_norm_eps),
            enable_probe=bool(enable_probe),
            support_mode=str(support_mode),
            prototype_aggregation=str(prototype_aggregation),
            prototype_geometry_mode=str(prototype_geometry_mode),
        )
        
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(
        self,
        z: torch.Tensor,
        *,
        fusion_alpha: float = 1.0,
        return_features: bool = False,
    ):
        # z is already extracted and detached by the runner
        base_logit = self.base_head(z)
        local_closed_form_logit = self.local_head(z)
        
        alpha = torch.tensor(float(fusion_alpha), dtype=base_logit.dtype, device=base_logit.device)
        
        # We don't implement readout_gate for MiniRocket first-round to keep it simple,
        # but the interface supports it if we want to add consistency gating later.
        final_logit = base_logit + alpha * self.beta * local_closed_form_logit
        
        if return_features:
            return MiniRocketDLCRForwardOutputs(
                latent=z,
                base_logit=base_logit,
                local_closed_form_logit=local_closed_form_logit,
                readout_gate=None,
                final_logit=final_logit,
                beta=self.beta,
            )
        return final_logit
