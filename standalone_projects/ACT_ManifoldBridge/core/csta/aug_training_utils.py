from __future__ import annotations
import math
import numpy as np
import torch
from core.bridge import batched_logvec_to_spd, bridge_batch


class ManifoldAugDataLoader:
    """
    Transparent GPU-accelerated augmentation wrapper for DataLoader.

    Intercepts raw (x_raw, sigma_orig, z_cand, y_cand) batches from the 
    underlying DataLoader and performs batched whitening-coloring on the GPU.
    """
    def __init__(self, dataloader: torch.utils.data.DataLoader, device: str | torch.device, mean_log: torch.Tensor | np.ndarray):
        self.dataloader = dataloader
        self.device = device
        if isinstance(mean_log, torch.Tensor):
            self.mean_log = mean_log.to(device)
        else:
            self.mean_log = torch.from_numpy(mean_log).to(device)

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        for x_raw, sigma_orig, z_cand, y_cand in self.dataloader:
            # Transfer raw components to GPU
            x_raw = x_raw.to(self.device, non_blocking=True)
            sigma_orig = sigma_orig.to(self.device, non_blocking=True)
            z_cand = z_cand.to(self.device, non_blocking=True)
            y_cand = y_cand.to(self.device, non_blocking=True)

            # 1. Construct target SPD batch on GPU
            sigma_aug = batched_logvec_to_spd(z_cand, self.mean_log)

            # 2. Execute batched bridge on GPU
            x_aug = bridge_batch(x_raw, sigma_orig, sigma_aug)

            yield x_aug, y_cand


class TauScheduler:
    """
    Cosine-annealing temperature scheduler for augmentation soft-gating.

    - Exploration phase: high temperature, permissive weighting.
    - Annealing phase: cosine decay from ``tau_max`` to ``tau_min``.
    """

    def __init__(
        self,
        total_epochs: int,
        tau_max: float = 2.0,
        tau_min: float = 0.1,
        warmup_ratio: float = 0.3,
    ) -> None:
        self.total_epochs = max(1, int(total_epochs))
        self.tau_max = float(tau_max)
        self.tau_min = float(tau_min)
        self.warmup_epochs = int(self.total_epochs * max(0.0, min(1.0, float(warmup_ratio))))

    def get_tau(self, epoch: int) -> float:
        if int(epoch) < self.warmup_epochs:
            return self.tau_max
        anneal_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, float(epoch - self.warmup_epochs) / float(anneal_epochs))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.tau_min + (self.tau_max - self.tau_min) * cosine_val)


def focal_margin_weight(
    margin: torch.Tensor,
    tau: float,
    low_clip: float = -5.0,
    high_clip: float = 5.0,
    easy_floor: float = 0.1,
) -> torch.Tensor:
    """
    U-shaped focal margin weighting for augmented samples.

    Boundary-adjacent samples receive higher weight, very easy samples are
    downweighted, and clearly wrong / noisy samples are suppressed.
    """
    tau_val = max(float(tau), 1e-6)
    m = margin.clamp(float(low_clip), float(high_clip))
    hard_weight = torch.sigmoid((m - float(low_clip) / 2.0) / tau_val)
    easy_penalty = torch.sigmoid((m - float(high_clip) / 2.0) / tau_val)
    w = hard_weight * (1.0 - (1.0 - float(easy_floor)) * easy_penalty)
    return w.clamp(0.0, 1.0).detach()
