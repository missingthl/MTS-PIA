from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from utils.external_baseline_methods.base import ExternalAugResult, finite_stack, rng


def timevae_classwise_optional(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    latent_dim: int = 8,
    hidden_dim: int = 128,
    beta: float = 1.0,
    min_class_size: int = 4,
    device: str = "cpu",
) -> ExternalAugResult:
    """Classwise TimeVAE-style generator adapter."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rng(seed)
    torch.manual_seed(int(seed))
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    n, c, t = X_train.shape
    input_dim = int(c * t)
    hidden_i = max(16, int(hidden_dim))
    latent_i = max(2, int(latent_dim))
    device_t = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    class _DenseTimeVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_i),
                nn.ReLU(),
                nn.Linear(hidden_i, hidden_i),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_i, latent_i)
            self.logvar = nn.Linear(hidden_i, latent_i)
            self.decoder = nn.Sequential(
                nn.Linear(latent_i, hidden_i),
                nn.ReLU(),
                nn.Linear(hidden_i, hidden_i),
                nn.ReLU(),
                nn.Linear(hidden_i, input_dim),
            )

        def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.encoder(x_flat)
            mu = self.mu(h)
            logvar = torch.clamp(self.logvar(h), min=-8.0, max=8.0)
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
            return self.decoder(z), mu, logvar

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    class_success = 0
    class_attempts = 0
    generation_fail_count = 0
    skipped_classes = 0
    final_losses: List[float] = []

    for cls in sorted(int(x) for x in np.unique(y_train)):
        cls_idx = np.flatnonzero(y_train == cls)
        class_attempts += 1
        n_cls = int(cls_idx.shape[0])
        n_aug_cls = int(multiplier) * n_cls
        if n_cls < int(min_class_size) or n_aug_cls <= 0:
            skipped_classes += 1
            generation_fail_count += n_aug_cls
            continue

        x_cls = X_train[cls_idx]
        mean = x_cls.mean(axis=0, keepdims=True).astype(np.float32)
        std = x_cls.std(axis=0, keepdims=True).astype(np.float32)
        std = np.where(std < 1e-4, 1.0, std).astype(np.float32)
        x_norm = ((x_cls - mean) / std).reshape(n_cls, input_dim).astype(np.float32)
        tensor = torch.from_numpy(x_norm)
        gen = torch.Generator()
        gen.manual_seed(int(seed) + 1009 * int(cls))
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=max(1, min(int(batch_size), n_cls)),
            shuffle=True,
            generator=gen,
        )
        model = _DenseTimeVAE().to(device_t)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        last_loss = float("nan")
        model.train()
        for _ in range(max(1, int(epochs))):
            epoch_losses: List[float] = []
            for (xb,) in loader:
                xb = xb.to(device_t)
                recon, mu, logvar = model(xb)
                recon_loss = torch.mean((recon - xb) ** 2)
                kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + float(beta) * kld
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.detach().cpu()))
            if epoch_losses:
                last_loss = float(np.mean(epoch_losses))
        final_losses.append(last_loss)
        model.eval()
        try:
            with torch.no_grad():
                z = torch.randn(n_aug_cls, latent_i, device=device_t)
                x_gen = model.decode(z).cpu().numpy().astype(np.float32)
            x_gen = x_gen.reshape(n_aug_cls, c, t)
            x_gen = x_gen * std + mean
            x_gen = np.nan_to_num(x_gen, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            X_out.extend([x for x in x_gen])
            y_out.extend([cls] * n_aug_cls)
            class_success += 1
        except Exception:
            generation_fail_count += n_aug_cls

    if X_out:
        X_aug = finite_stack(X_out)
        y_aug = np.asarray(y_out, dtype=np.int64)
    else:
        X_aug = np.empty((0, c, t), dtype=np.float32)
        y_aug = np.empty((0,), dtype=np.int64)

    success_rate = float(class_success) / max(float(class_attempts), 1.0)
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=y_aug,
        source_space="generative_model",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="classwise_timevae_style_pytorch_cleanroom",
        warning_count=int(skipped_classes),
        fallback_count=int(generation_fail_count),
        meta={
            "target_aug_ratio": float(multiplier),
            "actual_aug_ratio": float(X_aug.shape[0]) / max(float(n), 1.0),
            "class_fit_success_rate": float(success_rate),
            "generation_fail_count": float(generation_fail_count),
            "timevae_skipped_classes": float(skipped_classes),
            "timevae_latent_dim": float(latent_i),
            "timevae_hidden_dim": float(hidden_i),
            "timevae_epochs": float(epochs),
            "timevae_beta": float(beta),
            "timevae_min_class_size": float(min_class_size),
            "timevae_final_loss_mean": float(np.nanmean(final_losses)) if final_losses else float("nan"),
            "timevae_cleanroom_adapter": 1.0,
            "timevae_official_keras_pipeline": 0.0,
        },
    )
