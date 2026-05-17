from __future__ import annotations

import time
from typing import List

import numpy as np
import torch

from utils.external_baseline_methods.base import ExternalAugResult, finite_stack, rng


def timegan_classwise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 24,
    latent_dim: int = 24,
    min_class_size: int = 4,
    device: str = "cpu",
) -> ExternalAugResult:
    """Classwise TimeGAN-style generator adapter.

    This is a compact PyTorch adapter inspired by TimeGAN's embedding,
    recovery, generator, supervisor, and discriminator components.  It is
    designed for the ACT/CSTA external baseline matrix and does not claim
    line-by-line parity with the original TensorFlow author repository.
    """

    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rng(seed)
    torch.manual_seed(int(seed))
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    n, c, t = X_train.shape
    hidden_i = max(4, int(hidden_dim))
    latent_i = max(4, int(latent_dim))
    device_t = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    class _Embedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(input_size=c, hidden_size=hidden_i, batch_first=True)

        def forward(self, x_tc: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(x_tc)
            return h

    class _Recovery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(input_size=hidden_i, hidden_size=hidden_i, batch_first=True)
            self.proj = nn.Linear(hidden_i, c)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            z, _ = self.rnn(h)
            return self.proj(z)

    class _Generator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(input_size=latent_i, hidden_size=hidden_i, batch_first=True)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(z)
            return h

    class _Supervisor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(input_size=hidden_i, hidden_size=hidden_i, batch_first=True)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            s, _ = self.rnn(h)
            return s

    class _Discriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(input_size=hidden_i, hidden_size=hidden_i, batch_first=True)
            self.proj = nn.Linear(hidden_i, 1)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            z, _ = self.rnn(h)
            return self.proj(z[:, -1, :]).squeeze(-1)

    def _sample_noise(num: int) -> torch.Tensor:
        return torch.randn(int(num), int(t), latent_i, device=device_t)

    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    class_success = 0
    class_attempts = 0
    skipped_classes = 0
    generation_fail_count = 0
    auto_losses: List[float] = []
    gen_losses: List[float] = []
    disc_losses: List[float] = []
    generator_fit_sec = 0.0
    sample_gen_sec = 0.0

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

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
        # GRU models operate on [N, T, C].
        x_norm = ((x_cls - mean) / std).transpose(0, 2, 1).astype(np.float32)
        tensor = torch.from_numpy(x_norm)
        gen = torch.Generator()
        gen.manual_seed(int(seed) + 2027 * int(cls))
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=max(1, min(int(batch_size), n_cls)),
            shuffle=True,
            generator=gen,
        )

        embedder = _Embedder().to(device_t)
        recovery = _Recovery().to(device_t)
        generator = _Generator().to(device_t)
        supervisor = _Supervisor().to(device_t)
        discriminator = _Discriminator().to(device_t)

        opt_er = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=float(lr))
        opt_gs = torch.optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=float(lr))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(lr))

        last_auto = float("nan")
        last_gen = float("nan")
        last_disc = float("nan")

        t_fit0 = time.perf_counter()
        # Keep the adapter intentionally compact: each epoch performs an
        # autoencoder update, generator/supervisor update, and discriminator
        # update.  This is enough for cost-utility stress testing without
        # turning the baseline into a fragile long-running GAN project.
        for _ in range(max(1, int(epochs))):
            auto_epoch: List[float] = []
            gen_epoch: List[float] = []
            disc_epoch: List[float] = []
            for (xb_cpu,) in loader:
                xb = xb_cpu.to(device_t)
                bs = int(xb.shape[0])

                # Embedding / recovery reconstruction.
                h_real = embedder(xb)
                x_tilde = recovery(h_real)
                auto_loss = mse(x_tilde, xb)
                opt_er.zero_grad(set_to_none=True)
                auto_loss.backward()
                opt_er.step()
                auto_epoch.append(float(auto_loss.detach().cpu()))

                # Generator and supervisor.  Supervised temporal loss uses the
                # real embedding path; adversarial loss uses generated states.
                with torch.no_grad():
                    h_real_detached = embedder(xb)
                z = _sample_noise(bs)
                h_fake = generator(z)
                h_fake_sup = supervisor(h_fake)
                y_fake = discriminator(h_fake_sup)
                adv_loss = bce(y_fake, torch.ones_like(y_fake))
                if h_real_detached.shape[1] > 1:
                    sup_loss = mse(supervisor(h_real_detached[:, :-1, :]), h_real_detached[:, 1:, :])
                else:
                    sup_loss = torch.zeros((), device=device_t)
                x_fake = recovery(h_fake_sup)
                moment_loss = torch.abs(x_fake.mean(dim=(0, 1)) - xb.mean(dim=(0, 1))).mean()
                moment_loss = moment_loss + torch.abs(x_fake.std(dim=(0, 1)) - xb.std(dim=(0, 1))).mean()
                gen_loss = adv_loss + 10.0 * sup_loss + 5.0 * moment_loss
                opt_gs.zero_grad(set_to_none=True)
                gen_loss.backward()
                opt_gs.step()
                gen_epoch.append(float(gen_loss.detach().cpu()))

                # Discriminator with detached embeddings.
                with torch.no_grad():
                    h_real_detached = embedder(xb)
                    h_fake_detached = supervisor(generator(_sample_noise(bs)))
                y_real = discriminator(h_real_detached)
                y_fake_detached = discriminator(h_fake_detached)
                disc_loss = bce(y_real, torch.ones_like(y_real)) + bce(y_fake_detached, torch.zeros_like(y_fake_detached))
                opt_d.zero_grad(set_to_none=True)
                disc_loss.backward()
                opt_d.step()
                disc_epoch.append(float(disc_loss.detach().cpu()))

            if auto_epoch:
                last_auto = float(np.mean(auto_epoch))
            if gen_epoch:
                last_gen = float(np.mean(gen_epoch))
            if disc_epoch:
                last_disc = float(np.mean(disc_epoch))
        generator_fit_sec += time.perf_counter() - t_fit0

        auto_losses.append(last_auto)
        gen_losses.append(last_gen)
        disc_losses.append(last_disc)

        t_sample0 = time.perf_counter()
        try:
            embedder.eval()
            recovery.eval()
            generator.eval()
            supervisor.eval()
            with torch.no_grad():
                z = _sample_noise(n_aug_cls)
                h = supervisor(generator(z))
                x_gen_tc = recovery(h).cpu().numpy().astype(np.float32)
            x_gen = x_gen_tc.transpose(0, 2, 1).astype(np.float32)
            x_gen = x_gen * std + mean
            x_gen = np.nan_to_num(x_gen, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            X_out.extend([x for x in x_gen])
            y_out.extend([cls] * n_aug_cls)
            class_success += 1
        except Exception:
            generation_fail_count += n_aug_cls
        finally:
            sample_gen_sec += time.perf_counter() - t_sample0

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
        selection_rule="classwise_timegan_style_pytorch_cleanroom",
        warning_count=int(skipped_classes),
        fallback_count=int(generation_fail_count),
        meta={
            "target_aug_ratio": float(multiplier),
            "actual_aug_ratio": float(X_aug.shape[0]) / max(float(n), 1.0),
            "class_fit_success_rate": float(success_rate),
            "generation_fail_count": float(generation_fail_count),
            "timegan_skipped_classes": float(skipped_classes),
            "timegan_hidden_dim": float(hidden_i),
            "timegan_latent_dim": float(latent_i),
            "timegan_epochs": float(epochs),
            "timegan_final_auto_loss_mean": float(np.nanmean(auto_losses)) if auto_losses else float("nan"),
            "timegan_final_generator_loss_mean": float(np.nanmean(gen_losses)) if gen_losses else float("nan"),
            "timegan_final_discriminator_loss_mean": float(np.nanmean(disc_losses)) if disc_losses else float("nan"),
            "timegan_cleanroom_adapter": 1.0,
            "timegan_official_tensorflow_pipeline": 0.0,
            "generator_fit_sec": float(generator_fit_sec),
            "sample_gen_sec": float(sample_gen_sec),
            "aug_cost_sec": float(generator_fit_sec + sample_gen_sec),
        },
    )
