"""External augmentation baselines for ACT/CSTA experiments.

This module intentionally owns offline/external augmentation logic so
``run_act_pilot.py`` can stay focused on CSTA/PIA itself.  For a searchable
method index, see ``utils/external_baseline_manifest.py`` and
``docs/EXTERNAL_BASELINES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from core.bridge import bridge_single, logvec_to_spd


@dataclass
class ExternalAugResult:
    X_aug: np.ndarray
    y_aug: Optional[np.ndarray] = None
    y_aug_soft: Optional[np.ndarray] = None
    source_space: str = "raw_time"
    label_mode: str = "hard"
    uses_external_library: bool = False
    library_name: str = ""
    budget_matched: bool = True
    selection_rule: str = ""
    warning_count: int = 0
    fallback_count: int = 0
    meta: Dict[str, float] = field(default_factory=dict)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _repeat_anchor_indices(n_train: int, multiplier: int) -> np.ndarray:
    return np.repeat(np.arange(int(n_train), dtype=np.int64), int(multiplier))


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((int(y.shape[0]), int(n_classes)), dtype=np.float32)
    out[np.arange(int(y.shape[0])), y.astype(np.int64)] = 1.0
    return out


def _resample_ct(x_ct: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample one multivariate series from [C, T] to [C, target_len]."""
    x_ct = np.asarray(x_ct, dtype=np.float32)
    c, t = int(x_ct.shape[0]), int(x_ct.shape[1])
    target_len = int(target_len)
    if t == target_len:
        return x_ct.astype(np.float32, copy=True)
    if t <= 1:
        return np.repeat(x_ct, target_len, axis=1).astype(np.float32)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, target_len)
    out = np.empty((c, target_len), dtype=np.float32)
    for ch in range(c):
        out[ch] = np.interp(dst, src, x_ct[ch]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# JobDA clean-room Time-Series Warping helpers
# ---------------------------------------------------------------------------


def _downsample_avg_stride2_ct(segment_ct: np.ndarray) -> np.ndarray:
    """Average-pool a [C, L] segment with stride 2."""
    segment_ct = np.asarray(segment_ct, dtype=np.float32)
    c, length = segment_ct.shape
    if length <= 1:
        return segment_ct.astype(np.float32, copy=True)
    out_len = int(np.ceil(length / 2.0))
    out = np.empty((c, out_len), dtype=np.float32)
    for i in range(out_len):
        start = 2 * i
        stop = min(start + 2, length)
        out[:, i] = np.mean(segment_ct[:, start:stop], axis=1)
    return out


def _upsample_insert_avg_ct(segment_ct: np.ndarray) -> np.ndarray:
    """Insert pairwise averages between adjacent samples of a [C, L] segment."""
    segment_ct = np.asarray(segment_ct, dtype=np.float32)
    c, length = segment_ct.shape
    if length <= 1:
        return segment_ct.astype(np.float32, copy=True)
    out = np.empty((c, 2 * length - 1), dtype=np.float32)
    out[:, 0::2] = segment_ct
    out[:, 1::2] = 0.5 * (segment_ct[:, :-1] + segment_ct[:, 1:])
    return out


def time_series_warping_cleanroom(x_ct: np.ndarray, n_subseq: int) -> np.ndarray:
    """Clean-room Time-Series Warping (TSW) transform for JobDA.

    Based on the JobDA paper description: split a time series into N continuous
    subsequences, alternately compress and expand subsequences with
    downsampling/upsampling, concatenate them, then restore the original length.
    This is not copied from official code; no confirmed official implementation
    was found.
    """
    x_ct = np.asarray(x_ct, dtype=np.float32)
    c, t = x_ct.shape
    n_subseq = max(1, min(int(n_subseq), max(1, t)))
    segments = np.array_split(x_ct, n_subseq, axis=1)
    warped: List[np.ndarray] = []
    for seg_idx, segment in enumerate(segments):
        if segment.shape[1] == 0:
            continue
        if seg_idx % 2 == 0:
            warped.append(_downsample_avg_stride2_ct(segment))
        else:
            warped.append(_upsample_insert_avg_ct(segment))
    if not warped:
        return x_ct.astype(np.float32, copy=True)
    stitched = np.concatenate(warped, axis=1)
    return _resample_ct(stitched, t).reshape(c, t).astype(np.float32)


def _class_to_indices(y_train: np.ndarray) -> Dict[int, np.ndarray]:
    y_train = np.asarray(y_train, dtype=np.int64)
    return {int(c): np.flatnonzero(y_train == c) for c in np.unique(y_train)}


def _finite_stack(xs: List[np.ndarray]) -> np.ndarray:
    return np.nan_to_num(np.stack(xs, axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# DTW and guided-warping helpers shared by SPAWNER/RGW/DGW
# ---------------------------------------------------------------------------


def _dtw_path_tc(
    prototype_tc: np.ndarray,
    sample_tc: np.ndarray,
    *,
    slope_constraint: str = "symmetric",
    window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Small multivariate DTW path helper for guided-warping baselines.

    Inputs are [T, C].  The returned path follows the convention used by
    guided warping: prototype indices first, sample indices second.
    """
    prototype_tc = np.asarray(prototype_tc, dtype=np.float64)
    sample_tc = np.asarray(sample_tc, dtype=np.float64)
    p = int(prototype_tc.shape[0])
    s = int(sample_tc.shape[0])
    if p <= 0 or s <= 0:
        raise ValueError("DTW inputs must be non-empty.")
    if slope_constraint not in {"symmetric", "asymmetric"}:
        raise ValueError(f"Unsupported slope_constraint={slope_constraint!r}")
    if window is None:
        window_i = max(p, s)
    else:
        window_i = max(1, int(window))

    cost = np.full((p, s), np.inf, dtype=np.float64)
    for i in range(p):
        start = max(0, i - window_i)
        stop = min(s, i + window_i + 1)
        if start < stop:
            cost[i, start:stop] = np.linalg.norm(sample_tc[start:stop] - prototype_tc[i], axis=1)

    dtw = np.full((p + 1, s + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    if slope_constraint == "symmetric":
        for i in range(1, p + 1):
            for j in range(max(1, i - window_i), min(s, i + window_i) + 1):
                dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1])
    else:
        for i in range(1, p + 1):
            if i <= window_i + 1:
                dtw[i, 1] = cost[i - 1, 0] + min(dtw[i - 1, 0], dtw[i - 1, 1])
            for j in range(max(2, i - window_i), min(s, i + window_i) + 1):
                dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j - 2], dtw[i - 1, j - 1], dtw[i - 1, j])

    if not np.isfinite(dtw[p, s]):
        raise RuntimeError("DTW failed to find a finite path; try disabling the window constraint.")

    i, j = p, s
    path_p: List[int] = [p - 1]
    path_s: List[int] = [s - 1]
    while i > 1 or j > 1:
        if slope_constraint == "symmetric":
            options = (
                (dtw[i - 1, j - 1], i - 1, j - 1),
                (dtw[i - 1, j], i - 1, j),
                (dtw[i, j - 1], i, j - 1),
            )
        else:
            options = (
                (dtw[i - 1, j], i - 1, j),
                (dtw[i - 1, j - 1], i - 1, j - 1),
                (dtw[i - 1, j - 2] if j >= 2 else np.inf, i - 1, max(0, j - 2)),
            )
        _, i, j = min(options, key=lambda item: item[0])
        path_p.insert(0, max(0, i - 1))
        path_s.insert(0, max(0, j - 1))
        if i <= 1 and j <= 1:
            break

    return np.asarray(path_p, dtype=np.int64), np.asarray(path_s, dtype=np.int64), float(dtw[p, s])


def _guided_warp_ct(
    anchor_ct: np.ndarray,
    prototype_ct: np.ndarray,
    *,
    slope_constraint: str,
    use_window: bool,
) -> Tuple[np.ndarray, float, float]:
    """Warp an anchor [C, T] by the DTW path from prototype to anchor."""
    anchor_ct = np.asarray(anchor_ct, dtype=np.float32)
    prototype_ct = np.asarray(prototype_ct, dtype=np.float32)
    t = int(anchor_ct.shape[1])
    window = int(np.ceil(t / 10.0)) if use_window else None
    path_p, path_s, dtw_value = _dtw_path_tc(
        prototype_ct.T,
        anchor_ct.T,
        slope_constraint=slope_constraint,
        window=window,
    )
    warped_ct = anchor_ct[:, path_s]
    x_aug = _resample_ct(warped_ct, t)
    orig_steps = np.arange(t, dtype=np.float64)
    warp_path_interp = np.interp(orig_steps, np.linspace(0.0, max(t - 1.0, 0.0), num=len(path_s)), path_s)
    warp_amount = float(np.sum(np.abs(orig_steps - warp_path_interp)))
    return x_aug.astype(np.float32), float(dtw_value), float(warp_amount)


def _window_slice_ct(
    x_ct: np.ndarray,
    *,
    reduce_ratio: float,
    rng: np.random.Generator,
    min_window_len: int = 4,
) -> np.ndarray:
    x_ct = np.asarray(x_ct, dtype=np.float32)
    t = int(x_ct.shape[1])
    if t <= 1:
        return x_ct.astype(np.float32, copy=True)
    win_len = int(round(float(reduce_ratio) * t))
    win_len = max(int(min_window_len), win_len)
    win_len = min(max(1, win_len), t)
    if win_len >= t:
        return x_ct.astype(np.float32, copy=True)
    start = int(rng.integers(0, t - win_len + 1))
    return _resample_ct(x_ct[:, start:start + win_len], t)


# ---------------------------------------------------------------------------
# JobDA clean-room supervised augmentation baseline
# ---------------------------------------------------------------------------


def jobda_cleanroom_augmented_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    transform_subseqs: Tuple[int, ...] = (0, 2, 4, 8),
) -> ExternalAugResult:
    """Build the JobDA joint-label training set.

    The output labels are joint labels: original_class * M + transform_id.
    At inference, a JobDA-aware evaluator must sum probabilities over the
    transform axis to recover original class probabilities.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    transforms = tuple(int(x) for x in transform_subseqs)
    if not transforms or transforms[0] != 0:
        transforms = (0,) + tuple(x for x in transforms if int(x) != 0)
    X_out: List[np.ndarray] = []
    y_joint: List[int] = []
    warning_count = 0
    n_transforms = len(transforms)
    for i, x in enumerate(X_train):
        cls = int(y_train[i])
        for transform_id, n_subseq in enumerate(transforms):
            if int(n_subseq) <= 0:
                x_aug = x.astype(np.float32, copy=True)
            else:
                if int(n_subseq) > x.shape[1]:
                    warning_count += 1
                x_aug = time_series_warping_cleanroom(x, int(n_subseq))
            X_out.append(x_aug)
            y_joint.append(cls * n_transforms + int(transform_id))
    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_joint, dtype=np.int64),
        source_space="raw_time",
        label_mode="joint_hard",
        uses_external_library=False,
        library_name="",
        budget_matched=False,
        selection_rule="jobda_cleanroom_tsw_joint_label",
        warning_count=int(warning_count),
        fallback_count=0,
        meta={
            "jobda_num_transforms": float(n_transforms),
            "jobda_transform_subseqs": ",".join(str(x) for x in transforms),
            "jobda_cleanroom": 1.0,
            "jobda_official_code_confirmed": 0.0,
            "actual_aug_ratio": float(max(0, n_transforms - 1)),
        },
    )


# ---------------------------------------------------------------------------
# TimeVAE-style classwise generative baseline
# ---------------------------------------------------------------------------


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
    """Classwise TimeVAE-style generator adapter.

    The official TimeVAE project is Keras/TensorFlow-based.  The current
    experiment environment does not require TensorFlow, so this baseline uses a
    compact PyTorch VAE adapter with the same role in the matrix: classwise
    train-only time-series generation.  Metadata marks it as a clean-room
    adapter rather than an official pipeline run.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rng = _rng(seed)
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
        X_aug = _finite_stack(X_out)
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


def diffusionts_classwise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    max_epochs: int = 500,
    batch_size: int = 128,
    device: str = "cpu",
) -> ExternalAugResult:
    """Classwise Diffusion-TS generator with classifier guidance.

    This baseline utilizes the Diffusion-TS (ICLR 2024) official implementation
    adapted for class-conditional synthesis within the PIA framework.
    """
    from utils.diffusionts_wrapper import fit_sample_diffusionts

    X_aug, y_aug = fit_sample_diffusionts(
        X_train_ct=X_train,
        y_train=y_train,
        multiplier=multiplier,
        seed=seed,
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size
    )

    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=y_aug,
        source_space="generative_model",
        label_mode="hard",
        uses_external_library=True,
        library_name="Diffusion-TS",
        budget_matched=True,
        selection_rule="classwise_diffusionts_classifier_guidance",
        meta={
            "diffusionts_max_epochs": float(max_epochs),
            "target_aug_ratio": float(multiplier),
            "actual_aug_ratio": float(X_aug.shape[0]) / max(float(len(X_train)), 1.0),
        }
    )


def timevqvae_classwise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    vqvae_epochs: int = 100,
    maskgit_epochs: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
) -> ExternalAugResult:
    """Classwise TimeVQVAE generator (AISTATS 2023)."""
    from utils.timevqvae_wrapper import fit_sample_timevqvae

    X_aug, y_aug = fit_sample_timevqvae(
        X_train_ct=X_train,
        y_train=y_train,
        multiplier=multiplier,
        seed=seed,
        device=device,
        vqvae_epochs=vqvae_epochs,
        maskgit_epochs=maskgit_epochs,
        batch_size=batch_size
    )

    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=y_aug,
        source_space="generative_model",
        label_mode="hard",
        uses_external_library=True,
        library_name="TimeVQVAE",
        budget_matched=True,
        selection_rule="classwise_timevqvae_maskgit",
        meta={
            "timevqvae_vqvae_epochs": float(vqvae_epochs),
            "timevqvae_maskgit_epochs": float(maskgit_epochs),
            "target_aug_ratio": float(multiplier),
            "actual_aug_ratio": float(X_aug.shape[0]) / max(float(len(X_train)), 1.0),
        }
    )


# ---------------------------------------------------------------------------
# Raw-domain transformation baselines
# ---------------------------------------------------------------------------


def raw_aug_jitter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    scale: float = 0.05,
) -> ExternalAugResult:
    try:
        from tsaug import AddNoise
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_aug_jitter requires optional dependency `tsaug`.") from exc

    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    np.random.seed(int(seed) % (2**32 - 1))
    X_tc = np.transpose(X_src, (0, 2, 1))
    X_aug = AddNoise(scale=float(scale)).augment(X_tc)
    X_aug = np.transpose(np.asarray(X_aug, dtype=np.float32), (0, 2, 1))
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=True,
        library_name="tsaug",
        budget_matched=True,
        selection_rule="repeat_train_anchors_addnoise",
    )


def raw_aug_scaling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    low: float = 0.8,
    high: float = 1.2,
) -> ExternalAugResult:
    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    factors = rng.uniform(float(low), float(high), size=(len(idx), 1, 1)).astype(np.float32)
    return ExternalAugResult(
        X_aug=X_src * factors,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_amplitude_uniform",
        meta={"scaling_low": float(low), "scaling_high": float(high)},
    )


def raw_aug_timewarp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    n_speed_change: int = 3,
    max_speed_ratio: float = 2.0,
) -> ExternalAugResult:
    try:
        from tsaug import TimeWarp
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_aug_timewarp requires optional dependency `tsaug`.") from exc

    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    np.random.seed(int(seed) % (2**32 - 1))
    X_tc = np.transpose(X_src, (0, 2, 1))
    X_aug = TimeWarp(
        n_speed_change=int(n_speed_change),
        max_speed_ratio=float(max_speed_ratio),
    ).augment(X_tc)
    X_aug = np.transpose(np.asarray(X_aug, dtype=np.float32), (0, 2, 1))
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=True,
        library_name="tsaug",
        budget_matched=True,
        selection_rule="repeat_train_anchors_timewarp",
    )


def raw_aug_magnitude_warping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    sigma: float = 0.2,
    knots: int = 4,
    per_channel_curve: bool = True,
) -> ExternalAugResult:
    try:
        from scipy.interpolate import CubicSpline
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_aug_magnitude_warping requires optional dependency `scipy`.") from exc

    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    n_aug, c, t = X_src.shape
    n_knots = max(1, int(knots))
    x_knots = np.linspace(0.0, float(max(t - 1, 1)), n_knots + 2)
    x_full = np.arange(t, dtype=np.float64)
    X_out = np.empty_like(X_src, dtype=np.float32)

    for i in range(n_aug):
        n_curves = c if per_channel_curve else 1
        knot_vals = rng.normal(1.0, float(sigma), size=(n_curves, n_knots + 2))
        knot_vals = np.clip(knot_vals, 0.05, None)
        curves = []
        for curve_idx in range(n_curves):
            curve = CubicSpline(x_knots, knot_vals[curve_idx], bc_type="natural")(x_full)
            curves.append(np.clip(curve, 0.05, None).astype(np.float32))
        curve_arr = np.stack(curves, axis=0)
        if not per_channel_curve:
            curve_arr = np.repeat(curve_arr, c, axis=0)
        X_out[i] = X_src[i] * curve_arr

    return ExternalAugResult(
        X_aug=np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_magnitude_warping",
        meta={
            "warp_sigma": float(sigma),
            "warp_knots": float(knots),
            "per_channel_curve": float(bool(per_channel_curve)),
        },
    )


def raw_aug_window_warping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    window_ratio: float = 0.10,
    speed_factors: Tuple[float, ...] = (0.5, 2.0),
    min_window_len: int = 4,
) -> ExternalAugResult:
    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    t = int(X_src.shape[-1])
    X_out: List[np.ndarray] = []
    fallback_count = 0
    for x in X_src:
        if t < 3:
            fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        win_len = int(round(float(window_ratio) * t))
        win_len = max(int(min_window_len), win_len)
        win_len = min(max(1, win_len), max(1, t - 1))
        if win_len >= t:
            fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        start = int(rng.integers(0, t - win_len + 1))
        speed = float(rng.choice(np.asarray(speed_factors, dtype=np.float64)))
        warped_len = max(1, int(round(win_len * speed)))
        before = x[:, :start]
        segment = x[:, start:start + win_len]
        after = x[:, start + win_len:]
        warped = _resample_ct(segment, warped_len)
        stitched = np.concatenate([before, warped, after], axis=1)
        X_out.append(_resample_ct(stitched, t))

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_window_warping",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "window_warp_ratio": float(window_ratio),
            "window_warp_min_window_len": float(min_window_len),
            "window_warp_speed_min": float(np.min(speed_factors)),
            "window_warp_speed_max": float(np.max(speed_factors)),
        },
    )


def raw_aug_window_slicing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    slice_ratio: float = 0.90,
    min_window_len: int = 4,
) -> ExternalAugResult:
    rng = _rng(seed)
    idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_src = np.asarray(X_train[idx], dtype=np.float32)
    t = int(X_src.shape[-1])
    X_out: List[np.ndarray] = []
    fallback_count = 0
    for x in X_src:
        slice_len = int(round(float(slice_ratio) * t))
        slice_len = max(int(min_window_len), slice_len)
        slice_len = min(max(1, slice_len), t)
        if t <= 1 or slice_len >= t:
            if t <= 1:
                fallback_count += 1
            X_out.append(x.astype(np.float32, copy=True))
            continue
        start = int(rng.integers(0, t - slice_len + 1))
        X_out.append(_resample_ct(x[:, start:start + slice_len], t))

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_train[idx], dtype=np.int64),
        source_space="raw_time",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="repeat_train_anchors_window_slicing",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "window_slice_ratio": float(slice_ratio),
            "window_slice_min_window_len": float(min_window_len),
        },
    )


# ---------------------------------------------------------------------------
# Vicinal / interpolation baselines
# ---------------------------------------------------------------------------


def raw_mixup(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    alpha: float = 0.4,
    n_classes: Optional[int] = None,
) -> ExternalAugResult:
    rng = _rng(seed)
    n_train = int(len(X_train))
    n_aug = int(multiplier) * n_train
    n_classes_i = int(n_classes if n_classes is not None else np.max(y_train) + 1)
    i = rng.integers(0, n_train, size=n_aug)
    j = rng.integers(0, n_train, size=n_aug)
    lam = rng.beta(float(alpha), float(alpha), size=(n_aug, 1, 1)).astype(np.float32)
    X_aug = lam * np.asarray(X_train[i], dtype=np.float32) + (1.0 - lam) * np.asarray(X_train[j], dtype=np.float32)

    lam_y = lam.reshape(n_aug, 1)
    y_i = _one_hot(np.asarray(y_train[i], dtype=np.int64), n_classes_i)
    y_j = _one_hot(np.asarray(y_train[j], dtype=np.int64), n_classes_i)
    y_soft = lam_y * y_i + (1.0 - lam_y) * y_j
    return ExternalAugResult(
        X_aug=X_aug.astype(np.float32),
        y_aug_soft=y_soft.astype(np.float32),
        source_space="raw_mixup",
        label_mode="soft",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="train_split_random_pair_beta",
        meta={"mixup_alpha": float(alpha)},
    )


# ---------------------------------------------------------------------------
# DTW barycenter and DTW pattern-mixing baselines
# ---------------------------------------------------------------------------


def dba_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    k: int = 5,
    max_iter: int = 5,
) -> ExternalAugResult:
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("dba_sameclass requires optional dependency `tslearn`.") from exc

    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = {int(c): np.flatnonzero(y_train == c) for c in np.unique(y_train)}
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    warning_count = 0

    for anchor in anchor_idx:
        cls = int(y_train[int(anchor)])
        pool = class_to_idx[cls]
        replace = len(pool) < int(k)
        if replace:
            warning_count += 1
        chosen = rng.choice(pool, size=int(k), replace=replace)
        group_tc = np.transpose(np.asarray(X_train[chosen], dtype=np.float64), (0, 2, 1))
        bary_tc = dtw_barycenter_averaging(group_tc, max_iter=int(max_iter))
        X_out.append(np.transpose(np.asarray(bary_tc, dtype=np.float32), (1, 0)))
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=np.stack(X_out, axis=0).astype(np.float32),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_barycenter",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="same_class_dba",
        warning_count=int(warning_count),
        meta={"dba_k": float(k), "dba_max_iter": float(max_iter)},
    )


def wdba_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    k: int = 5,
    max_iter: int = 5,
    tau: Optional[float] = None,
) -> ExternalAugResult:
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
        from tslearn.metrics import dtw
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("wdba_sameclass requires optional dependency `tslearn`.") from exc

    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = _class_to_indices(y_train)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0
    tau_values: List[float] = []
    k_eff = int(k)

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        pool = class_to_idx[cls]
        replace = len(pool) < k_eff
        if replace:
            fallback_count += 1
        chosen = rng.choice(pool, size=k_eff, replace=replace)
        if anchor_i not in chosen:
            chosen[0] = anchor_i
        group_tc = np.transpose(np.asarray(X_train[chosen], dtype=np.float64), (0, 2, 1))
        anchor_tc = np.asarray(X_train[anchor_i], dtype=np.float64).T
        dists = np.asarray([float(dtw(anchor_tc, group_tc[j])) for j in range(k_eff)], dtype=np.float64)
        if tau is None:
            positive = dists[dists > 1e-12]
            tau_i = float(np.median(positive)) if positive.size else 1.0
            if not np.isfinite(tau_i) or tau_i <= 1e-12:
                tau_i = 1.0
                fallback_count += 1
        else:
            tau_i = float(tau)
        tau_values.append(tau_i)
        logits = -dists / max(tau_i, 1e-12)
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        weights /= float(np.sum(weights) + 1e-12)
        try:
            bary_tc = dtw_barycenter_averaging(group_tc, weights=weights, max_iter=int(max_iter))
        except Exception:
            fallback_count += 1
            bary_tc = np.average(group_tc, axis=0, weights=weights)
        X_out.append(np.transpose(np.asarray(bary_tc, dtype=np.float32), (1, 0)))
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_barycenter",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="same_class_weighted_dba_anchor_dtw_softmax",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "wdba_k": float(k),
            "wdba_max_iter": float(max_iter),
            "wdba_tau": float(np.mean(tau_values)) if tau_values else float("nan"),
        },
    )


def spawner_sameclass_style(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    noise_scale: float = 0.05,
) -> ExternalAugResult:
    try:
        from tslearn.metrics import dtw_path
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("spawner_sameclass_style requires optional dependency `tslearn`.") from exc

    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = _class_to_indices(y_train)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        x = np.asarray(X_train[anchor_i], dtype=np.float32)
        pool = class_to_idx[cls]
        candidates = pool[pool != anchor_i]
        if len(candidates) == 0:
            mate_i = anchor_i
            fallback_count += 1
        else:
            mate_i = int(rng.choice(candidates))
        mate = np.asarray(X_train[mate_i], dtype=np.float32)
        try:
            path, _ = dtw_path(x.T.astype(np.float64), mate.T.astype(np.float64))
            aligned = np.empty_like(x, dtype=np.float32)
            buckets: List[List[int]] = [[] for _ in range(x.shape[1])]
            for i_t, j_t in path:
                if 0 <= int(i_t) < x.shape[1] and 0 <= int(j_t) < mate.shape[1]:
                    buckets[int(i_t)].append(int(j_t))
            for i_t, js in enumerate(buckets):
                if js:
                    aligned[:, i_t] = np.mean(mate[:, js], axis=1)
                else:
                    aligned[:, i_t] = mate[:, min(i_t, mate.shape[1] - 1)]
        except Exception:
            aligned = mate
            fallback_count += 1
        mixed = 0.5 * x + 0.5 * aligned
        ch_std = np.std(x, axis=1, keepdims=True).astype(np.float32)
        noise = rng.normal(0.0, float(noise_scale), size=x.shape).astype(np.float32) * (ch_std + 1e-6)
        X_out.append(mixed + noise)
        y_out.append(cls)

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_pattern_mix",
        label_mode="hard",
        uses_external_library=True,
        library_name="tslearn",
        budget_matched=True,
        selection_rule="spawner_style_same_class_dtw_aligned_average",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={"spawner_noise_scale": float(noise_scale)},
    )


def rgw_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    slope_constraint: str = "symmetric",
    use_window: bool = True,
) -> ExternalAugResult:
    """Random Guided Warping clean-room adapter.

    Based on the RGW idea from the uchidalab guided-warping implementation:
    choose a random same-class prototype and time-warp the anchor along the
    DTW path from prototype to anchor.  This implementation keeps the project
    native [N, C, T] shape and does not vendor the external TensorFlow/Keras
    repository.
    """
    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = _class_to_indices(y_train)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0
    dtw_values: List[float] = []
    warp_amounts: List[float] = []

    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        pool = class_to_idx[cls]
        candidates = pool[pool != anchor_i]
        if len(candidates) == 0:
            X_out.append(np.asarray(X_train[anchor_i], dtype=np.float32).copy())
            y_out.append(cls)
            fallback_count += 1
            continue
        prototype_i = int(rng.choice(candidates))
        try:
            x_aug, dtw_value, warp_amount = _guided_warp_ct(
                np.asarray(X_train[anchor_i], dtype=np.float32),
                np.asarray(X_train[prototype_i], dtype=np.float32),
                slope_constraint=slope_constraint,
                use_window=bool(use_window),
            )
        except Exception:
            x_aug = np.asarray(X_train[anchor_i], dtype=np.float32).copy()
            dtw_value = float("nan")
            warp_amount = 0.0
            fallback_count += 1
        X_out.append(x_aug)
        y_out.append(cls)
        dtw_values.append(float(dtw_value))
        warp_amounts.append(float(warp_amount))

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_guided_warp",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="random_guided_warp_same_class_dtw_cleanroom",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "guided_warp_mode": "rgw",
            "guided_warp_slope_constraint": str(slope_constraint),
            "guided_warp_use_window": float(bool(use_window)),
            "guided_warp_dtw_value_mean": float(np.nanmean(dtw_values)) if dtw_values else float("nan"),
            "guided_warp_amount_mean": float(np.nanmean(warp_amounts)) if warp_amounts else float("nan"),
            "guided_warp_cleanroom_adapter": 1.0,
        },
    )


def dgw_sameclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    batch_size: int = 6,
    slope_constraint: str = "symmetric",
    use_window: bool = True,
    use_variable_slice: bool = True,
    min_window_len: int = 4,
) -> ExternalAugResult:
    """Discriminative Guided Warping clean-room adapter.

    For each anchor, sample same-class and different-class prototypes.  Select
    the same-class prototype that is far from negatives and close to positives,
    then warp the anchor along that prototype-anchor DTW path.  Optional
    variable slicing follows the original DGW idea: stronger warp paths keep a
    larger slice after warping.
    """
    rng = _rng(seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    class_to_idx = _class_to_indices(y_train)
    all_idx = np.arange(len(X_train), dtype=np.int64)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    X_out: List[np.ndarray] = []
    y_out: List[int] = []
    fallback_count = 0
    dtw_values: List[float] = []
    warp_amounts: List[float] = []
    score_values: List[float] = []
    positive_batch = max(1, int(np.ceil(float(batch_size) / 2.0)))
    negative_batch = max(1, int(np.floor(float(batch_size) / 2.0)))

    def distance_ct(a_ct: np.ndarray, b_ct: np.ndarray) -> float:
        _, _, dist = _dtw_path_tc(
            np.asarray(a_ct, dtype=np.float32).T,
            np.asarray(b_ct, dtype=np.float32).T,
            slope_constraint=slope_constraint,
            window=int(np.ceil(a_ct.shape[1] / 10.0)) if use_window else None,
        )
        return float(dist)

    warped_items: List[Tuple[np.ndarray, float]] = []
    for anchor in anchor_idx:
        anchor_i = int(anchor)
        cls = int(y_train[anchor_i])
        positive = class_to_idx[cls]
        positive = positive[positive != anchor_i]
        negative = all_idx[y_train != cls]
        if len(positive) == 0 or len(negative) == 0:
            warped_items.append((np.asarray(X_train[anchor_i], dtype=np.float32).copy(), 0.0))
            y_out.append(cls)
            fallback_count += 1
            continue

        pos_chosen = rng.choice(positive, size=min(len(positive), positive_batch), replace=False)
        neg_chosen = rng.choice(negative, size=min(len(negative), negative_batch), replace=False)
        best_score = -np.inf
        best_proto_i = int(pos_chosen[0])
        for proto_i in pos_chosen:
            proto = np.asarray(X_train[int(proto_i)], dtype=np.float32)
            other_pos = [int(x) for x in pos_chosen if int(x) != int(proto_i)]
            try:
                pos_dist = float(np.mean([distance_ct(proto, np.asarray(X_train[j], dtype=np.float32)) for j in other_pos])) if other_pos else 0.0
                neg_dist = float(np.mean([distance_ct(proto, np.asarray(X_train[int(j)], dtype=np.float32)) for j in neg_chosen]))
                score = neg_dist - pos_dist
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score = float(score)
                best_proto_i = int(proto_i)

        try:
            x_aug, dtw_value, warp_amount = _guided_warp_ct(
                np.asarray(X_train[anchor_i], dtype=np.float32),
                np.asarray(X_train[best_proto_i], dtype=np.float32),
                slope_constraint=slope_constraint,
                use_window=bool(use_window),
            )
        except Exception:
            x_aug = np.asarray(X_train[anchor_i], dtype=np.float32).copy()
            dtw_value = float("nan")
            warp_amount = 0.0
            fallback_count += 1
        warped_items.append((x_aug, float(warp_amount)))
        y_out.append(cls)
        dtw_values.append(float(dtw_value))
        warp_amounts.append(float(warp_amount))
        score_values.append(float(best_score))

    max_warp = max([amount for _, amount in warped_items], default=0.0)
    for x_aug, warp_amount in warped_items:
        if use_variable_slice:
            if max_warp > 1e-12:
                reduce_ratio = 0.9 + 0.1 * float(warp_amount) / float(max_warp)
            else:
                reduce_ratio = 0.9
            x_aug = _window_slice_ct(x_aug, reduce_ratio=reduce_ratio, rng=rng, min_window_len=min_window_len)
        X_out.append(x_aug)

    return ExternalAugResult(
        X_aug=_finite_stack(X_out),
        y_aug=np.asarray(y_out, dtype=np.int64),
        source_space="dtw_guided_warp",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="discriminative_guided_warp_same_class_dtw_cleanroom",
        warning_count=int(fallback_count),
        fallback_count=int(fallback_count),
        meta={
            "guided_warp_mode": "dgw",
            "guided_warp_batch_size": float(batch_size),
            "guided_warp_positive_batch": float(positive_batch),
            "guided_warp_negative_batch": float(negative_batch),
            "guided_warp_slope_constraint": str(slope_constraint),
            "guided_warp_use_window": float(bool(use_window)),
            "guided_warp_use_variable_slice": float(bool(use_variable_slice)),
            "guided_warp_dtw_value_mean": float(np.nanmean(dtw_values)) if dtw_values else float("nan"),
            "guided_warp_amount_mean": float(np.nanmean(warp_amounts)) if warp_amounts else float("nan"),
            "guided_warp_score_mean": float(np.nanmean(score_values)) if score_values else float("nan"),
            "guided_warp_cleanroom_adapter": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Flattened raw-space SMOTE baseline
# ---------------------------------------------------------------------------


def raw_smote_flatten_balanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> ExternalAugResult:
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("raw_smote_flatten_balanced requires optional dependency `imbalanced-learn`.") from exc

    y_train = np.asarray(y_train, dtype=np.int64)
    _, counts = np.unique(y_train, return_counts=True)
    if counts.size == 0 or int(counts.min()) < 2:
        empty = np.empty((0, X_train.shape[1], X_train.shape[2]), dtype=np.float32)
        return ExternalAugResult(
            X_aug=empty,
            y_aug=np.empty((0,), dtype=np.int64),
            source_space="flattened_raw",
            label_mode="hard",
            uses_external_library=True,
            library_name="imbalanced-learn",
            budget_matched=False,
            selection_rule="class_balancing_smote_auto",
            warning_count=1,
        )

    k_neighbors = max(1, min(5, int(counts.min()) - 1))
    flat = np.asarray(X_train, dtype=np.float32).reshape(len(X_train), -1)
    smote = SMOTE(sampling_strategy="auto", random_state=int(seed), k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(flat, y_train)
    X_new = np.asarray(X_res[len(X_train):], dtype=np.float32).reshape(-1, X_train.shape[1], X_train.shape[2])
    y_new = np.asarray(y_res[len(y_train):], dtype=np.int64)
    return ExternalAugResult(
        X_aug=X_new,
        y_aug=y_new,
        source_space="flattened_raw",
        label_mode="hard",
        uses_external_library=True,
        library_name="imbalanced-learn",
        budget_matched=False,
        selection_rule="class_balancing_smote_auto",
        meta={"smote_k_neighbors": float(k_neighbors)},
    )


# ---------------------------------------------------------------------------
# Naive Log-Euclidean covariance-state baselines
# ---------------------------------------------------------------------------


def _build_covariance_records(X_train: np.ndarray, spd_eps: float = 1e-4) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    records: List[Dict[str, np.ndarray]] = []
    log_covs = []
    for x_np in np.asarray(X_train, dtype=np.float32):
        x = torch.from_numpy(x_np).double()
        x = x - x.mean(dim=-1, keepdim=True)
        cov = (x @ x.transpose(-1, -2)) / float(max(1, x.shape[-1] - 1))
        cov = cov + float(spd_eps) * torch.eye(cov.shape[0], dtype=cov.dtype)
        vals, vecs = torch.linalg.eigh(cov)
        log_cov = vecs @ torch.diag_embed(torch.log(torch.clamp(vals, min=spd_eps))) @ vecs.transpose(-1, -2)
        log_covs.append(log_cov.numpy())
        records.append({"x_raw": x_np, "sigma_orig": cov.numpy(), "log_cov": log_cov.numpy()})

    mean_log = np.mean(log_covs, axis=0)
    idx = np.triu_indices(mean_log.shape[0])
    for record in records:
        record["z"] = (record["log_cov"] - mean_log)[idx].astype(np.float32)
    return records, mean_log.astype(np.float64)


def _materialize_cov_state_aug(
    records: List[Dict[str, np.ndarray]],
    mean_log: np.ndarray,
    z_cands: np.ndarray,
    anchor_idx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    X_aug = []
    transport_errors = []
    for z, idx in zip(z_cands, anchor_idx):
        rec = records[int(idx)]
        sigma_aug = logvec_to_spd(np.asarray(z, dtype=np.float32), mean_log)
        x_aug, meta = bridge_single(
            torch.from_numpy(rec["x_raw"]),
            torch.from_numpy(rec["sigma_orig"]),
            torch.from_numpy(sigma_aug),
        )
        X_aug.append(x_aug.cpu().numpy().astype(np.float32))
        transport_errors.append(float(meta.get("transport_error_logeuc", np.nan)))
    mean_err = float(np.nanmean(transport_errors)) if transport_errors else float("nan")
    return np.stack(X_aug, axis=0).astype(np.float32), mean_err


def random_cov_state(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    gamma: float,
) -> ExternalAugResult:
    rng = _rng(seed)
    records, mean_log = _build_covariance_records(X_train)
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    z_dim = int(records[0]["z"].shape[0])
    dirs = rng.normal(size=(len(anchor_idx), z_dim)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    signs = np.where(np.arange(len(anchor_idx)) % 2 == 0, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    z0 = np.stack([records[int(i)]["z"] for i in anchor_idx], axis=0)
    z_cands = z0 + signs * float(gamma) * dirs
    X_aug, transport_err = _materialize_cov_state_aug(records, mean_log, z_cands, anchor_idx)
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[anchor_idx], dtype=np.int64),
        source_space="covariance_state",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="random_unit_z_direction",
        meta={"transport_error_logeuc_mean": transport_err},
    )


def pca_cov_state(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    multiplier: int,
    seed: int,
    gamma: float,
    k_dir: int,
) -> ExternalAugResult:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("pca_cov_state requires optional dependency `scikit-learn`.") from exc

    records, mean_log = _build_covariance_records(X_train)
    Z = np.stack([rec["z"] for rec in records], axis=0)
    n_components = max(1, min(int(k_dir), int(Z.shape[0]), int(Z.shape[1])))
    pca = PCA(n_components=n_components, random_state=int(seed))
    pca.fit(Z)
    components = np.asarray(pca.components_, dtype=np.float32)
    components /= np.linalg.norm(components, axis=1, keepdims=True) + 1e-12
    anchor_idx = _repeat_anchor_indices(len(X_train), multiplier)
    slots = np.arange(len(anchor_idx), dtype=np.int64)
    dirs = components[slots % n_components]
    signs = np.where((slots // n_components) % 2 == 0, 1.0, -1.0).astype(np.float32).reshape(-1, 1)
    z0 = np.stack([records[int(i)]["z"] for i in anchor_idx], axis=0)
    z_cands = z0 + signs * float(gamma) * dirs
    X_aug, transport_err = _materialize_cov_state_aug(records, mean_log, z_cands, anchor_idx)
    return ExternalAugResult(
        X_aug=X_aug,
        y_aug=np.asarray(y_train[anchor_idx], dtype=np.int64),
        source_space="covariance_state",
        label_mode="hard",
        uses_external_library=False,
        library_name="",
        budget_matched=True,
        selection_rule="pca_top_z_direction",
        meta={
            "pca_n_components": float(n_components),
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "transport_error_logeuc_mean": transport_err,
        },
    )
