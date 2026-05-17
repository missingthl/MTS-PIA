from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.curriculum import estimate_local_manifold_margins

from .materialize import materialize_z_aug_out
from .state import TrialRecord


LATENT_RESIDUAL_METHODS = {"latent_residual_direct", "latent_residual_flow"}


@dataclass(frozen=True)
class LatentResidualConfig:
    flow_epochs: int = 50
    flow_batch_size: int = 128
    flow_lr: float = 1e-3
    flow_weight_decay: float = 1e-4
    hidden_layers: int = 2
    hidden_width: int = 0
    class_embedding_dim: int = 0
    lambda_cos: float = 0.5
    rbf_tau_floor: float = 1e-12
    eps: float = 1e-12


class _LatentResidualMLP(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        n_classes: int,
        class_embedding_dim: int,
        hidden_width: int,
        hidden_layers: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, class_embedding_dim)
        in_dim = z_dim + class_embedding_dim + z_dim + 1
        layers: List[nn.Module] = []
        for idx in range(max(1, int(hidden_layers))):
            layers.append(nn.Linear(in_dim if idx == 0 else hidden_width, hidden_width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_width, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, r_s: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(y)
        if s.ndim == 1:
            s = s[:, None]
        return self.net(torch.cat([z, emb, r_s, s], dim=1))


def _unit(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        return np.zeros_like(v, dtype=np.float64), norm
    return (v / norm).astype(np.float64), norm


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return np.nan
    return float(np.dot(a, b) / denom)


def _effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0 or X.shape[0] <= 1:
        return 0.0
    s = np.linalg.svd(X - np.mean(X, axis=0, keepdims=True), compute_uv=False)
    energy = s * s
    total = float(np.sum(energy))
    if total <= eps:
        return 0.0
    p = energy / total
    return float(np.exp(-np.sum(p * np.log(p + eps))))


def _pairwise_cosine_mean(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] <= 1:
        return np.nan
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    sims = Xn @ Xn.T
    iu = np.triu_indices(X.shape[0], k=1)
    return float(np.mean(sims[iu])) if len(iu[0]) else np.nan


def _direction_diversity(rows: List[Dict[str, object]], eps: float = 1e-12) -> Dict[str, object]:
    by_anchor: Dict[int, List[np.ndarray]] = {}
    for row in rows:
        by_anchor.setdefault(int(row["anchor_index"]), []).append(np.asarray(row["_direction_vec"], dtype=np.float64))
    unique_ratios: List[float] = []
    pairwise: List[float] = []
    effective_multipliers: List[int] = []
    for dirs in by_anchor.values():
        if not dirs:
            continue
        D = np.stack(dirs)
        rounded = np.round(D, decimals=6)
        unique_count = int(np.unique(rounded, axis=0).shape[0])
        unique_ratios.append(unique_count / max(float(len(dirs)), 1.0))
        effective_multipliers.append(unique_count)
        if len(dirs) > 1:
            pairwise.append(_pairwise_cosine_mean(D, eps=eps))
    return {
        "latent_unique_direction_ratio": float(np.mean(unique_ratios)) if unique_ratios else np.nan,
        "latent_generated_direction_pairwise_cosine_mean": float(np.nanmean(pairwise)) if pairwise else np.nan,
        "latent_effective_aug_multiplier": float(np.mean(effective_multipliers)) if effective_multipliers else 0.0,
    }


def _class_remap(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    labels = sorted(int(v) for v in np.unique(y))
    mapping = {label: idx for idx, label in enumerate(labels)}
    return np.asarray([mapping[int(v)] for v in y], dtype=np.int64), mapping


def _build_rbf_sampler(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    cfg: LatentResidualConfig,
) -> Dict[str, object]:
    """Precompute train-only same-class RBF sampling distributions."""
    n = int(Z.shape[0])
    global_sq: List[float] = []
    class_tau: Dict[int, float] = {}
    for cls in sorted(int(v) for v in np.unique(y)):
        ids = np.where(y == cls)[0]
        if ids.size > 1:
            diffs = Z[ids][:, None, :] - Z[ids][None, :, :]
            sq = np.sum(diffs * diffs, axis=2)
            iu = np.triu_indices(ids.size, k=1)
            vals = sq[iu]
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > float(cfg.rbf_tau_floor)]
            if vals.size:
                global_sq.extend(float(v) for v in vals)
                class_tau[cls] = float(np.median(vals))
    global_tau = float(np.median(global_sq)) if global_sq else 1.0
    if not np.isfinite(global_tau) or global_tau <= float(cfg.rbf_tau_floor):
        global_tau = 1.0

    candidates: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    entropies: List[float] = []
    by_class_entropy: Dict[int, List[float]] = {}
    for i in range(n):
        cls = int(y[i])
        same = np.where(y == cls)[0]
        same = same[same != i]
        if same.size == 0:
            candidates.append(np.asarray([], dtype=np.int64))
            probs.append(np.asarray([], dtype=np.float64))
            entropies.append(0.0)
            by_class_entropy.setdefault(cls, []).append(0.0)
            continue
        diff = Z[same] - Z[i]
        dist_sq = np.sum(diff * diff, axis=1)
        tau = float(class_tau.get(cls, global_tau))
        if not np.isfinite(tau) or tau <= float(cfg.rbf_tau_floor):
            tau = global_tau
        logits = -dist_sq / max(tau, float(cfg.rbf_tau_floor))
        logits = logits - float(np.max(logits))
        p = np.exp(logits)
        p_sum = float(np.sum(p))
        if not np.isfinite(p_sum) or p_sum <= float(cfg.eps):
            p = np.ones_like(p, dtype=np.float64) / max(float(p.size), 1.0)
        else:
            p = p / p_sum
        entropy = float(-np.sum(p * np.log(p + float(cfg.eps))))
        candidates.append(same.astype(np.int64))
        probs.append(p.astype(np.float64))
        entropies.append(entropy)
        by_class_entropy.setdefault(cls, []).append(entropy)

    class_entropy_values = [float(np.mean(vals)) for vals in by_class_entropy.values() if vals]
    return {
        "candidates": candidates,
        "probs": probs,
        "global_tau": global_tau,
        "class_tau": class_tau,
        "sampling_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "sampling_entropy_by_class_mean": float(np.mean(class_entropy_values)) if class_entropy_values else 0.0,
        "sampling_entropy_by_class_min": float(np.min(class_entropy_values)) if class_entropy_values else 0.0,
    }


def _sample_residual(
    *,
    Z: np.ndarray,
    y: np.ndarray,
    sampler: Dict[str, object],
    anchor_index: int,
    rng: np.random.Generator,
    cfg: LatentResidualConfig,
) -> Dict[str, object]:
    candidates: List[np.ndarray] = sampler["candidates"]
    probs: List[np.ndarray] = sampler["probs"]
    cand = candidates[int(anchor_index)]
    p = probs[int(anchor_index)]
    fallback = False
    reason = ""
    if cand.size == 0:
        target_index = -1
        target_class = int(y[anchor_index])
        raw = rng.normal(size=(Z.shape[1],))
        residual, residual_norm = _unit(raw, eps=float(cfg.eps))
        target_dist = float(residual_norm)
        sampling_prob = 0.0
        fallback = True
        reason = "singleton_class_seeded_random"
    else:
        pos = int(rng.choice(np.arange(cand.size), p=p))
        target_index = int(cand[pos])
        target_class = int(y[target_index])
        raw = Z[target_index] - Z[anchor_index]
        residual, residual_norm = _unit(raw, eps=float(cfg.eps))
        target_dist = float(np.linalg.norm(raw))
        sampling_prob = float(p[pos])
        if residual_norm <= float(cfg.eps):
            raw = rng.normal(size=(Z.shape[1],))
            residual, residual_norm = _unit(raw, eps=float(cfg.eps))
            fallback = True
            reason = "zero_residual_seeded_random"
    return {
        "target_index": int(target_index),
        "target_class": int(target_class),
        "target_dist": float(target_dist),
        "target_sampling_prob": float(sampling_prob),
        "residual": residual.astype(np.float64),
        "residual_norm": float(residual_norm),
        "fallback_flag": bool(fallback),
        "fallback_reason": reason,
    }


def build_latent_residual_targets(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cfg: LatentResidualConfig | None = None,
) -> Dict[str, object]:
    """Build train-only residual-flow training tuples from same-class RBF residuals."""
    cfg = cfg or LatentResidualConfig()
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n, z_dim = Z.shape
    rng = np.random.default_rng(int(seed) + 9041)
    sampler = _build_rbf_sampler(Z, y, cfg=cfg)

    residuals = np.zeros((n, z_dim), dtype=np.float64)
    epsilons = np.zeros((n, z_dim), dtype=np.float64)
    r_s = np.zeros((n, z_dim), dtype=np.float64)
    s_train = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float64)
    u_target = np.zeros((n, z_dim), dtype=np.float64)
    target_indices = np.full((n,), -1, dtype=np.int64)
    target_classes = np.asarray(y, dtype=np.int64).copy()
    target_dists: List[float] = []
    target_probs: List[float] = []
    residual_norms: List[float] = []
    target_velocity_norms: List[float] = []
    fallback_flags: List[bool] = []
    fallback_reasons: List[str] = []

    for i in range(n):
        sample = _sample_residual(Z=Z, y=y, sampler=sampler, anchor_index=i, rng=rng, cfg=cfg)
        eps, _ = _unit(rng.normal(size=(z_dim,)), eps=float(cfg.eps))
        residual = np.asarray(sample["residual"], dtype=np.float64)
        u = residual - eps
        residuals[i] = residual
        epsilons[i] = eps
        r_s[i] = (1.0 - s_train[i]) * eps + s_train[i] * residual
        u_target[i] = u
        target_indices[i] = int(sample["target_index"])
        target_classes[i] = int(sample["target_class"])
        target_dists.append(float(sample["target_dist"]))
        target_probs.append(float(sample["target_sampling_prob"]))
        residual_norms.append(float(sample["residual_norm"]))
        target_velocity_norms.append(float(np.linalg.norm(u)))
        fallback_flags.append(bool(sample["fallback_flag"]))
        fallback_reasons.append(str(sample["fallback_reason"]))

    summary = {
        "latent_target_dist_mean": float(np.mean(target_dists)) if target_dists else np.nan,
        "latent_target_dist_std": float(np.std(target_dists)) if target_dists else np.nan,
        "latent_target_sampling_entropy": float(sampler["sampling_entropy"]),
        "latent_target_sampling_entropy_by_class_mean": float(sampler["sampling_entropy_by_class_mean"]),
        "latent_target_sampling_entropy_by_class_min": float(sampler["sampling_entropy_by_class_min"]),
        "latent_fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
        "latent_residual_effective_rank": _effective_rank(residuals),
        "latent_residual_pairwise_cosine_mean": _pairwise_cosine_mean(residuals),
        "latent_target_velocity_norm_mean": float(np.mean(target_velocity_norms)) if target_velocity_norms else np.nan,
        "latent_target_velocity_norm_std": float(np.std(target_velocity_norms)) if target_velocity_norms else np.nan,
    }
    return {
        "sampler": sampler,
        "residuals": residuals,
        "epsilons": epsilons,
        "r_s": r_s,
        "s_train": s_train,
        "u_target": u_target,
        "target_indices": target_indices,
        "target_classes": target_classes,
        "target_dists": np.asarray(target_dists, dtype=np.float64),
        "target_sampling_probs": np.asarray(target_probs, dtype=np.float64),
        "residual_norms": np.asarray(residual_norms, dtype=np.float64),
        "target_velocity_norms": np.asarray(target_velocity_norms, dtype=np.float64),
        "fallback_flags": np.asarray(fallback_flags, dtype=bool),
        "fallback_reasons": fallback_reasons,
        "summary": summary,
    }


def fit_latent_residual_operator(
    Z: np.ndarray,
    y: np.ndarray,
    target_out: Dict[str, object],
    *,
    seed: int,
    device: str,
    cfg: LatentResidualConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or LatentResidualConfig()
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    y_mapped, mapping = _class_remap(y)
    _, z_dim = Z.shape
    n_classes = len(mapping)
    hidden_width = int(cfg.hidden_width) if int(cfg.hidden_width) > 0 else max(128, 2 * int(z_dim))
    emb_dim = int(cfg.class_embedding_dim) if int(cfg.class_embedding_dim) > 0 else min(16, max(4, n_classes))
    torch.manual_seed(int(seed) + 9137)
    np.random.seed(int(seed) + 9137)
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = _LatentResidualMLP(
        z_dim=z_dim,
        n_classes=n_classes,
        class_embedding_dim=emb_dim,
        hidden_width=hidden_width,
        hidden_layers=int(cfg.hidden_layers),
    ).to(dev)
    ds = TensorDataset(
        torch.from_numpy(np.asarray(Z, dtype=np.float32)),
        torch.from_numpy(np.asarray(target_out["r_s"], dtype=np.float32)),
        torch.from_numpy(np.asarray(target_out["s_train"], dtype=np.float32)),
        torch.from_numpy(y_mapped.astype(np.int64)),
        torch.from_numpy(np.asarray(target_out["u_target"], dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 9137)
    loader = DataLoader(ds, batch_size=int(cfg.flow_batch_size), shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.flow_lr), weight_decay=float(cfg.flow_weight_decay))
    model.train()
    for _ in range(int(cfg.flow_epochs)):
        for z_b, r_b, s_b, y_b, u_b in loader:
            z_b = z_b.to(dev)
            r_b = r_b.to(dev)
            s_b = s_b.to(dev)
            y_b = y_b.to(dev)
            u_b = u_b.to(dev)
            pred = model(z_b, r_b, s_b, y_b)
            mse = torch.mean((pred - u_b) ** 2)
            pred_n = pred / torch.clamp(torch.linalg.norm(pred, dim=1, keepdim=True), min=float(cfg.eps))
            u_n = u_b / torch.clamp(torch.linalg.norm(u_b, dim=1, keepdim=True), min=float(cfg.eps))
            cos_loss = torch.mean(1.0 - torch.sum(pred_n * u_n, dim=1))
            loss = mse + float(cfg.lambda_cos) * cos_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        z_all = torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(dev)
        r_all = torch.from_numpy(np.asarray(target_out["r_s"], dtype=np.float32)).to(dev)
        s_all = torch.from_numpy(np.asarray(target_out["s_train"], dtype=np.float32)).to(dev)
        y_all = torch.from_numpy(y_mapped.astype(np.int64)).to(dev)
        target = torch.from_numpy(np.asarray(target_out["u_target"], dtype=np.float32)).to(dev)
        pred = model(z_all, r_all, s_all, y_all)
        mse = torch.mean((pred - target) ** 2).item()
        pred_norm = torch.linalg.norm(pred, dim=1)
        target_norm = torch.linalg.norm(target, dim=1)
        pred_n = pred / torch.clamp(pred_norm[:, None], min=float(cfg.eps))
        target_n = target / torch.clamp(target_norm[:, None], min=float(cfg.eps))
        cosine_vals = torch.sum(pred_n * target_n, dim=1)
        cosine = torch.mean(cosine_vals).item()
        cosine_loss = torch.mean(1.0 - cosine_vals).item()
    return {
        "model": model,
        "label_mapping": mapping,
        "device": dev,
        "summary": {
            "latent_train_mse_mean": float(mse),
            "latent_train_cosine_loss_mean": float(cosine_loss),
            "latent_train_pred_target_cosine_mean": float(cosine),
            "latent_pred_velocity_norm_mean": float(pred_norm.mean().item()),
            "latent_pred_velocity_norm_std": float(pred_norm.std(unbiased=False).item()),
            "latent_hidden_width": int(hidden_width),
            "latent_class_embedding_dim": int(emb_dim),
            "latent_hidden_layers": int(cfg.hidden_layers),
            "latent_lambda_cos": float(cfg.lambda_cos),
        },
    }


def _predict_latent_velocity(
    op: Dict[str, object],
    z: np.ndarray,
    y_value: int,
    r_s: np.ndarray,
    s_value: float,
) -> np.ndarray:
    model: _LatentResidualMLP = op["model"]
    mapping: Dict[int, int] = op["label_mapping"]
    dev: torch.device = op["device"]
    y_idx = mapping[int(y_value)]
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(np.asarray(z[None, :], dtype=np.float32)).to(dev),
            torch.from_numpy(np.asarray(r_s[None, :], dtype=np.float32)).to(dev),
            torch.tensor([float(s_value)], dtype=torch.float32, device=dev),
            torch.tensor([int(y_idx)], dtype=torch.long, device=dev),
        )
    return pred.cpu().numpy()[0].astype(np.float64)


def build_latent_residual_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    method: str,
) -> Dict[str, object]:
    if method not in LATENT_RESIDUAL_METHODS:
        raise ValueError(f"Unknown latent residual method: {method}")
    if str(getattr(args, "template_selection", "top_response")) != "top_response":
        raise ValueError("Latent Residual Flow is a direction operator and must not use template-selection modes.")
    cfg = LatentResidualConfig(
        flow_epochs=int(getattr(args, "latent_flow_epochs", 50)),
        flow_batch_size=int(getattr(args, "latent_flow_batch_size", 128)),
        flow_lr=float(getattr(args, "latent_flow_lr", 1e-3)),
        flow_weight_decay=float(getattr(args, "latent_flow_weight_decay", 1e-4)),
        hidden_layers=int(getattr(args, "latent_hidden_layers", 2)),
        hidden_width=int(getattr(args, "latent_hidden_width", 0)),
        class_embedding_dim=int(getattr(args, "latent_class_embedding_dim", 0)),
        lambda_cos=float(getattr(args, "latent_lambda_cos", 0.5)),
        rbf_tau_floor=float(getattr(args, "latent_rbf_tau_floor", 1e-12)),
    )
    Z = np.asarray(X_train_z, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.int64).ravel()
    target_out = build_latent_residual_targets(Z, y_arr, seed=seed, cfg=cfg)
    if method == "latent_residual_direct":
        op = None
        op_summary = {
            "latent_train_mse_mean": 0.0,
            "latent_train_cosine_loss_mean": 0.0,
            "latent_train_pred_target_cosine_mean": 1.0,
            "latent_pred_velocity_norm_mean": float(np.mean(target_out["target_velocity_norms"])),
            "latent_pred_velocity_norm_std": float(np.std(target_out["target_velocity_norms"])),
            "latent_hidden_width": 0,
            "latent_class_embedding_dim": 0,
            "latent_hidden_layers": 0,
            "latent_lambda_cos": float(cfg.lambda_cos),
        }
    else:
        op = fit_latent_residual_operator(
            Z,
            y_arr,
            target_out,
            seed=seed,
            device=str(getattr(args, "device", "cpu")),
            cfg=cfg,
        )
        op_summary = dict(op["summary"])

    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    margins = estimate_local_manifold_margins(Z, y_arr)
    eta_safe = None if bool(getattr(args, "disable_safe_step", False)) else float(getattr(args, "eta_safe", 0.75))
    gamma_requested = float(getattr(args, "pia_gamma", 0.1))
    multiplier = int(getattr(args, "multiplier", 10))
    rng = np.random.default_rng(int(seed) + 9293)
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []
    pred_cosines: List[float] = []

    for i in range(len(Z)):
        for c in range(multiplier):
            sample = _sample_residual(
                Z=Z,
                y=y_arr,
                sampler=target_out["sampler"],
                anchor_index=i,
                rng=rng,
                cfg=cfg,
            )
            residual = np.asarray(sample["residual"], dtype=np.float64)
            eps_vec, eps_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(cfg.eps))
            s_value = 1.0 if method == "latent_residual_direct" else 0.0
            if method == "latent_residual_direct":
                target_u = residual - eps_vec
                pred_u = target_u
                direction = residual
                pred_target_cos = 1.0
            else:
                target_u = residual - eps_vec
                pred_u = _predict_latent_velocity(op, Z[i], int(y_arr[i]), eps_vec, 0.0)
                r_hat = eps_vec + pred_u
                direction, _ = _unit(r_hat, eps=float(cfg.eps))
                pred_target_cos = _cosine(pred_u, target_u, eps=float(cfg.eps))
            if not np.isfinite(pred_target_cos):
                pred_target_cos = 0.0
            direction, direction_norm = _unit(direction, eps=float(cfg.eps))
            if direction_norm <= float(cfg.eps):
                direction, direction_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(cfg.eps))
            d_min = float(margins[i])
            if eta_safe is None:
                gamma_used = gamma_requested
                safe_upper_bound = float("inf")
                safe_radius_ratio = 1.0
            else:
                safe_upper_bound = float(eta_safe) * d_min / (direction_norm + cfg.eps)
                gamma_used = min(gamma_requested, safe_upper_bound)
                safe_radius = float(eta_safe) * d_min
                safe_radius_ratio = float(abs(gamma_used) * direction_norm / (safe_radius + cfg.eps)) if safe_radius > 0 else 0.0
            W_i = (gamma_used * direction).astype(np.float32)
            z_aug.append((Z[i] + W_i).astype(np.float32))
            y_aug.append(int(y_arr[i]))
            tid_aug.append(tid_arr[i])
            pred_cosines.append(float(pred_target_cos))
            rows.append(
                {
                    "anchor_index": int(i),
                    "tid": tid_arr[i],
                    "class_id": int(y_arr[i]),
                    "candidate_order": int(c),
                    "slot_index": int(len(rows)),
                    "direction_source": "latent_residual_flow_operator",
                    "template_id": -1,
                    "template_rank": -1,
                    "template_sign": np.nan,
                    "template_response_abs": np.nan,
                    "selected_template_rank": -1,
                    "selected_template_response_abs": np.nan,
                    "direction_id": -1,
                    "latent_s": float(s_value),
                    "latent_eps_norm": float(eps_norm),
                    "latent_target_index": int(sample["target_index"]),
                    "latent_target_class": int(sample["target_class"]),
                    "latent_target_dist": float(sample["target_dist"]),
                    "latent_target_sampling_prob": float(sample["target_sampling_prob"]),
                    "latent_residual_norm": float(sample["residual_norm"]),
                    "latent_pred_velocity_norm": float(np.linalg.norm(pred_u)),
                    "latent_pred_target_cosine": float(pred_target_cos),
                    "latent_fallback_flag": bool(sample["fallback_flag"]),
                    "latent_fallback_reason": str(sample["fallback_reason"]),
                    "gamma_requested": float(gamma_requested),
                    "gamma_used": float(gamma_used),
                    "gamma_used_ratio": float(gamma_used / gamma_requested) if abs(gamma_requested) > cfg.eps else np.nan,
                    "direction_norm": float(direction_norm),
                    "pre_safe_displacement_norm": float(abs(gamma_requested) * direction_norm),
                    "post_safe_displacement_norm": float(np.linalg.norm(W_i)),
                    "z_displacement_norm": float(np.linalg.norm(W_i)),
                    "safe_upper_bound": float(safe_upper_bound),
                    "safe_radius_ratio": float(safe_radius_ratio),
                    "manifold_margin": d_min,
                    "is_clipped": float(gamma_requested > safe_upper_bound + 1e-9),
                    "selection_stage": "latent_residual_flow_operator",
                    "selector_name": method,
                    "feasible_flag": 1.0,
                    "selector_accept_flag": 1.0,
                    "_direction_vec": direction,
                }
            )

    diversity = _direction_diversity(rows, eps=float(cfg.eps))
    for row in rows:
        row.pop("_direction_vec", None)
    latent_summary = {
        **dict(target_out["summary"]),
        **op_summary,
        **diversity,
        "latent_pred_target_cosine_mean": float(np.mean(pred_cosines)) if pred_cosines else np.nan,
        "latent_flow_epochs": int(cfg.flow_epochs),
        "latent_flow_batch_size": int(cfg.flow_batch_size),
        "latent_flow_lr": float(cfg.flow_lr),
        "latent_flow_weight_decay": float(cfg.flow_weight_decay),
        "latent_rbf_tau_floor": float(cfg.rbf_tau_floor),
    }
    direction_meta = {
        "bank_source": "latent_residual_flow_operator",
        "direction_source": "latent_residual_flow_operator",
        "operator_source": method,
        **latent_summary,
    }
    return materialize_z_aug_out(
        z_aug=np.stack(z_aug).astype(np.float32) if z_aug else np.empty((0, Z.shape[1]), dtype=np.float32),
        y_aug=np.asarray(y_aug, dtype=np.int64),
        tid_aug=np.asarray(tid_aug, dtype=object),
        audit_rows=rows,
        train_recs=train_recs,
        mean_log=mean_log,
        direction_bank_meta=direction_meta,
        effective_k=1,
        eta_safe=eta_safe,
        algo_name=method,
        engine_id=method,
        extra_meta={
            **latent_summary,
            "selection_stage": "latent_residual_flow_operator",
            "selector_name": method,
            "multi_template_pairs": 0,
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
        },
    )
