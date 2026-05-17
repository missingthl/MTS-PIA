from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.curriculum import estimate_local_manifold_margins
from core.pia import build_zpia_direction_bank

from .ag_pia import _effective_rank, _pairwise_cosine_mean, _unit
from .latent_residual_flow import _direction_diversity
from .materialize import materialize_z_aug_out
from .state import TrialRecord


SPG_PIA_METHODS = {
    "spg_pia_zhead",
    "spg_pia_zhead_deterministic",
    "ecl_spg_pia_zhead",
    "ecl_spg_pia_zhead_deterministic",
    "rn_ecl_spg_pia_zhead",
    "rn_ecl_spg_pia_zhead_deterministic",
    "gi_spg_pia_zhead",
}


@dataclass(frozen=True)
class SPGPIAConfig:
    zhead_epochs: int = 50
    zhead_hidden_dim: int = 0
    zhead_lr: float = 1e-3
    zhead_weight_decay: float = 1e-4
    batch_size: int = 128
    projection_ridge: float = 1e-6
    noise_sigma: float = 0.1
    eps: float = 1e-12


class _ZHeadMLP(nn.Module):
    def __init__(self, z_dim: int, n_classes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def fit_zhead_classifier(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    device: str,
    cfg: SPGPIAConfig,
) -> Dict[str, object]:
    Z = np.asarray(Z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).ravel()
    z_dim = int(Z.shape[1])
    n_classes = int(np.max(y)) + 1
    hidden_dim = int(cfg.zhead_hidden_dim) if int(cfg.zhead_hidden_dim) > 0 else max(64, 2 * z_dim)
    torch.manual_seed(int(seed) + 14101)
    np.random.seed(int(seed) + 14101)
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = _ZHeadMLP(z_dim=z_dim, n_classes=n_classes, hidden_dim=hidden_dim).to(dev)
    ds = TensorDataset(torch.from_numpy(Z), torch.from_numpy(y))
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 14101)
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.zhead_lr), weight_decay=float(cfg.zhead_weight_decay))
    losses: List[float] = []
    model.train()
    for _ in range(int(cfg.zhead_epochs)):
        for bz, by in loader:
            bz = bz.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)
            logits = model(bz)
            loss = nn.functional.cross_entropy(logits, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Z).to(dev))
        pred = logits.argmax(dim=1).detach().cpu().numpy()
    acc = float(np.mean(pred == y)) if y.size else np.nan
    return {
        "model": model,
        "device": dev,
        "spg_zhead_train_acc": acc,
        "spg_zhead_train_loss_mean": float(np.mean(losses)) if losses else np.nan,
        "spg_zhead_hidden_dim": hidden_dim,
    }


def _support_projector(bank: np.ndarray, ridge: float, eps: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    D = np.asarray(bank, dtype=np.float64)
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), eps)
    B = D.T  # [z_dim, k]
    gram = B.T @ B + float(ridge) * np.eye(B.shape[1], dtype=np.float64)
    gram_inv = np.linalg.pinv(gram)

    def project(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float64).ravel()
        return B @ (gram_inv @ (B.T @ vv))

    s = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(s > 1e-8))
    meta = {
        "spg_support_rank": rank,
        "spg_support_condition": float(s[0] / max(s[-1], eps)) if s.size else np.nan,
    }
    return project, B, meta


def _orthonormal_support_basis(raw_basis: np.ndarray, eps: float) -> np.ndarray:
    U, s, _ = np.linalg.svd(np.asarray(raw_basis, dtype=np.float64), full_matrices=False)
    rank = int(np.sum(s > 1e-8))
    if rank <= 0:
        return np.empty((raw_basis.shape[0], 0), dtype=np.float64)
    return U[:, :rank]


def compute_spg_gradients(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    zhead: Dict[str, object],
) -> np.ndarray:
    Z_np = np.asarray(Z, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.int64).ravel()
    model: nn.Module = zhead["model"]
    dev = zhead["device"]
    grads = np.zeros_like(Z_np, dtype=np.float64)
    model.eval()
    for i in range(len(Z_np)):
        z = torch.tensor(Z_np[i : i + 1], dtype=torch.float32, device=dev, requires_grad=True)
        yy = torch.tensor([int(y_np[i])], dtype=torch.long, device=dev)
        loss = nn.functional.cross_entropy(model(z), yy)
        grad = torch.autograd.grad(loss, z)[0]
        grads[i] = grad.detach().cpu().numpy()[0].astype(np.float64)
    return grads


def build_spg_pia_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    method: str,
) -> Dict[str, object]:
    if method not in SPG_PIA_METHODS:
        raise ValueError(f"Unknown SPG-PIA method: {method}")
    if str(getattr(args, "template_selection", "top_response")) != "top_response":
        raise ValueError("SPG-PIA must not use template-selection modes.")
    cfg = SPGPIAConfig(
        zhead_epochs=int(getattr(args, "spg_zhead_epochs", 50)),
        zhead_hidden_dim=int(getattr(args, "spg_zhead_hidden_dim", 0)),
        zhead_lr=float(getattr(args, "spg_zhead_lr", 1e-3)),
        zhead_weight_decay=float(getattr(args, "spg_zhead_weight_decay", 1e-4)),
        batch_size=int(getattr(args, "spg_zhead_batch_size", 128)),
        projection_ridge=float(getattr(args, "spg_projection_ridge", 1e-6)),
        noise_sigma=float(getattr(args, "spg_noise_sigma", 0.1)),
    )
    Z = np.asarray(X_train_z, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64).ravel()
    zhead = fit_zhead_classifier(Z, y, seed=seed, device=str(getattr(args, "device", "cpu")), cfg=cfg)
    grads = compute_spg_gradients(Z, y, zhead=zhead)
    bank, bank_meta = build_zpia_direction_bank(
        Z,
        k_dir=int(getattr(args, "k_dir", 10)),
        seed=seed,
        telm2_n_iters=int(getattr(args, "telm2_n_iters", 50)),
        telm2_c_repr=float(getattr(args, "telm2_c_repr", 10.0)),
        telm2_activation=str(getattr(args, "telm2_activation", "sine")),
        telm2_bias_update_mode=str(getattr(args, "telm2_bias_update_mode", "none")),
    )
    project, support_basis, support_meta = _support_projector(bank, ridge=float(cfg.projection_ridge), eps=float(cfg.eps))
    proj_grads = np.stack([project(g) for g in grads]).astype(np.float64)
    grad_norms = np.linalg.norm(grads, axis=1)
    proj_norms = np.linalg.norm(proj_grads, axis=1)
    projection_energy = proj_norms / np.maximum(grad_norms, float(cfg.eps))
    ecl_projection_energy = np.clip(
        (proj_norms * proj_norms) / np.maximum(grad_norms * grad_norms, float(cfg.eps)),
        0.0,
        1.0,
    )
    ecl_alpha = np.sqrt((1.0 - ecl_projection_energy) / np.maximum(ecl_projection_energy, float(cfg.eps)))
    ortho_basis = _orthonormal_support_basis(support_basis, eps=float(cfg.eps))
    rn_rank = int(ortho_basis.shape[1])
    rn_proj_grads = (ortho_basis @ (ortho_basis.T @ grads.T)).T if rn_rank > 0 else np.zeros_like(grads)
    rn_proj_norms = np.linalg.norm(rn_proj_grads, axis=1)
    rn_projection_energy = np.clip(
        (rn_proj_norms * rn_proj_norms) / np.maximum(grad_norms * grad_norms, float(cfg.eps)),
        0.0,
        1.0,
    )
    rn_alpha = np.sqrt(
        (1.0 - rn_projection_energy)
        / np.maximum(float(max(rn_rank - 1, 1)) * rn_projection_energy, float(cfg.eps))
    )

    eta_safe = None if bool(getattr(args, "disable_safe_step", False)) else float(getattr(args, "eta_safe", 0.75))
    gamma_requested = float(getattr(args, "pia_gamma", 0.1))
    margins = estimate_local_manifold_margins(Z, y)
    multiplier = int(getattr(args, "multiplier", 10))
    is_ecl = method.startswith("ecl_spg_pia_")
    is_rn = method.startswith("rn_ecl_spg_pia_")
    deterministic = method in {
        "spg_pia_zhead_deterministic",
        "ecl_spg_pia_zhead_deterministic",
        "rn_ecl_spg_pia_zhead_deterministic",
    }
    if is_rn:
        direction_source = "rank_normalized_ecl_spg_operator"
        selection_stage = "rank_normalized_ecl_spg_operator"
        source_space = "covariance_state_rank_normalized_ecl_spg"
    elif is_ecl:
        direction_source = "energy_calibrated_langevin_spg_operator"
        selection_stage = "energy_calibrated_langevin_spg_operator"
        source_space = "covariance_state_energy_calibrated_langevin_spg"
    else:
        direction_source = "support_projected_gradient_operator"
        selection_stage = "support_projected_gradient_operator"
        source_space = "covariance_state_support_projected_gradient"
    rng = np.random.default_rng(int(seed) + 14207)
    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []

    for i in range(len(Z)):
        active_proj = rn_proj_grads[i] if is_rn else proj_grads[i]
        active_proj_norm = rn_proj_norms[i] if is_rn else proj_norms[i]
        base_dir, base_norm = _unit(active_proj, eps=float(cfg.eps))
        if base_norm <= float(cfg.eps):
            base_dir, base_norm = _unit(grads[i], eps=float(cfg.eps))
        if base_norm <= float(cfg.eps):
            base_dir, base_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(cfg.eps))
        for c in range(multiplier):
            support_noise_norm = 0.0
            rn_fallback_flag = 0.0
            rn_fallback_reason = ""
            if deterministic:
                direction = base_dir
            else:
                if is_rn:
                    if rn_rank <= 1 or rn_projection_energy[i] <= float(cfg.eps):
                        direction = base_dir
                        rn_fallback_flag = 1.0
                        rn_fallback_reason = "rank_or_projection_energy_too_low"
                    else:
                        u_B = ortho_basis.T @ base_dir
                        u_B, _ = _unit(u_B, eps=float(cfg.eps))
                        xi = rng.normal(size=(rn_rank,))
                        xi_perp = xi - float(np.dot(xi, u_B)) * u_B
                        xi_perp, xi_norm = _unit(xi_perp, eps=float(cfg.eps))
                        if xi_norm <= float(cfg.eps):
                            direction = base_dir
                            rn_fallback_flag = 1.0
                            rn_fallback_reason = "zero_perpendicular_noise"
                        else:
                            support_noise_norm = float(np.linalg.norm(ortho_basis @ xi_perp))
                            c_vec, _ = _unit(u_B + float(rn_alpha[i]) * xi_perp, eps=float(cfg.eps))
                            direction, _ = _unit(ortho_basis @ c_vec, eps=float(cfg.eps))
                elif is_ecl:
                    xi = rng.normal(size=(support_basis.shape[1],))
                    xi_dir, xi_norm = _unit(xi, eps=float(cfg.eps))
                    if xi_norm <= float(cfg.eps):
                        zeta = np.zeros_like(base_dir)
                    else:
                        zeta = support_basis @ xi_dir
                    support_noise_norm = float(np.linalg.norm(zeta))
                    direction, _ = _unit(base_dir + float(ecl_alpha[i]) * zeta, eps=float(cfg.eps))
                else:
                    noise = project(rng.normal(size=(Z.shape[1],)))
                    noise_dir, noise_norm = _unit(noise, eps=float(cfg.eps))
                    if noise_norm <= float(cfg.eps):
                        noise_dir = np.zeros_like(base_dir)
                    support_noise_norm = float(noise_norm)
                    direction, _ = _unit(base_dir + float(cfg.noise_sigma) * noise_dir, eps=float(cfg.eps))
            direction, direction_norm = _unit(direction, eps=float(cfg.eps))
            align_to_grad = float(np.dot(direction, base_dir) / max(np.linalg.norm(direction) * np.linalg.norm(base_dir), cfg.eps))
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
            y_aug.append(int(y[i]))
            tid_aug.append(tid_arr[i])
            rows.append(
                {
                    "anchor_index": int(i),
                    "tid": tid_arr[i],
                    "class_id": int(y[i]),
                    "candidate_order": int(c),
                    "slot_index": int(len(rows)),
                    "direction_source": direction_source,
                    "template_id": -1,
                    "template_rank": -1,
                    "template_sign": np.nan,
                    "template_response_abs": np.nan,
                    "selected_template_rank": -1,
                    "selected_template_response_abs": np.nan,
                    "direction_id": -1,
                    "spg_grad_norm": float(grad_norms[i]),
                    "spg_projected_grad_norm": float(active_proj_norm),
                    "spg_projection_energy": float(rn_projection_energy[i] if is_rn else projection_energy[i]),
                    "ecl_projection_energy": float(ecl_projection_energy[i]) if is_ecl else np.nan,
                    "ecl_alpha": float(ecl_alpha[i]) if is_ecl else np.nan,
                    "ecl_alignment_to_projected_gradient": align_to_grad if is_ecl else np.nan,
                    "ecl_support_noise_norm": support_noise_norm if is_ecl else np.nan,
                    "ecl_fallback_flag": 0.0 if is_ecl else np.nan,
                    "ecl_fallback_reason": "" if is_ecl else "",
                    "rn_ecl_projection_energy": float(rn_projection_energy[i]) if is_rn else np.nan,
                    "rn_ecl_alpha": float(rn_alpha[i]) if is_rn else np.nan,
                    "rn_ecl_alignment_to_projected_gradient": align_to_grad if is_rn else np.nan,
                    "rn_ecl_support_noise_norm": support_noise_norm if is_rn else np.nan,
                    "rn_ecl_fallback_flag": rn_fallback_flag if is_rn else np.nan,
                    "rn_ecl_fallback_reason": rn_fallback_reason if is_rn else "",
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
                    "selection_stage": selection_stage,
                    "selector_name": method,
                    "feasible_flag": 1.0,
                    "selector_accept_flag": 1.0,
                    "_direction_vec": direction,
                }
            )

    diversity = _direction_diversity(rows, eps=float(cfg.eps))
    for row in rows:
        row.pop("_direction_vec", None)
    gamma_ratios = [float(row.get("gamma_used_ratio", np.nan)) for row in rows]
    pre_safe_norms = [float(row.get("pre_safe_displacement_norm", np.nan)) for row in rows]
    post_safe_norms = [float(row.get("post_safe_displacement_norm", np.nan)) for row in rows]
    ecl_alignments = [float(row.get("ecl_alignment_to_projected_gradient", np.nan)) for row in rows]
    ecl_noise_norms = [float(row.get("ecl_support_noise_norm", np.nan)) for row in rows]
    rn_alignments = [float(row.get("rn_ecl_alignment_to_projected_gradient", np.nan)) for row in rows]
    rn_noise_norms = [float(row.get("rn_ecl_support_noise_norm", np.nan)) for row in rows]
    rn_fallback_flags = [float(row.get("rn_ecl_fallback_flag", np.nan)) for row in rows]
    summary = {
        **dict(zhead),
        **support_meta,
        **diversity,
        "spg_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms.size else np.nan,
        "spg_grad_norm_std": float(np.std(grad_norms)) if grad_norms.size else np.nan,
        "spg_projected_grad_norm_mean": float(np.mean(rn_proj_norms if is_rn else proj_norms)),
        "spg_projected_grad_norm_std": float(np.std(rn_proj_norms if is_rn else proj_norms)),
        "spg_projection_energy": float(np.mean(rn_projection_energy if is_rn else projection_energy)),
        "spg_projection_energy_std": float(np.std(rn_projection_energy if is_rn else projection_energy)),
        "spg_direction_pairwise_cosine_mean": float(diversity.get("latent_generated_direction_pairwise_cosine_mean", np.nan)),
        "spg_effective_aug_multiplier": float(diversity.get("latent_effective_aug_multiplier", 0.0)),
        "ecl_projection_energy_mean": float(np.mean(ecl_projection_energy)) if is_ecl and ecl_projection_energy.size else np.nan,
        "ecl_projection_energy_std": float(np.std(ecl_projection_energy)) if is_ecl and ecl_projection_energy.size else np.nan,
        "ecl_alpha_mean": float(np.mean(ecl_alpha)) if is_ecl and ecl_alpha.size else np.nan,
        "ecl_alpha_std": float(np.std(ecl_alpha)) if is_ecl and ecl_alpha.size else np.nan,
        "ecl_alignment_to_projected_gradient_mean": float(np.nanmean(ecl_alignments)) if is_ecl and ecl_alignments else np.nan,
        "ecl_direction_pairwise_cosine_mean": (
            float(diversity.get("latent_generated_direction_pairwise_cosine_mean", np.nan)) if is_ecl else np.nan
        ),
        "ecl_effective_aug_multiplier": float(diversity.get("latent_effective_aug_multiplier", 0.0)) if is_ecl else np.nan,
        "ecl_support_rank": int(support_meta.get("spg_support_rank", 0)) if is_ecl else np.nan,
        "ecl_support_noise_norm_mean": float(np.nanmean(ecl_noise_norms)) if is_ecl and ecl_noise_norms else np.nan,
        "ecl_support_noise_norm_std": float(np.nanstd(ecl_noise_norms)) if is_ecl and ecl_noise_norms else np.nan,
        "ecl_fallback_rate": 0.0 if is_ecl else np.nan,
        "rn_ecl_projection_energy_mean": float(np.mean(rn_projection_energy)) if is_rn and rn_projection_energy.size else np.nan,
        "rn_ecl_projection_energy_std": float(np.std(rn_projection_energy)) if is_rn and rn_projection_energy.size else np.nan,
        "rn_ecl_alpha_mean": float(np.mean(rn_alpha)) if is_rn and rn_alpha.size else np.nan,
        "rn_ecl_alpha_std": float(np.std(rn_alpha)) if is_rn and rn_alpha.size else np.nan,
        "rn_ecl_direction_pairwise_cosine_mean": (
            float(diversity.get("latent_generated_direction_pairwise_cosine_mean", np.nan)) if is_rn else np.nan
        ),
        "rn_ecl_alignment_to_projected_gradient_mean": float(np.nanmean(rn_alignments)) if is_rn and rn_alignments else np.nan,
        "rn_ecl_effective_aug_multiplier": float(diversity.get("latent_effective_aug_multiplier", 0.0)) if is_rn else np.nan,
        "rn_ecl_support_rank": int(rn_rank) if is_rn else np.nan,
        "rn_ecl_support_noise_norm_mean": float(np.nanmean(rn_noise_norms)) if is_rn and rn_noise_norms else np.nan,
        "rn_ecl_support_noise_norm_std": float(np.nanstd(rn_noise_norms)) if is_rn and rn_noise_norms else np.nan,
        "rn_ecl_fallback_rate": float(np.nanmean(rn_fallback_flags)) if is_rn and rn_fallback_flags else np.nan,
        "gamma_used_ratio_mean": float(np.nanmean(gamma_ratios)) if gamma_ratios else np.nan,
        "pre_safe_displacement_norm_mean": float(np.nanmean(pre_safe_norms)) if pre_safe_norms else np.nan,
        "post_safe_displacement_norm_mean": float(np.nanmean(post_safe_norms)) if post_safe_norms else np.nan,
        "spg_projection_ridge": float(cfg.projection_ridge),
        "spg_noise_sigma": float(cfg.noise_sigma),
        "spg_zhead_epochs": int(cfg.zhead_epochs),
    }
    # Drop model/device objects before passing through CSV-oriented metadata.
    summary.pop("model", None)
    summary.pop("device", None)
    direction_meta = {
        "bank_source": direction_source,
        "direction_source": direction_source,
        "operator_source": method,
        "source_space": source_space,
        "zpia_meta": bank_meta,
        **summary,
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
            **summary,
            "selection_stage": selection_stage,
            "selector_name": method,
            "multi_template_pairs": 0,
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
        },
    )


# ---------------------------------------------------------------------------
# GI-SPG-PIA: Generalized-Inverse Support-Projected Gradient PIA
# ---------------------------------------------------------------------------


def fit_gi_spg_operator(
    Z: np.ndarray,
    G: np.ndarray,
    *,
    hidden_dim: int,
    ridge: float,
    seed: int,
    eps: float,
    activation: str = "tanh",
) -> Dict[str, object]:
    """Fit a random-feature ELM operator: Phi(Z) Beta ≈ G.

    Z : [n, z_dim]  input latent vectors
    G : [n, z_dim]  normalised support-projected task gradient targets
    Returns a dict with 'W', 'b', 'Beta', and training diagnostics.
    """
    Z = np.asarray(Z, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    n, z_dim = Z.shape
    rng_op = np.random.default_rng(int(seed) + 29317)
    W = rng_op.standard_normal(size=(hidden_dim, z_dim)).astype(np.float64)
    b = rng_op.standard_normal(size=(hidden_dim,)).astype(np.float64)
    H = Z @ W.T + b[np.newaxis, :]  # [n, hidden_dim]
    if activation == "tanh":
        H = np.tanh(H)
    elif activation == "sigmoid":
        H = 1.0 / (1.0 + np.exp(-H))
    # else: linear / none
    # Ridge solve: Beta = (H^T H + ridge * I)^{-1} H^T G
    A = H.T @ H + float(ridge) * np.eye(hidden_dim, dtype=np.float64)
    rhs = H.T @ G  # [hidden_dim, z_dim]
    Beta = np.linalg.solve(A, rhs)  # [hidden_dim, z_dim]
    G_pred = H @ Beta  # [n, z_dim]
    mse = float(np.mean((G_pred - G) ** 2))
    # Per-sample cosine between predicted and target direction
    g_norms = np.linalg.norm(G, axis=1)
    p_norms = np.linalg.norm(G_pred, axis=1)
    cos_vals = np.einsum("ij,ij->i", G, G_pred) / np.maximum(g_norms * p_norms, eps)
    train_cosine = float(np.mean(cos_vals))
    return {
        "W": W,
        "b": b,
        "Beta": Beta,
        "activation": activation,
        "gi_spg_operator_train_mse": mse,
        "gi_spg_operator_train_cosine": train_cosine,
        "gi_spg_hidden_dim": hidden_dim,
        "gi_spg_ridge": float(ridge),
    }


def _predict_gi_spg_direction(
    op: Dict[str, object],
    z: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Predict normalised direction for a single latent vector z."""
    z = np.asarray(z, dtype=np.float64).ravel()
    h = z @ op["W"].T + op["b"]  # [hidden_dim]
    act = op["activation"]
    if act == "tanh":
        h = np.tanh(h)
    elif act == "sigmoid":
        h = 1.0 / (1.0 + np.exp(-h))
    pred = h @ op["Beta"]  # [z_dim]
    pred_dir, _ = _unit(pred, eps=eps)
    return pred_dir


def build_gi_spg_pia_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    method: str,
) -> Dict[str, object]:
    """GI-SPG-PIA augmentation builder.

    1. Fit a Z-head classifier (reused from SPG).
    2. Compute support-projected task gradients G_i = P_S ∇_z CE.
    3. Fit an ELM operator Phi(Z) Beta ≈ G via ridge regression.
    4. Predict direction d_i = normalize(phi(z_i) Beta) for each anchor.
    5. Generate z_aug = z_i + gamma_used * d_i via existing safe-step + bridge.
    """
    if method not in {"gi_spg_pia_zhead"}:
        raise ValueError(f"Unknown GI-SPG-PIA method: {method}")
    if str(getattr(args, "template_selection", "top_response")) != "top_response":
        raise ValueError("GI-SPG-PIA must not use template-selection modes.")

    cfg = SPGPIAConfig(
        zhead_epochs=int(getattr(args, "spg_zhead_epochs", 50)),
        zhead_hidden_dim=int(getattr(args, "spg_zhead_hidden_dim", 0)),
        zhead_lr=float(getattr(args, "spg_zhead_lr", 1e-3)),
        zhead_weight_decay=float(getattr(args, "spg_zhead_weight_decay", 1e-4)),
        batch_size=int(getattr(args, "spg_zhead_batch_size", 128)),
        projection_ridge=float(getattr(args, "spg_projection_ridge", 1e-6)),
    )
    gi_hidden_dim_arg = int(getattr(args, "gi_spg_hidden_dim", 0))
    gi_ridge = float(getattr(args, "gi_spg_ridge", 1e-3))
    gi_activation = str(getattr(args, "gi_spg_activation", "tanh"))

    Z = np.asarray(X_train_z, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64).ravel()
    z_dim = int(Z.shape[1])
    gi_hidden_dim = gi_hidden_dim_arg if gi_hidden_dim_arg > 0 else max(64, 2 * z_dim)

    # --- Step 1: fit Z-head classifier ---
    zhead = fit_zhead_classifier(
        Z, y,
        seed=seed,
        device=str(getattr(args, "device", "cpu")),
        cfg=cfg,
    )
    # --- Step 2: compute raw task gradients ---
    grads = compute_spg_gradients(Z, y, zhead=zhead)
    grad_norms = np.linalg.norm(grads, axis=1)

    # --- Step 3: build support basis and project gradients ---
    bank, bank_meta = build_zpia_direction_bank(
        Z,
        k_dir=int(getattr(args, "k_dir", 10)),
        seed=seed,
        telm2_n_iters=int(getattr(args, "telm2_n_iters", 50)),
        telm2_c_repr=float(getattr(args, "telm2_c_repr", 10.0)),
        telm2_activation=str(getattr(args, "telm2_activation", "sine")),
        telm2_bias_update_mode=str(getattr(args, "telm2_bias_update_mode", "none")),
    )
    ortho_basis = _orthonormal_support_basis(bank.T, eps=float(cfg.eps))  # bank is [k, z_dim]
    gi_rank = int(ortho_basis.shape[1])
    # G_i = P_S g_i = B B^T g_i
    if gi_rank > 0:
        proj_grads = (ortho_basis @ (ortho_basis.T @ grads.T)).T  # [n, z_dim]
    else:
        proj_grads = grads.copy()
    proj_norms = np.linalg.norm(proj_grads, axis=1)
    projection_energy = np.clip(
        (proj_norms ** 2) / np.maximum(grad_norms ** 2, float(cfg.eps)),
        0.0, 1.0,
    )

    # Normalise targets for the ELM (direction targets, not magnitude)
    G_targets = np.zeros_like(proj_grads)
    for i in range(len(proj_grads)):
        G_targets[i], _ = _unit(proj_grads[i], eps=float(cfg.eps))

    # --- Step 4: fit ELM operator ---
    op = fit_gi_spg_operator(
        Z, G_targets,
        hidden_dim=gi_hidden_dim,
        ridge=gi_ridge,
        seed=seed,
        eps=float(cfg.eps),
        activation=gi_activation,
    )

    # --- Step 5: predict directions for all anchors ---
    pred_dirs = np.zeros((len(Z), z_dim), dtype=np.float64)
    pred_norms_raw = np.zeros(len(Z), dtype=np.float64)
    for i in range(len(Z)):
        h = Z[i] @ op["W"].T + op["b"]
        act = op["activation"]
        if act == "tanh":
            h = np.tanh(h)
        elif act == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-h))
        pred_raw = h @ op["Beta"]
        pred_norms_raw[i] = float(np.linalg.norm(pred_raw))
        pred_dirs[i], _ = _unit(pred_raw, eps=float(cfg.eps))

    # Pred-target cosine per sample
    gt_norms = np.linalg.norm(G_targets, axis=1)
    pd_norms = np.linalg.norm(pred_dirs, axis=1)
    cos_pt = np.einsum("ij,ij->i", G_targets, pred_dirs) / np.maximum(gt_norms * pd_norms, float(cfg.eps))

    eta_safe = None if bool(getattr(args, "disable_safe_step", False)) else float(getattr(args, "eta_safe", 0.75))
    gamma_requested = float(getattr(args, "pia_gamma", 0.1))
    margins = estimate_local_manifold_margins(Z, y)
    multiplier = int(getattr(args, "multiplier", 10))

    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []

    direction_source = "generalized_inverse_spg_operator"
    selection_stage = "generalized_inverse_spg_operator"
    source_space = "covariance_state_generalized_inverse_spg"

    for i in range(len(Z)):
        direction = pred_dirs[i]
        direction, direction_norm = _unit(direction, eps=float(cfg.eps))
        # Fallback: if prediction is degenerate, use raw projected gradient
        if direction_norm <= float(cfg.eps):
            direction, direction_norm = _unit(proj_grads[i], eps=float(cfg.eps))
        if direction_norm <= float(cfg.eps):
            direction, direction_norm = _unit(grads[i], eps=float(cfg.eps))
        d_min = float(margins[i])
        if eta_safe is None:
            gamma_used = gamma_requested
            safe_upper_bound = float("inf")
            safe_radius_ratio = 1.0
        else:
            safe_upper_bound = float(eta_safe) * d_min / (direction_norm + cfg.eps)
            gamma_used = min(gamma_requested, safe_upper_bound)
            safe_radius = float(eta_safe) * d_min
            safe_radius_ratio = (
                float(abs(gamma_used) * direction_norm / (safe_radius + cfg.eps))
                if safe_radius > 0 else 0.0
            )
        for c in range(multiplier):
            W_i = (gamma_used * direction).astype(np.float32)
            z_aug.append((Z[i] + W_i).astype(np.float32))
            y_aug.append(int(y[i]))
            tid_aug.append(tid_arr[i])
            rows.append(
                {
                    "anchor_index": int(i),
                    "tid": tid_arr[i],
                    "class_id": int(y[i]),
                    "candidate_order": int(c),
                    "slot_index": int(len(rows)),
                    "direction_source": direction_source,
                    "template_id": -1,
                    "template_rank": -1,
                    "template_sign": np.nan,
                    "template_response_abs": np.nan,
                    "selected_template_rank": -1,
                    "selected_template_response_abs": np.nan,
                    "direction_id": -1,
                    "gi_spg_grad_norm": float(grad_norms[i]),
                    "gi_spg_projected_grad_norm": float(proj_norms[i]),
                    "gi_spg_projection_energy": float(projection_energy[i]),
                    "gi_spg_pred_target_cosine": float(cos_pt[i]),
                    "gi_spg_pred_norm": float(pred_norms_raw[i]),
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
                    "selection_stage": selection_stage,
                    "selector_name": method,
                    "feasible_flag": 1.0,
                    "selector_accept_flag": 1.0,
                    "_direction_vec": direction,
                }
            )

    diversity = _direction_diversity(rows, eps=float(cfg.eps))
    for row in rows:
        row.pop("_direction_vec", None)

    gamma_ratios = [float(r.get("gamma_used_ratio", np.nan)) for r in rows]
    pre_safe_norms = [float(r.get("pre_safe_displacement_norm", np.nan)) for r in rows]
    post_safe_norms = [float(r.get("post_safe_displacement_norm", np.nan)) for r in rows]

    summary = {
        **dict(zhead),
        **bank_meta,
        **op,
        **diversity,
        "gi_spg_support_rank": int(gi_rank),
        "gi_spg_projection_energy_mean": float(np.mean(projection_energy)),
        "gi_spg_projection_energy_std": float(np.std(projection_energy)),
        "gi_spg_target_norm_mean": float(np.mean(np.linalg.norm(G_targets, axis=1))),
        "gi_spg_target_norm_std": float(np.std(np.linalg.norm(G_targets, axis=1))),
        "gi_spg_pred_norm_mean": float(np.mean(pred_norms_raw)),
        "gi_spg_pred_norm_std": float(np.std(pred_norms_raw)),
        "gi_spg_pred_target_cosine_mean": float(np.mean(cos_pt)),
        "gi_spg_direction_pairwise_cosine_mean": float(
            diversity.get("latent_generated_direction_pairwise_cosine_mean", np.nan)
        ),
        "gi_spg_effective_aug_multiplier": float(
            diversity.get("latent_effective_aug_multiplier", 0.0)
        ),
        "gi_spg_zhead_train_acc": float(zhead.get("spg_zhead_train_acc", np.nan)),
        "gamma_used_ratio_mean": float(np.nanmean(gamma_ratios)) if gamma_ratios else np.nan,
        "pre_safe_displacement_norm_mean": float(np.nanmean(pre_safe_norms)) if pre_safe_norms else np.nan,
        "post_safe_displacement_norm_mean": float(np.nanmean(post_safe_norms)) if post_safe_norms else np.nan,
        "gi_spg_hidden_dim": int(gi_hidden_dim),
        "gi_spg_ridge": float(gi_ridge),
        "gi_spg_activation": str(gi_activation),
        "spg_zhead_epochs": int(cfg.zhead_epochs),
    }
    # Drop un-serialisable objects
    summary.pop("model", None)
    summary.pop("device", None)
    summary.pop("W", None)
    summary.pop("b", None)
    summary.pop("Beta", None)

    direction_meta = {
        "bank_source": direction_source,
        "direction_source": direction_source,
        "operator_source": method,
        "source_space": source_space,
        "zpia_meta": bank_meta,
        **summary,
    }
    return materialize_z_aug_out(
        z_aug=np.stack(z_aug).astype(np.float32) if z_aug else np.empty((0, z_dim), dtype=np.float32),
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
            **summary,
            "selection_stage": selection_stage,
            "selector_name": method,
            "multi_template_pairs": 0,
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
        },
    )
