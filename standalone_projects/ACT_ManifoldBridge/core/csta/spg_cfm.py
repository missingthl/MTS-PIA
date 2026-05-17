import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.curriculum import estimate_local_manifold_margins

from .latent_residual_flow import LatentResidualConfig, build_latent_residual_targets
from .materialize import materialize_z_aug_out
from .spg_pia import SPGPIAConfig, compute_spg_gradients, fit_zhead_classifier
from .state import TrialRecord

SPG_CFM_METHODS = {"spg_cfm_one_step", "spg_cfm_k3", "spg_cfm_film_one_step", "spg_cfm_align_one_step"}


@dataclass(frozen=True)
class SPGCFMConfig:
    flow_epochs: int = 50
    flow_batch_size: int = 128
    flow_lr: float = 1e-3
    flow_weight_decay: float = 1e-4
    hidden_layers: int = 2
    hidden_width: int = 0
    class_embedding_dim: int = 0
    lambda_cos: float = 0.5
    lambda_align: float = 0.0
    injection_mode: str = "concat"
    eps: float = 1e-12

    # SPG Z-head params
    zhead_epochs: int = 50
    zhead_hidden_dim: int = 0
    zhead_lr: float = 1e-3
    zhead_weight_decay: float = 1e-4
    zhead_batch_size: int = 128
    projection_ridge: float = 1e-6


class _FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.mod = nn.Linear(cond_dim, hidden_dim * 2)
        # Initialize to identity: gamma=1 (shift=0), beta=0
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.mod(cond)
        gamma_shift, beta = params.chunk(2, dim=-1)
        return h * (1.0 + gamma_shift) + beta


class _SPGCFMFiLMMLP(nn.Module):
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
        # State: [z_i, class_emb, r_t, t]
        state_dim = z_dim + class_embedding_dim + z_dim + 1
        # Condition: [s_hat_i, E_i]
        cond_dim = z_dim + 1

        self.input_proj = nn.Linear(state_dim, hidden_width)
        self.layers = nn.ModuleList()
        for _ in range(max(1, int(hidden_layers))):
            self.layers.append(nn.ModuleDict({
                "film": _FiLMLayer(cond_dim, hidden_width),
                "act": nn.GELU(),
                "dense": nn.Linear(hidden_width, hidden_width)
            }))
        self.output_head = nn.Linear(hidden_width, z_dim)

    def forward(
        self, z: torch.Tensor, y: torch.Tensor, s_hat: torch.Tensor, E: torch.Tensor, r_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedding(y)
        if s_hat.ndim == 1: s_hat = s_hat[:, None]
        if E.ndim == 1: E = E[:, None]
        if t.ndim == 1: t = t[:, None]

        state = torch.cat([z, emb, r_t, t], dim=1)
        cond = torch.cat([s_hat, E], dim=1)

        h = self.input_proj(state)
        for layer in self.layers:
            h = layer["film"](h, cond)
            h = layer["act"](h)
            h = layer["dense"](h)
        return self.output_head(h)


class _SPGCFMMLP(nn.Module):
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
        # Inputs: [z_i, class_emb, s_hat_i, E_i, r_t, t]
        # Dimensions: z_dim + class_embedding_dim + z_dim + 1 + z_dim + 1
        in_dim = z_dim + class_embedding_dim + z_dim + 1 + z_dim + 1
        layers: List[nn.Module] = []
        for idx in range(max(1, int(hidden_layers))):
            layers.append(nn.Linear(in_dim if idx == 0 else hidden_width, hidden_width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_width, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, z: torch.Tensor, y: torch.Tensor, s_hat: torch.Tensor, E: torch.Tensor, r_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedding(y)
        if s_hat.ndim == 1:
            s_hat = s_hat[:, None]
        if E.ndim == 1:
            E = E[:, None]
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([z, emb, s_hat, E, r_t, t], dim=1))


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


def _pairwise_cosine_mean(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] <= 1:
        return np.nan
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    sims = Xn @ Xn.T
    iu = np.triu_indices(X.shape[0], k=1)
    return float(np.mean(sims[iu])) if len(iu[0]) else np.nan


def _class_remap(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    labels = sorted(int(v) for v in np.unique(y))
    mapping = {label: idx for idx, label in enumerate(labels)}
    return np.asarray([mapping[int(v)] for v in y], dtype=np.int64), mapping


def fit_spg_cfm_operator(
    Z: np.ndarray,
    y: np.ndarray,
    target_out: Dict[str, object],
    s_hat: np.ndarray,
    E: np.ndarray,
    *,
    seed: int,
    device: str,
    cfg: SPGCFMConfig,
) -> Dict[str, object]:
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    y_mapped, mapping = _class_remap(y)
    _, z_dim = Z.shape
    n_classes = len(mapping)
    hidden_width = int(cfg.hidden_width) if int(cfg.hidden_width) > 0 else max(128, 2 * int(z_dim))
    emb_dim = int(cfg.class_embedding_dim) if int(cfg.class_embedding_dim) > 0 else min(16, max(4, n_classes))
    torch.manual_seed(int(seed) + 5551)
    np.random.seed(int(seed) + 5551)
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    if str(cfg.injection_mode) == "film":
        model = _SPGCFMFiLMMLP(
            z_dim=z_dim,
            n_classes=n_classes,
            class_embedding_dim=emb_dim,
            hidden_width=hidden_width,
            hidden_layers=int(cfg.hidden_layers),
        ).to(dev)
    else:
        model = _SPGCFMMLP(
            z_dim=z_dim,
            n_classes=n_classes,
            class_embedding_dim=emb_dim,
            hidden_width=hidden_width,
            hidden_layers=int(cfg.hidden_layers),
        ).to(dev)

    ds = TensorDataset(
        torch.from_numpy(np.asarray(Z, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_mapped, dtype=np.int64)),
        torch.from_numpy(np.asarray(s_hat, dtype=np.float32)),
        torch.from_numpy(np.asarray(E, dtype=np.float32)),
        torch.from_numpy(np.asarray(target_out["r_s"], dtype=np.float32)),
        torch.from_numpy(np.asarray(target_out["s_train"], dtype=np.float32)),
        torch.from_numpy(np.asarray(target_out["u_target"], dtype=np.float32)),
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed) + 5551)
    loader = DataLoader(ds, batch_size=int(cfg.flow_batch_size), shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.flow_lr), weight_decay=float(cfg.flow_weight_decay))

    model.train()
    for _ in range(int(cfg.flow_epochs)):
        for z_b, y_b, s_hat_b, E_b, r_b, t_b, u_b in loader:
            z_b = z_b.to(dev)
            y_b = y_b.to(dev)
            s_hat_b = s_hat_b.to(dev)
            E_b = E_b.to(dev)
            r_b = r_b.to(dev)
            t_b = t_b.to(dev)
            u_b = u_b.to(dev)

            pred = model(z_b, y_b, s_hat_b, E_b, r_b, t_b)
            mse = torch.mean((pred - u_b) ** 2)

            pred_n = pred / torch.clamp(torch.linalg.norm(pred, dim=1, keepdim=True), min=float(cfg.eps))
            u_n = u_b / torch.clamp(torch.linalg.norm(u_b, dim=1, keepdim=True), min=float(cfg.eps))
            cos_loss = torch.mean(1.0 - torch.sum(pred_n * u_n, dim=1))

            loss = mse + float(cfg.lambda_cos) * cos_loss

            if float(cfg.lambda_align) > 0:
                # Alignment Loss: align pred with s_hat_b
                # pred_n is already computed above
                # s_hat_b is unit vector
                loss_align = - torch.mean(torch.sum(pred_n * s_hat_b, dim=1))
                loss = loss + float(cfg.lambda_align) * loss_align

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        z_all = torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(dev)
        y_all = torch.from_numpy(y_mapped.astype(np.int64)).to(dev)
        s_hat_all = torch.from_numpy(np.asarray(s_hat, dtype=np.float32)).to(dev)
        E_all = torch.from_numpy(np.asarray(E, dtype=np.float32)).to(dev)
        r_all = torch.from_numpy(np.asarray(target_out["r_s"], dtype=np.float32)).to(dev)
        t_all = torch.from_numpy(np.asarray(target_out["s_train"], dtype=np.float32)).to(dev)
        target = torch.from_numpy(np.asarray(target_out["u_target"], dtype=np.float32)).to(dev)

        pred = model(z_all, y_all, s_hat_all, E_all, r_all, t_all)
        mse = torch.mean((pred - target) ** 2).item()
        
        pred_norm = torch.linalg.norm(pred, dim=1)
        target_norm = torch.linalg.norm(target, dim=1)
        pred_n = pred / torch.clamp(pred_norm[:, None], min=float(cfg.eps))
        target_n = target / torch.clamp(target_norm[:, None], min=float(cfg.eps))
        cosine_vals = torch.sum(pred_n * target_n, dim=1)
        cosine = torch.mean(cosine_vals).item()
        
    return {
        "model": model,
        "label_mapping": mapping,
        "device": dev,
        "summary": {
            "spg_cfm_train_mse_mean": float(mse),
            "spg_cfm_train_cosine_mean": float(cosine),
        },
    }


def _predict_spg_cfm_velocity(
    op: Dict[str, object],
    z: np.ndarray,
    y_value: int,
    s_hat_value: np.ndarray,
    E_value: float,
    eps_vec: np.ndarray,
    t_value: float,
) -> np.ndarray:
    model: nn.Module = op["model"]
    mapping: Dict[int, int] = op["label_mapping"]
    dev: torch.device = op["device"]
    y_idx = mapping[int(y_value)]
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(np.asarray(z[None, :], dtype=np.float32)).to(dev),
            torch.tensor([int(y_idx)], dtype=torch.long, device=dev),
            torch.from_numpy(np.asarray(s_hat_value[None, :], dtype=np.float32)).to(dev),
            torch.tensor([float(E_value)], dtype=torch.float32, device=dev),
            torch.from_numpy(np.asarray(eps_vec[None, :], dtype=np.float32)).to(dev),
            torch.tensor([float(t_value)], dtype=torch.float32, device=dev),
        )
    return pred.cpu().numpy()[0].astype(np.float64)


def build_spg_cfm_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    method: str,
) -> Dict[str, object]:
    t_start_build = time.perf_counter()
    if method not in SPG_CFM_METHODS:
        raise ValueError(f"Unknown SPG-CFM method: {method}")

    cfg = SPGCFMConfig(
        flow_epochs=int(getattr(args, "spg_cfm_flow_epochs", 50)),
        flow_batch_size=int(getattr(args, "spg_cfm_flow_batch_size", 128)),
        flow_lr=float(getattr(args, "spg_cfm_flow_lr", 1e-3)),
        flow_weight_decay=float(getattr(args, "spg_cfm_flow_weight_decay", 1e-4)),
        hidden_layers=int(getattr(args, "spg_cfm_hidden_layers", 2)),
        hidden_width=int(getattr(args, "spg_cfm_hidden_width", 0)),
        class_embedding_dim=int(getattr(args, "spg_cfm_class_embedding_dim", 0)),
        lambda_cos=float(getattr(args, "spg_cfm_lambda_cos", 0.5)),
        lambda_align=float(getattr(args, "spg_cfm_lambda_align", 0.15 if method == "spg_cfm_align_one_step" else 0.0)),
        zhead_epochs=int(getattr(args, "spg_zhead_epochs", 50)),
        zhead_hidden_dim=int(getattr(args, "spg_zhead_hidden_dim", 0)),
        zhead_lr=float(getattr(args, "spg_zhead_lr", 1e-3)),
        zhead_weight_decay=float(getattr(args, "spg_zhead_weight_decay", 1e-4)),
        zhead_batch_size=int(getattr(args, "spg_zhead_batch_size", 128)),
        projection_ridge=float(getattr(args, "spg_projection_ridge", 1e-6)),
        injection_mode="film" if method == "spg_cfm_film_one_step" else "concat",
    )

    Z = np.asarray(X_train_z, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.int64).ravel()
    
    # 1. Z-head & SPG Condition computation
    t0_zhead = time.perf_counter()
    spg_pia_cfg = SPGPIAConfig(
        zhead_epochs=cfg.zhead_epochs,
        zhead_hidden_dim=cfg.zhead_hidden_dim,
        zhead_lr=cfg.zhead_lr,
        zhead_weight_decay=cfg.zhead_weight_decay,
        batch_size=cfg.zhead_batch_size,
        projection_ridge=cfg.projection_ridge,
    )
    zhead = fit_zhead_classifier(Z, y_arr, seed=seed, device=str(getattr(args, "device", "cpu")), cfg=spg_pia_cfg)
    t_zhead = time.perf_counter() - t0_zhead
    
    t0_cond = time.perf_counter()
    grads = compute_spg_gradients(Z, y_arr, zhead=zhead)
    
    # Re-use SPG projector logic:
    from .spg_pia import _support_projector, build_zpia_direction_bank
    bank, _ = build_zpia_direction_bank(
        Z,
        k_dir=int(getattr(args, "k_dir", 10)),
        seed=seed,
        telm2_n_iters=int(getattr(args, "telm2_n_iters", 50)),
        telm2_c_repr=float(getattr(args, "telm2_c_repr", 10.0)),
        telm2_activation=str(getattr(args, "telm2_activation", "sine")),
        telm2_bias_update_mode=str(getattr(args, "telm2_bias_update_mode", "none")),
    )
    project, _, _ = _support_projector(bank, ridge=float(cfg.projection_ridge), eps=float(cfg.eps))
    
    proj_grads = np.stack([project(g) for g in grads]).astype(np.float64)
    s_hat = np.zeros_like(grads)
    E = np.zeros((len(Z),), dtype=np.float64)
    for i in range(len(Z)):
        g_i = grads[i]
        p_i = proj_grads[i]
        norm_gi = float(np.linalg.norm(g_i))
        norm_pi = float(np.linalg.norm(p_i))
        s_hat[i], _ = _unit(p_i, eps=float(cfg.eps))
        E[i] = (norm_pi * norm_pi) / (norm_gi * norm_gi + float(cfg.eps))
    t_cond = time.perf_counter() - t0_cond

    # 2. Latent Residual Target Sampling
    lrf_cfg = LatentResidualConfig(rbf_tau_floor=1e-12)
    target_out = build_latent_residual_targets(Z, y_arr, seed=seed, cfg=lrf_cfg)

    # 3. Fit SPG-CFM Flow Matching model
    t0_train = time.perf_counter()
    op = fit_spg_cfm_operator(Z, y_arr, target_out, s_hat, E, seed=seed, device=str(getattr(args, "device", "cpu")), cfg=cfg)
    t_train = time.perf_counter() - t0_train

    # 4. Single-step Generation
    t0_gen = time.perf_counter()
    eta_safe = None if bool(getattr(args, "disable_safe_step", False)) else float(getattr(args, "eta_safe", 0.75))
    gamma_requested = float(getattr(args, "pia_gamma", 0.1))
    multiplier = int(getattr(args, "multiplier", 10))
    rng = np.random.default_rng(int(seed) + 4099)
    margins = estimate_local_manifold_margins(Z, y_arr)
    
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []
    
    gammas_used: List[float] = []
    
    n_steps = 3 if method == "spg_cfm_k3" else 1
    
    for i in range(len(Z)):
        for c in range(multiplier):
            eps_vec, eps_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(cfg.eps))
            
            # Multi-step generation (Euler integration)
            r_vec = eps_vec.copy()
            u_norms = []
            for k in range(n_steps):
                t_val = float(k) / float(n_steps)
                pred_u = _predict_spg_cfm_velocity(op, Z[i], int(y_arr[i]), s_hat[i], E[i], r_vec, t_val)
                u_norms.append(float(np.linalg.norm(pred_u)))
                r_vec = r_vec + (1.0 / float(n_steps)) * pred_u
            
            # Compute trajectory
            r_hat = r_vec
            direction, direction_norm = _unit(r_hat, eps=float(cfg.eps))
            
            if direction_norm <= float(cfg.eps):
                direction, direction_norm = _unit(rng.normal(size=(Z.shape[1],)), eps=float(cfg.eps))
            
            # Safe step calculation
            d_min = float(margins[i])
            if eta_safe is None:
                gamma_used = gamma_requested
                safe_upper_bound = float("inf")
            else:
                safe_upper_bound = float(eta_safe) * d_min / (direction_norm + cfg.eps)
                gamma_used = min(gamma_requested, safe_upper_bound)
            
            W_i = (gamma_used * direction).astype(np.float32)
            z_aug.append((Z[i] + W_i).astype(np.float32))
            y_aug.append(int(y_arr[i]))
            tid_aug.append(train_recs[i].tid)
            gammas_used.append(float(gamma_used))
            
            # Record keeping
            align_to_spg = _cosine(direction, s_hat[i], eps=float(cfg.eps))
            
            rows.append({
                "anchor_index": int(i),
                "class_id": int(y_arr[i]),
                "candidate_order": int(c),
                "direction_source": (
                    "spg_conditioned_cfm_film_operator"
                    if method == "spg_cfm_film_one_step"
                    else "spg_conditioned_cfm_align_operator"
                    if method == "spg_cfm_align_one_step"
                    else "spg_conditioned_cfm_operator"
                ),
                "template_id": -1,
                "template_rank": -1,
                "template_response_abs": np.nan,
                "direction_id": -1,
                # SPG-CFM specific audit
                "spg_cfm_t": 1.0, # Final integration point
                "spg_cfm_steps": int(n_steps),
                "spg_cfm_eps_norm": float(eps_norm),
                "spg_cfm_pred_velocity_norm": float(np.mean(u_norms)),
                "spg_cfm_r_hat_norm": float(np.linalg.norm(r_hat)),
                "spg_cfm_alignment_to_spg": float(align_to_spg),
                "spg_projection_energy": float(E[i]),
                "spg_condition_norm": float(np.linalg.norm(s_hat[i])),
                # Standard manifold audit
                "gamma_requested": float(gamma_requested),
                "gamma_used": float(gamma_used),
                "gamma_used_ratio": float(gamma_used / gamma_requested) if abs(gamma_requested) > 1e-12 else 1.0,
                "pre_safe_displacement_norm": float(gamma_requested),
                "post_safe_displacement_norm": float(gamma_used),
                "z_displacement_norm": float(gamma_used),
                "safe_radius_ratio": float(safe_upper_bound / gamma_requested) if abs(gamma_requested) > 1e-12 else 1.0,
                "manifold_margin": float(d_min),
                "is_clipped": float(gamma_requested > safe_upper_bound + 1e-9),
                "_direction_vec": direction,
            })
            
    t_gen = time.perf_counter() - t0_gen

    # Diversity metrics
    by_anchor: Dict[int, List[np.ndarray]] = {}
    for row in rows:
        by_anchor.setdefault(int(row["anchor_index"]), []).append(np.asarray(row["_direction_vec"], dtype=np.float64))
    
    effective_multipliers: List[int] = []
    pairwise_cosines: List[float] = []
    for dirs in by_anchor.values():
        if not dirs:
            continue
        D = np.stack(dirs)
        rounded = np.round(D, decimals=6)
        unique_count = int(np.unique(rounded, axis=0).shape[0])
        effective_multipliers.append(unique_count)
        if len(dirs) > 1:
            pairwise_cosines.append(_pairwise_cosine_mean(D, eps=float(cfg.eps)))
            
    for row in rows:
        row.pop("_direction_vec", None)
        
    t_build_total = time.perf_counter() - t_start_build

    extra_meta = {
        "spg_cfm_train_mse_mean": float(op["summary"]["spg_cfm_train_mse_mean"]),
        "spg_cfm_train_cosine_mean": float(op["summary"]["spg_cfm_train_cosine_mean"]),
        "spg_cfm_train_pred_target_cosine_mean": float(op["summary"]["spg_cfm_train_cosine_mean"]),
        "spg_cfm_generation_pred_target_cosine_mean": float(np.nan),
        "spg_cfm_generated_direction_pairwise_cosine_mean": float(np.nanmean(pairwise_cosines)) if pairwise_cosines else np.nan,
        "spg_cfm_effective_aug_multiplier": float(np.mean(effective_multipliers)) if effective_multipliers else 1.0,
        "spg_cfm_alignment_to_spg_mean": float(np.nanmean([r["spg_cfm_alignment_to_spg"] for r in rows])),
        "spg_cfm_steps": int(n_steps),
        "spg_cfm_projection_energy_mean": float(np.mean(E)),
        "spg_cfm_projection_energy_std": float(np.std(E)),
        "spg_cfm_condition_norm_mean": float(np.mean(np.linalg.norm(s_hat, axis=1))),
        "spg_cfm_condition_norm_std": float(np.std(np.linalg.norm(s_hat, axis=1))),
        "spg_zhead_train_acc": float(zhead.get("spg_zhead_train_acc", np.nan)),
        "gamma_used_ratio_mean": float(np.mean([g / gamma_requested for g in gammas_used])) if abs(gamma_requested) > 0 else np.nan,
        "transport_error_logeuc_mean": float(np.nan), # Not applicable for CFM generation typically
        
        # Timing metrics
        "augmentation_build_time_sec": float(t_build_total),
        "spg_cfm_zhead_time_sec": float(t_zhead),
        "spg_cfm_condition_time_sec": float(t_cond),
        "spg_cfm_train_time_sec": float(t_train),
        "spg_cfm_generation_time_sec": float(t_gen),
        "generation_time_per_aug_sample_ms": float(t_gen * 1000.0 / max(1, len(z_aug))),
    }

    return materialize_z_aug_out(
        z_aug=np.stack(z_aug).astype(np.float32) if z_aug else np.empty((0, Z.shape[1]), dtype=np.float32),
        y_aug=np.asarray(y_aug, dtype=np.int64),
        tid_aug=np.asarray(tid_aug, dtype=object),
        audit_rows=rows,
        train_recs=train_recs,
        mean_log=mean_log,
        direction_bank_meta={},
        effective_k=1,
        eta_safe=eta_safe,
        algo_name=method,
        engine_id=method,
        extra_meta=extra_meta,
    )
