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


CS_FLOW_METHODS = {"cs_flow_target_direct", "cs_flow_single_step"}


@dataclass(frozen=True)
class CSFlowConfig:
    k_same: int = 5
    flow_epochs: int = 50
    flow_batch_size: int = 128
    flow_lr: float = 1e-3
    flow_weight_decay: float = 1e-4
    hidden_layers: int = 2
    hidden_width: int = 0
    class_embedding_dim: int = 0
    t_gen: float = 0.0
    eps: float = 1e-12


class _CSFlowMLP(nn.Module):
    def __init__(self, *, z_dim: int, n_classes: int, class_embedding_dim: int, hidden_width: int, hidden_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, class_embedding_dim)
        in_dim = z_dim + 1 + class_embedding_dim
        layers: List[nn.Module] = []
        for idx in range(max(1, int(hidden_layers))):
            layers.append(nn.Linear(in_dim if idx == 0 else hidden_width, hidden_width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_width, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(y)
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([z, t, emb], dim=1))


def _unit(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        return np.zeros_like(v, dtype=np.float64), norm
    return (v / norm).astype(np.float64), norm


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
        "unique_direction_ratio": float(np.mean(unique_ratios)) if unique_ratios else np.nan,
        "generated_direction_pairwise_cosine_mean": float(np.nanmean(pairwise)) if pairwise else np.nan,
        "effective_aug_multiplier": float(np.mean(effective_multipliers)) if effective_multipliers else 0.0,
    }


def _class_remap(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    labels = sorted(int(v) for v in np.unique(y))
    mapping = {label: idx for idx, label in enumerate(labels)}
    return np.asarray([mapping[int(v)] for v in y], dtype=np.int64), mapping


def build_cs_flow_targets(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cfg: CSFlowConfig | None = None,
) -> Dict[str, object]:
    """Build train-only same-class one-step flow matching targets."""
    cfg = cfg or CSFlowConfig()
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n, z_dim = Z.shape
    rng = np.random.default_rng(int(seed) + 4417)
    target_indices = np.full((n,), -1, dtype=np.int64)
    target_classes = np.asarray(y, dtype=np.int64).copy()
    velocity_raw = np.zeros((n, z_dim), dtype=np.float64)
    velocity_unit = np.zeros((n, z_dim), dtype=np.float64)
    t_train = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float64)
    z_t = np.zeros((n, z_dim), dtype=np.float64)
    fallback_flags: List[bool] = []
    fallback_reasons: List[str] = []
    target_dists: List[float] = []
    velocity_norms: List[float] = []

    for i in range(n):
        dists = np.linalg.norm(Z - Z[i], axis=1)
        same = np.where(y == y[i])[0]
        same = same[same != i]
        same_order = same[np.argsort(dists[same])] if same.size else same
        candidates = same_order[: int(cfg.k_same)]
        fallback = False
        reason = ""
        if candidates.size:
            j = int(rng.choice(candidates))
            vel = Z[j] - Z[i]
            target_indices[i] = j
        else:
            j = -1
            vel = rng.normal(size=(z_dim,))
            fallback = True
            reason = "singleton_class_seeded_random"
        vel_unit, vel_norm = _unit(vel, eps=float(cfg.eps))
        if vel_norm <= float(cfg.eps):
            vel_unit, vel_norm = _unit(rng.normal(size=(z_dim,)), eps=float(cfg.eps))
            fallback = True
            reason = reason or "zero_velocity_seeded_random"
        velocity_raw[i] = vel
        velocity_unit[i] = vel_unit
        if j >= 0:
            z_t[i] = (1.0 - t_train[i]) * Z[i] + t_train[i] * Z[j]
            dist = float(np.linalg.norm(Z[j] - Z[i]))
        else:
            z_t[i] = Z[i]
            dist = float(vel_norm)
        fallback_flags.append(bool(fallback))
        fallback_reasons.append(str(reason))
        target_dists.append(dist)
        velocity_norms.append(float(vel_norm))

    summary = {
        "cs_flow_target_dist_mean": float(np.mean(target_dists)) if target_dists else np.nan,
        "cs_flow_target_dist_std": float(np.std(target_dists)) if target_dists else np.nan,
        "cs_flow_velocity_norm_mean": float(np.mean(velocity_norms)) if velocity_norms else np.nan,
        "cs_flow_velocity_norm_std": float(np.std(velocity_norms)) if velocity_norms else np.nan,
        "cs_flow_fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
        "cs_flow_target_effective_rank": _effective_rank(velocity_raw),
        "cs_flow_target_pairwise_cosine_mean": _pairwise_cosine_mean(velocity_raw),
        "cs_flow_velocity_effective_rank": _effective_rank(velocity_unit),
        "cs_flow_velocity_pairwise_cosine_mean": _pairwise_cosine_mean(velocity_unit),
    }
    return {
        "z_t": z_t.astype(np.float64),
        "t_train": t_train.astype(np.float64),
        "velocity_unit": velocity_unit.astype(np.float64),
        "velocity_raw": velocity_raw.astype(np.float64),
        "target_indices": target_indices,
        "target_classes": target_classes,
        "fallback_flags": np.asarray(fallback_flags, dtype=bool),
        "fallback_reasons": fallback_reasons,
        "target_dists": np.asarray(target_dists, dtype=np.float64),
        "velocity_norms": np.asarray(velocity_norms, dtype=np.float64),
        "summary": summary,
    }


def fit_cs_flow_operator(
    Z: np.ndarray,
    y: np.ndarray,
    target_out: Dict[str, object],
    *,
    seed: int,
    device: str,
    cfg: CSFlowConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or CSFlowConfig()
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    y_mapped, mapping = _class_remap(y)
    n, z_dim = Z.shape
    n_classes = len(mapping)
    hidden_width = int(cfg.hidden_width) if int(cfg.hidden_width) > 0 else max(128, 2 * int(z_dim))
    emb_dim = int(cfg.class_embedding_dim) if int(cfg.class_embedding_dim) > 0 else min(16, max(4, n_classes))
    torch.manual_seed(int(seed) + 5531)
    np.random.seed(int(seed) + 5531)
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = _CSFlowMLP(
        z_dim=z_dim,
        n_classes=n_classes,
        class_embedding_dim=emb_dim,
        hidden_width=hidden_width,
        hidden_layers=int(cfg.hidden_layers),
    ).to(dev)
    ds = TensorDataset(
        torch.from_numpy(np.asarray(target_out["z_t"], dtype=np.float32)),
        torch.from_numpy(np.asarray(target_out["t_train"], dtype=np.float32)),
        torch.from_numpy(y_mapped.astype(np.int64)),
        torch.from_numpy(np.asarray(target_out["velocity_unit"], dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 5531)
    loader = DataLoader(ds, batch_size=int(cfg.flow_batch_size), shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.flow_lr), weight_decay=float(cfg.flow_weight_decay))
    model.train()
    for _ in range(int(cfg.flow_epochs)):
        for z_b, t_b, y_b, v_b in loader:
            z_b = z_b.to(dev)
            t_b = t_b.to(dev)
            y_b = y_b.to(dev)
            v_b = v_b.to(dev)
            pred = model(z_b, t_b, y_b)
            loss = torch.mean((pred - v_b) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        z_all = torch.from_numpy(np.asarray(target_out["z_t"], dtype=np.float32)).to(dev)
        t_all = torch.from_numpy(np.asarray(target_out["t_train"], dtype=np.float32)).to(dev)
        y_all = torch.from_numpy(y_mapped.astype(np.int64)).to(dev)
        target = torch.from_numpy(np.asarray(target_out["velocity_unit"], dtype=np.float32)).to(dev)
        pred = model(z_all, t_all, y_all)
        mse = torch.mean((pred - target) ** 2).item()
        pred_n = pred / torch.clamp(torch.linalg.norm(pred, dim=1, keepdim=True), min=float(cfg.eps))
        target_n = target / torch.clamp(torch.linalg.norm(target, dim=1, keepdim=True), min=float(cfg.eps))
        cosine = torch.mean(torch.sum(pred_n * target_n, dim=1)).item()
    return {
        "model": model,
        "label_mapping": mapping,
        "device": dev,
        "summary": {
            "cs_flow_train_mse_mean": float(mse),
            "cs_flow_train_cosine_mean": float(cosine),
            "cs_flow_hidden_width": int(hidden_width),
            "cs_flow_class_embedding_dim": int(emb_dim),
            "cs_flow_hidden_layers": int(cfg.hidden_layers),
        },
    }


def _predict_flow_direction(
    op: Dict[str, object],
    z: np.ndarray,
    y_value: int,
    t_value: float,
    *,
    cfg: CSFlowConfig,
) -> Tuple[np.ndarray, float]:
    model: _CSFlowMLP = op["model"]
    mapping: Dict[int, int] = op["label_mapping"]
    dev: torch.device = op["device"]
    y_idx = mapping[int(y_value)]
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(np.asarray(z[None, :], dtype=np.float32)).to(dev),
            torch.tensor([float(t_value)], dtype=torch.float32, device=dev),
            torch.tensor([int(y_idx)], dtype=torch.long, device=dev),
        )
    pred_np = pred.cpu().numpy()[0].astype(np.float64)
    return _unit(pred_np, eps=float(cfg.eps))


def build_cs_flow_aug_out(
    *,
    args,
    seed: int,
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    train_recs: List[TrialRecord],
    mean_log: np.ndarray,
    method: str,
) -> Dict[str, object]:
    if method not in CS_FLOW_METHODS:
        raise ValueError(f"Unknown CS-Flow method: {method}")
    if str(getattr(args, "template_selection", "top_response")) != "top_response":
        raise ValueError("CS-Flow is a direction operator and must not use template-selection modes.")
    cfg = CSFlowConfig(
        k_same=int(getattr(args, "cs_flow_k_same", 5)),
        flow_epochs=int(getattr(args, "cs_flow_epochs", 50)),
        flow_batch_size=int(getattr(args, "cs_flow_batch_size", 128)),
        flow_lr=float(getattr(args, "cs_flow_lr", 1e-3)),
        flow_weight_decay=float(getattr(args, "cs_flow_weight_decay", 1e-4)),
        hidden_layers=int(getattr(args, "cs_flow_hidden_layers", 2)),
        hidden_width=int(getattr(args, "cs_flow_hidden_width", 0)),
        class_embedding_dim=int(getattr(args, "cs_flow_class_embedding_dim", 0)),
        t_gen=float(getattr(args, "cs_flow_t_gen", 0.0)),
    )
    Z = np.asarray(X_train_z, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.int64).ravel()
    target_out = build_cs_flow_targets(Z, y_arr, seed=seed, cfg=cfg)
    if method == "cs_flow_target_direct":
        op_summary = {
            "cs_flow_train_mse_mean": 0.0,
            "cs_flow_train_cosine_mean": 1.0,
            "cs_flow_hidden_width": 0,
            "cs_flow_class_embedding_dim": 0,
            "cs_flow_hidden_layers": 0,
        }
        op = None
    else:
        op = fit_cs_flow_operator(Z, y_arr, target_out, seed=seed, device=str(getattr(args, "device", "cpu")), cfg=cfg)
        op_summary = dict(op["summary"])

    tid_arr = np.asarray([record.tid for record in train_recs], dtype=object)
    margins = estimate_local_manifold_margins(Z, y_arr)
    eta_safe = None if bool(getattr(args, "disable_safe_step", False)) else float(getattr(args, "eta_safe", 0.75))
    gamma_requested = float(getattr(args, "pia_gamma", 0.1))
    multiplier = int(getattr(args, "multiplier", 10))
    rng = np.random.default_rng(int(seed) + 7219)
    z_aug: List[np.ndarray] = []
    y_aug: List[int] = []
    tid_aug: List[object] = []
    rows: List[Dict[str, object]] = []
    pred_cosines: List[float] = []

    for i in range(len(Z)):
        for c in range(multiplier):
            if float(cfg.t_gen) < 0.0:
                t_gen = float(rng.uniform(0.0, 0.5))
            else:
                t_gen = float(cfg.t_gen)
            target_direction = np.asarray(target_out["velocity_unit"][i], dtype=np.float64)
            if method == "cs_flow_target_direct":
                direction, pred_norm = _unit(target_direction, eps=float(cfg.eps))
                pred_target_cos = 1.0
            else:
                direction, pred_norm = _predict_flow_direction(op, Z[i], int(y_arr[i]), t_gen, cfg=cfg)
                pred_target_cos = float(np.dot(direction, target_direction) / max(float(np.linalg.norm(target_direction)), cfg.eps))
            direction_norm = float(np.linalg.norm(direction))
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
            pred_cosines.append(pred_target_cos)
            rows.append(
                {
                    "anchor_index": int(i),
                    "tid": tid_arr[i],
                    "class_id": int(y_arr[i]),
                    "candidate_order": int(c),
                    "slot_index": int(len(rows)),
                    "direction_source": "cs_flow_operator",
                    "template_id": -1,
                    "template_rank": -1,
                    "template_sign": np.nan,
                    "template_response_abs": np.nan,
                    "selected_template_rank": -1,
                    "selected_template_response_abs": np.nan,
                    "direction_id": -1,
                    "cs_flow_t": float(t_gen),
                    "cs_flow_target_index": int(target_out["target_indices"][i]),
                    "cs_flow_target_class": int(target_out["target_classes"][i]),
                    "cs_flow_target_dist": float(target_out["target_dists"][i]),
                    "cs_flow_velocity_norm": float(target_out["velocity_norms"][i]),
                    "cs_flow_pred_target_cosine": pred_target_cos,
                    "cs_flow_fallback_flag": bool(target_out["fallback_flags"][i]),
                    "cs_flow_fallback_reason": str(target_out["fallback_reasons"][i]),
                    "gamma_requested": float(gamma_requested),
                    "gamma_used": float(gamma_used),
                    "gamma_used_ratio": float(gamma_used / gamma_requested) if abs(gamma_requested) > cfg.eps else np.nan,
                    "direction_norm": direction_norm,
                    "pre_safe_displacement_norm": float(abs(gamma_requested) * direction_norm),
                    "post_safe_displacement_norm": float(np.linalg.norm(W_i)),
                    "z_displacement_norm": float(np.linalg.norm(W_i)),
                    "safe_upper_bound": float(safe_upper_bound),
                    "safe_radius_ratio": float(safe_radius_ratio),
                    "manifold_margin": d_min,
                    "is_clipped": float(gamma_requested > safe_upper_bound + 1e-9),
                    "selection_stage": "cs_flow_operator",
                    "selector_name": method,
                    "feasible_flag": 1.0,
                    "selector_accept_flag": 1.0,
                    "_direction_vec": direction,
                }
            )

    diversity = _direction_diversity(rows, eps=float(cfg.eps))
    for row in rows:
        row.pop("_direction_vec", None)
    cs_summary = {
        **dict(target_out["summary"]),
        **op_summary,
        **diversity,
        "cs_flow_pred_target_cosine_mean": float(np.mean(pred_cosines)) if pred_cosines else np.nan,
        "cs_flow_t_gen": float(cfg.t_gen),
        "cs_flow_k_same": int(cfg.k_same),
        "cs_flow_epochs": int(cfg.flow_epochs),
        "cs_flow_batch_size": int(cfg.flow_batch_size),
        "cs_flow_lr": float(cfg.flow_lr),
        "cs_flow_weight_decay": float(cfg.flow_weight_decay),
    }
    direction_meta = {
        "bank_source": "cs_flow_operator",
        "direction_source": "cs_flow_operator",
        "operator_source": method,
        **cs_summary,
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
            **cs_summary,
            "selection_stage": "cs_flow_operator",
            "selector_name": method,
            "multi_template_pairs": 0,
            "template_usage_entropy": 0.0,
            "top_template_concentration": 0.0,
        },
    )
