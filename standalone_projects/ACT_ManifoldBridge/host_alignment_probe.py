import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

def compute_gradient_alignment(
    model: nn.Module,
    x_orig: torch.Tensor,
    y_orig: torch.Tensor,
    x_aug: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Proposition 3: Measuring the Geometry-aware Preconditioner effect.
    Calculates the cosine similarity between the host's loss gradient 
    and the ACT-induced geometric shift.
    
    Narrative:
    - Synergy (cos > 0): ACT acts as a preconditioner, accelerating convergence 
      by pushing towards the descent direction.
    - Regularization (cos < 0): ACT acts as a regularizer, preventing over-fitting 
      by providing offsets that challenge the model in safe manifold areas.
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    
    x_orig = x_orig.to(dev).requires_grad_(True)
    y_orig = y_orig.to(dev)
    x_aug = x_aug.to(dev)
    
    # ACT Geometric shift direction: delta_x
    delta_x = x_aug - x_orig
    delta_x_flat = delta_x.view(-1)
    
    # Host Inductive Bias: -Gradient (the direction the host wants to move to minimize loss)
    try:
        logits = model(x_orig)
        loss = nn.CrossEntropyLoss()(logits, y_orig)
        model.zero_grad()
        
        # Use torch.autograd.grad for atomic gradient extraction into the input. 
        # This is more resilient to internal model in-place operations than loss.backward().
        grads = torch.autograd.grad(loss, x_orig, retain_graph=False, create_graph=False)
        grad_x = grads[0]
        
        if grad_x is None:
            return {"alignment_cosine": 0.0, "is_conflict": 0.0}
            
        grad_x_flat = grad_x.view(-1)
        # Target direction is -grad (descent)
        target_dir = -grad_x_flat
        
        # Cosine similarity
        cos = torch.nn.functional.cosine_similarity(delta_x_flat.unsqueeze(0), target_dir.unsqueeze(0))
        alignment = float(cos.item())
        
        return {
            "alignment_cosine": alignment,
            "is_conflict": 1.0 if alignment < 0 else 0.0
        }
    except Exception as e:
        # Fallback for extremely pathological models
        print(f"Prop 3 Probe Failure: {e}")
        return {"alignment_cosine": 0.0, "is_conflict": 0.0}

def compute_entropy_shift(
    model: nn.Module,
    x_orig: torch.Tensor,
    x_aug: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, float]:
    """Measures the shift in prediction uncertainty caused by augmentation."""
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        logits_orig = model(x_orig.to(dev))
        logits_aug = model(x_aug.to(dev))
        
        probs_orig = torch.softmax(logits_orig, dim=-1)
        probs_aug = torch.softmax(logits_aug, dim=-1)
        
        ent_orig = -torch.sum(probs_orig * torch.log(probs_orig + 1e-12), dim=-1).mean()
        ent_aug = -torch.sum(probs_aug * torch.log(probs_aug + 1e-12), dim=-1).mean()
        
    return {
        "entropy_orig": float(ent_orig.item()),
        "entropy_aug": float(ent_aug.item()),
        "entropy_shift": float((ent_aug - ent_orig).item())
    }


def compute_candidate_usefulness_batch(
    model: nn.Module,
    x_orig_batch: torch.Tensor,
    y_batch: torch.Tensor,
    x_cand_batch: torch.Tensor,
    device: str = "cuda"
) -> List[Dict[str, float]]:
    """
    Batch-oriented wrapper used by the ACL pipeline. Gradient alignment still runs
    sample-wise to keep the probe numerically stable, while entropy is evaluated
    alongside it for each candidate.
    """
    if x_orig_batch.shape[0] != x_cand_batch.shape[0] or x_orig_batch.shape[0] != y_batch.shape[0]:
        raise ValueError("Batch size mismatch for candidate usefulness probing")

    outputs: List[Dict[str, float]] = []
    for i in range(int(x_orig_batch.shape[0])):
        x_orig = x_orig_batch[i : i + 1]
        y_orig = y_batch[i : i + 1]
        x_cand = x_cand_batch[i : i + 1]

        align = compute_gradient_alignment(model, x_orig, y_orig, x_cand, device=device)
        entropy = compute_entropy_shift(model, x_orig, x_cand, device=device)
        outputs.append({
            "alignment_cosine": float(align.get("alignment_cosine", 0.0)),
            "is_conflict": float(align.get("is_conflict", 0.0)),
            "entropy_orig": float(entropy.get("entropy_orig", 0.0)),
            "entropy_aug": float(entropy.get("entropy_aug", 0.0)),
            "entropy_shift": float(entropy.get("entropy_shift", 0.0)),
        })
    return outputs


def score_hard_positive_candidates(
    candidate_rows: List[Dict[str, object]],
    *,
    alignment_weight: float = 0.7,
    positives_per_anchor: int = 1,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Score candidates within each anchor-local pool and select the top-m hard
    positives. Alignment is the primary challenge signal, while entropy shift is
    a secondary difficulty cue.
    """
    if positives_per_anchor < 1 or positives_per_anchor > 2:
        raise ValueError("positives_per_anchor must be 1 or 2")

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in candidate_rows:
        anchor_idx = int(row["anchor_index"])
        grouped.setdefault(anchor_idx, []).append(row)

    scored_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    for anchor_idx in sorted(grouped.keys()):
        rows = grouped[anchor_idx]
        alignment_raw = np.asarray(
            [(1.0 - float(r.get("alignment_cosine", 0.0))) / 2.0 for r in rows],
            dtype=np.float64,
        )
        entropy_raw = np.asarray(
            [max(0.0, float(r.get("entropy_shift", 0.0))) for r in rows],
            dtype=np.float64,
        )

        if alignment_raw.size == 0:
            continue

        align_norm = _minmax_norm(alignment_raw, constant_fill=1.0)
        entropy_norm = _minmax_norm(entropy_raw, constant_fill=0.0)

        valid_rows: List[Dict[str, object]] = []
        for i, row in enumerate(rows):
            scored = dict(row)
            scored["alignment_challenge"] = float(alignment_raw[i])
            scored["entropy_challenge"] = float(entropy_raw[i])
            scored["alignment_norm"] = float(align_norm[i])
            scored["entropy_norm"] = float(entropy_norm[i])
            challenge = (
                float(alignment_weight) * scored["alignment_norm"]
                + (1.0 - float(alignment_weight)) * scored["entropy_norm"]
            )
            fidelity = float(np.exp(-float(scored.get("transport_error_logeuc", np.inf))))
            safe_ratio = float(scored.get("safe_radius_ratio", 0.0))
            hard_positive_score = challenge * safe_ratio * fidelity
            is_valid = bool(
                np.isfinite(challenge)
                and np.isfinite(fidelity)
                and np.isfinite(safe_ratio)
                and np.isfinite(hard_positive_score)
            )
            scored["challenge_score"] = float(challenge)
            scored["fidelity_score"] = fidelity
            scored["hard_positive_score"] = float(hard_positive_score) if is_valid else float("-inf")
            scored["is_valid_candidate"] = int(is_valid)
            scored_rows.append(scored)
            if is_valid:
                valid_rows.append(scored)

        valid_rows = sorted(valid_rows, key=lambda x: float(x["hard_positive_score"]), reverse=True)
        for rank, selected in enumerate(valid_rows[:positives_per_anchor], start=1):
            picked = dict(selected)
            picked["selected_rank"] = int(rank)
            selected_rows.append(picked)

    return scored_rows, selected_rows


def _minmax_norm(x: np.ndarray, *, constant_fill: float) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.size == 0:
        return np.asarray([], dtype=np.float64)
    xmin, xmax = float(np.min(xx)), float(np.max(xx))
    if abs(xmax - xmin) <= 1e-12:
        return np.full(xx.shape, float(constant_fill), dtype=np.float64)
    return (xx - xmin) / (xmax - xmin)
