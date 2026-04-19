import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

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
    and the MBA-induced geometric shift.
    
    Narrative:
    - Synergy (cos > 0): MBA acts as a preconditioner, accelerating convergence 
      by pushing towards the descent direction.
    - Regularization (cos < 0): MBA acts as a regularizer, preventing over-fitting 
      by providing offsets that challenge the model in safe manifold areas.
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    
    x_orig = x_orig.to(dev).requires_grad_(True)
    y_orig = y_orig.to(dev)
    x_aug = x_aug.to(dev)
    
    # MBA Geometric shift direction: delta_x
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
