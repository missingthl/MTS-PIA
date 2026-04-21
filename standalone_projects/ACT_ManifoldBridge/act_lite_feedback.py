from __future__ import annotations

import torch


def compute_true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return true-class margin: logit_y - max_{c!=y} logit_c."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [N]")

    num_classes = int(logits.shape[1])
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    if num_classes <= 1:
        return torch.zeros_like(true_logits)

    neg_logits = logits.clone()
    neg_logits.scatter_(1, labels.view(-1, 1), float("-inf"))
    hardest_negative = neg_logits.max(dim=1).values
    return true_logits - hardest_negative


def margin_to_feedback_weight(margins: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    tau = max(float(temperature), 1e-6)
    return torch.sigmoid(margins / tau).detach()


def compute_margin_feedback(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    margins = compute_true_class_margin(logits, labels)
    weights = margin_to_feedback_weight(margins, temperature=temperature)
    return margins, weights
