import torch
from torch.nn import functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    valid = positive.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    mean_positive = (log_prob.masked_fill(~positive, 0.0).sum(1) / positive.sum(1).clamp_min(1))
    return -mean_positive[valid].mean()


def invariance_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(first, second)).mean()

