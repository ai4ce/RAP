import torch
import torch.nn.functional as F


class InfoNCELoss(torch.nn.Module):
    """Symmetric InfoNCE loss with in-batch negatives."""

    def __init__(self, temperature=0.07, reduction="mean"):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, key):
        if query.shape[0] != key.shape[0]:
            raise ValueError("query and key must have the same batch dimension")

        query = query.reshape(query.shape[0], -1)
        key = key.reshape(key.shape[0], -1)

        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        logits = torch.matmul(query, key.T) / self.temperature
        targets = torch.arange(query.shape[0], device=query.device)

        # Symmetric objective: query->key and key->query.
        loss_q_to_k = F.cross_entropy(logits, targets, reduction=self.reduction)
        loss_k_to_q = F.cross_entropy(logits.T, targets, reduction=self.reduction)
        return 0.5 * (loss_q_to_k + loss_k_to_q)

