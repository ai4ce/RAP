import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.mask = None
        self.positives = None

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Tensor of shape [N, D] – first view of each sample
            z_j: Tensor of shape [N, D] – second view of each sample
        Returns:
            Scalar loss value
        """
        z_i = z_i.reshape(z_i.shape[0], -1)
        z_j = z_j.reshape(z_i.shape[0], -1)

        z = torch.cat([z_i, z_j], dim=0)  # shape: [2N, D]
        z = F.normalize(z, dim=1)  # cosine similarity needs normalized vectors

        # Compute cosine similarity matrix: [2N, 2N]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Remove self-similarity
        N = z_i.size(0)
        if self.mask is None:
            self.mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)

            # Positive pairs: i-th and (i+N)-th are positives
            self.positives = torch.cat([
                torch.arange(N, 2 * N, device=z.device),
                torch.arange(0, N, device=z.device)
            ], dim=0)

        sim_matrix = sim_matrix.masked_select(self.mask).view(2 * N, -1)

        sim_pos = torch.exp(torch.sum(z * z[self.positives], dim=1) / self.temperature)
        sim_all = torch.exp(sim_matrix).sum(dim=1)

        loss = -torch.log(sim_pos / sim_all)
        return loss.mean()
