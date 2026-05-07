from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MemAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, memory_size: int = 64, shrink_threshold: float = 0.0025, hidden_dims: list[int] = None, dropout: float = 0.0):
        super().__init__()
        self.shrink_threshold = shrink_threshold
        if hidden_dims is None:
            hidden = max(64, min(256, input_dim * 2))
            hidden_dims = [hidden]
            
        enc_layers = []
        in_d = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU()])
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_d = h
        enc_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        self.memory = nn.Parameter(torch.randn(memory_size, latent_dim) * 0.05)
        
        dec_layers = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU()])
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            in_d = h
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def hard_shrink_relu(self, weights: torch.Tensor) -> torch.Tensor:
        if self.shrink_threshold <= 0:
            return weights
        weights = F.relu(weights - self.shrink_threshold) * weights / (torch.abs(weights - self.shrink_threshold) + 1e-12)
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        attn = torch.softmax(torch.matmul(z, self.memory.t()), dim=1)
        attn = self.hard_shrink_relu(attn)
        z_hat = torch.matmul(attn, self.memory)
        x_hat = self.decoder(z_hat)
        return x_hat, z, z_hat, attn

    def memory_diversity_loss(self) -> torch.Tensor:
        normed = F.normalize(self.memory, dim=1)
        sim = torch.matmul(normed, normed.t())
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()


def memae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    attn: torch.Tensor,
    entropy_weight: float = 0.0002,
    diversity_loss: torch.Tensor | None = None,
    diversity_weight: float = 0.0,
):
    recon = F.mse_loss(x_hat, x)
    entropy = (-attn * torch.log(attn + 1e-12)).sum(dim=1).mean()
    diversity = diversity_loss if diversity_loss is not None else recon.new_tensor(0.0)
    loss = recon + entropy_weight * entropy + diversity_weight * diversity
    return loss, recon.detach(), entropy.detach(), diversity.detach()
