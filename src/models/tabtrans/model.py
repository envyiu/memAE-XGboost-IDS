from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x


class NumericFeatureTokenizer(nn.Module):
    """Project each processed numeric column into a trainable feature token."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        self.weight = nn.Parameter(torch.randn(input_dim, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(input_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError(f"Expected [batch, {self.weight.shape[0]}], got {tuple(x.shape)}")
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class NumericTabTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        latent_dim: int = 128,
        ff_mult: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        classifier_dropout: float = 0.1,
        pooling: str = "cls_mean",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if depth <= 0:
            raise ValueError("depth must be > 0")
        if heads <= 0:
            raise ValueError("heads must be > 0")
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if pooling not in {"cls", "mean", "cls_mean"}:
            raise ValueError("pooling must be one of: cls, mean, cls_mean")

        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.pooling = pooling
        self.tokenizer = NumericFeatureTokenizer(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    heads=heads,
                    ff_mult=ff_mult,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        pooled_dim = embed_dim * 2 if pooling == "cls_mean" else embed_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(pooled_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(latent_dim, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.final_norm(tokens)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(x)
        cls = tokens[:, 0]
        if self.pooling == "cls":
            pooled = cls
        else:
            mean_tokens = tokens[:, 1:].mean(dim=1)
            pooled = torch.cat([cls, mean_tokens], dim=1) if self.pooling == "cls_mean" else mean_tokens
        return self.feature_projection(pooled)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.extract_features(x)
        logits = self.classifier(features).squeeze(-1)
        if return_features:
            return logits, features
        return logits
