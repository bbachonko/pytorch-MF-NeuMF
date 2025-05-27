import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Classic Matrix Factorization model for collaborative filtering.

    This model learns latent representations (embeddings) for users and items.
    It predicts a rating as a dot product of embeddings plus bias terms.

    Suitable for explicit feedback prediction (e.g., 1–5 star ratings).

    Attributes:
        user_embedding (nn.Embedding): Embedding layer for users.
        item_embedding (nn.Embedding): Embedding layer for items.
        bias_user (nn.Embedding): Bias term per user.
        bias_item (nn.Embedding): Bias term per item.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        dot = (self.user_emb(users) * self.item_emb(items)).sum(1)
        bias = self.user_bias(users).squeeze() + self.item_bias(items).squeeze()
        return dot + bias
