import torch
import torch.nn as nn


class LatentProjection(nn.Module):
    """MLP projection head: input_dim → hidden_dim → proj_dim → L2 normalize.

    Symmetric architecture for both EEG and audio modalities.
    """

    def __init__(self, input_dim: int, proj_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return nn.functional.normalize(x, dim=-1)
