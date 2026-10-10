"""A small 1D CNN with no batch-statistic adaptation at test time."""
import torch
from torch import nn


class TransientCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b, k in [(1, 16, 15), (16, 32, 9), (32, 64, 5)]:
            layers.extend([nn.Conv1d(a, b, k, stride=2, padding=k // 2),
                           nn.GroupNorm(4, b), nn.GELU()])
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(128, 1)

    def embedding(self, x):
        features = self.encoder(x)
        return torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)

    def forward(self, x):
        return self.head(self.embedding(x)).squeeze(-1)
