"""Encoder, projection head, classifier."""
from __future__ import annotations
import torch
import torch.nn as nn


class IQEncoder(nn.Module):
    """1D-CNN + light Transformer producing 128-D features.

    Input  : (B, 2, 1024)
    Output : (B, 128)
    """

    def __init__(self,
                 in_ch: int = 2,
                 feat_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(128, feat_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.permute(0, 2, 1)            # (B, T', 128)
        h = self.transformer(h)
        h = h.mean(dim=1)
        return self.head(h)


class ProjectionHead(nn.Module):
    """SimCLR-style 2-layer projection MLP. Discarded after pre-training."""

    def __init__(self, in_dim: int = 128, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier(nn.Module):
    """Encoder + linear head used for closed-set training and OOD scoring."""

    def __init__(self, encoder: IQEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x))
