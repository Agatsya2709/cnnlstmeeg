import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.35):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(48, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        feat = x.unsqueeze(1)
        feat = self.temporal(feat)
        feat = self.spatial(feat).squeeze(2)  # [batch, 48, time]
        feat = F.adaptive_avg_pool1d(feat, output_size=1).squeeze(-1)
        feat = self.dropout(feat)
        return self.head(feat)


class LSTMBaseline(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_size: int = 96, dropout: float = 0.35):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time] -> [batch, time, channels]
        seq = x.transpose(1, 2)
        seq = self.norm(seq)
        out, _ = self.lstm(seq)
        feat = out[:, -1, :]
        feat = self.dropout(feat)
        return self.head(feat)

