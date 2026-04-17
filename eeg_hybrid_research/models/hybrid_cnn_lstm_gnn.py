import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, features], adj: [nodes, nodes]
        support = torch.matmul(x, self.weight)
        out = torch.einsum("ij,bjk->bik", adj, support)
        return out + self.bias


class HybridCNNLSTMGNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        adjacency: torch.Tensor,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.register_buffer("adjacency", adjacency)

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.node_time_proj = nn.Linear(n_times, 96)
        self.gcn1 = GraphConvolution(96, 96)
        self.gcn2 = GraphConvolution(96, 96)

        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(48 + 192 + 96, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(160, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        cnn = self.temporal(x.unsqueeze(1))
        cnn = self.spatial(cnn).squeeze(2)  # [batch, 48, time]
        cnn_feat = F.adaptive_avg_pool1d(cnn, output_size=1).squeeze(-1)

        lstm_in = cnn.transpose(1, 2)  # [batch, time, 48]
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat = lstm_out[:, -1, :]

        node_feat = self.node_time_proj(x)  # [batch, channels, 96]
        node_feat = F.relu(self.gcn1(node_feat, self.adjacency))
        node_feat = self.dropout(node_feat)
        node_feat = F.relu(self.gcn2(node_feat, self.adjacency))
        gnn_feat = torch.mean(node_feat, dim=1)

        fused = torch.cat([cnn_feat, lstm_feat, gnn_feat], dim=1)
        fused = self.dropout(fused)
        return self.fusion(fused)

