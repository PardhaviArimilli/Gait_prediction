import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, ch: int, k: int = 5, d: int = 1, p: float = 0.1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, dilation=d, padding=pad),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x):
        out = self.net(x)
        return out[:, :, : x.size(2)]  # trim padding tail to keep length


class TCN(nn.Module):
    def __init__(self, num_classes: int = 3, layers: int = 4, ch: int = 64, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Conv1d(3, ch, kernel_size=3, padding=1)
        blocks = []
        for i in range(layers):
            blocks.append(TCNBlock(ch, k=5, d=2 ** i, p=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(ch, num_classes, kernel_size=1)

    def forward(self, x):  # (B, T, C)
        x = x.transpose(1, 2)       # (B, C, T)
        x = self.inp(x)
        x = self.tcn(x)
        logits = self.head(x)       # (B, K, T)
        return logits.transpose(1, 2)
