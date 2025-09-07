import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 9, d: int = 1, p: float = 0.1):
        super().__init__()
        pad = (k // 2) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x):  # (B, C, T)
        return self.net(x)


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int = 3, conv_blocks: int = 3, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        ch = 64
        layers = []
        in_ch = 3
        for i in range(conv_blocks):
            layers.append(ConvBlock(in_ch, ch, k=9, d=1, p=dropout))
            in_ch = ch
        self.cnn = nn.Sequential(*layers)
        self.rnn = nn.LSTM(input_size=ch, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, x):  # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.cnn(x)        # (B, ch, T)
        x = x.transpose(1, 2)  # (B, T, ch)
        out, _ = self.rnn(x)   # (B, T, 2*hidden)
        logits = self.head(out)  # (B, T, num_classes)
        return logits
