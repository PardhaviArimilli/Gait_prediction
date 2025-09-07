import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional
from .windows import make_windows


class WindowDataset(Dataset):
    def __init__(self, paths: List[str], window_s: float, overlap: float, sample_rate_hz: float,
                 label_cols: Optional[List[str]] = None, transform=None):
        self.paths = paths
        self.window_s = window_s
        self.overlap = overlap
        self.sample_rate_hz = sample_rate_hz
        self.label_cols = label_cols or []
        self.transform = transform
        self.cache = []
        for p in self.paths:
            df = pd.read_csv(p)
            X, Y = make_windows(df, window_s, overlap, sample_rate_hz, self.label_cols)
            self.cache.append((X, Y))
        self.index = []
        for idx_file, (X, Y) in enumerate(self.cache):
            for idx_win in range(len(X)):
                self.index.append((idx_file, idx_win))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, wi = self.index[idx]
        X, Y = self.cache[fi]
        x = torch.from_numpy(X[wi])  # (T, C)
        if self.label_cols:
            y = torch.from_numpy(Y[wi])  # (num_classes,)
        else:
            y = torch.empty(0)
        if self.transform:
            x = self.transform(x)
        return x, y
