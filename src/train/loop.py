import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


def train_one_epoch(model: torch.nn.Module, dl: DataLoader, device: str = "cpu") -> Dict[str, float]:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)          # (B, T, K)
        logits_win = logits.mean(dim=1)  # (B, K)
        loss = crit(logits_win, y)
        loss.backward()
        opt.step()
        total += float(loss.item())
        n += 1
    return {"loss": total / max(n, 1)}
