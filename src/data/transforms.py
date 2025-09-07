from __future__ import annotations

import math
import random
from typing import Callable, List

import torch


class Compose:
    def __init__(self, transforms: List[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class StandardizePerWindow:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, C)
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + self.eps)


class Jitter:
    def __init__(self, sigma: float = 0.01, p: float = 0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class Scale:
    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1, p: float = 0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        s = random.uniform(self.min_scale, self.max_scale)
        return x * s


class Rotate3D:
    def __init__(self, max_deg: float = 5.0, p: float = 0.5):
        self.max_deg = max_deg
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, C=3)
        if x.shape[1] != 3 or random.random() > self.p:
            return x
        deg = math.radians(random.uniform(-self.max_deg, self.max_deg))
        ca, sa = math.cos(deg), math.sin(deg)
        # simple rotation around vertical axis (AccV ~ z), rotate ML/AP
        R = x.new_tensor([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        return x @ R


