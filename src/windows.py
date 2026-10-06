"""PyTorch Dataset/DataLoader: (lookback past hours, all features) -> next `horizon` hours of the target.

Shared by everyone (P2 imports get_dataloaders):
    from src.windows import get_dataloaders
    loaders, meta = get_dataloaders("ett", lookback=96, horizon=24, batch_size=64)
    for x, y in loaders["train"]:   # x: [B, lookback, F]   y: [B, horizon]
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data_io import load_processed, split_ranges, target_windows  # noqa: F401


class WindowDataset(Dataset):
    def __init__(self, data, target_idx, lookback, horizon, start, end):
        self.d = torch.from_numpy(np.ascontiguousarray(data[start:end]))
        self.t, self.L, self.H = target_idx, lookback, horizon
        self.n = max(0, len(self.d) - lookback - horizon + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.d[i:i + self.L]                              # [L, F]
        y = self.d[i + self.L:i + self.L + self.H, self.t]    # [H]
        return x, y


def get_dataloaders(name, lookback=96, horizon=24, batch_size=64, num_workers=0):
    data, meta = load_processed(name)
    loaders = {}
    for split, (s, e) in split_ranges(meta, lookback).items():
        ds = WindowDataset(data, meta["target_idx"], lookback, horizon, s, e)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"), num_workers=num_workers)
    return loaders, meta
