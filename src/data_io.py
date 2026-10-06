"""Numpy-only helpers (no torch needed) for loading processed data and making windows."""
import json
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PROC = Path("data/processed")


def load_processed(name, proc=PROC):
    meta = json.loads((Path(proc) / f"{name}.json").read_text())
    data = np.load(Path(proc) / f"{name}.npz")["data"]
    return data, meta


def split_ranges(meta, lookback):
    """Row ranges per split. Val/test start `lookback` rows early so their FIRST window
    has a full input; every TARGET value still lies inside its own split."""
    b = meta["borders"]
    return {"train": (b["train"][0], b["train"][1]),
            "val": (b["val"][0] - lookback, b["val"][1]),
            "test": (b["test"][0] - lookback, b["test"][1])}


def target_windows(name, split, lookback=96, horizon=24):
    """Windows of the TARGET series only: X [N, lookback], Y [N, horizon] (used by baselines)."""
    data, meta = load_processed(name)
    s, e = split_ranges(meta, lookback)[split]
    series = data[s:e, meta["target_idx"]]
    w = sliding_window_view(series, lookback + horizon)
    return w[:, :lookback].copy(), w[:, lookback:].copy()
