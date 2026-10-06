"""Load -> hourly -> fill missing -> chronological 70/10/20 split -> scale (fit on TRAIN only).

Usage:  python -m src.data_prep                # all datasets
        python -m src.data_prep --datasets ett # one dataset

Output per dataset in data/processed/:
    <name>.npz   : data (scaled float32, shape [T, F]), timestamps
    <name>.json  : features, target, target_idx, borders, mean, std
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
TRAIN_FRAC, VAL_FRAC = 0.7, 0.1      # test = remaining 0.2
ECL_N_CLIENTS = 20                    # ECL has hundreds of clients; keep a manageable subset (agree with team!)


def load_uci(raw=RAW):
    df = pd.read_csv(raw / "household_power_consumption.txt", sep=";",
                     na_values=["?"], low_memory=False)
    df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).astype(float)
    df = df.resample("1h").mean()                       # minute -> hourly
    return df, "Global_active_power"


def load_ett(raw=RAW):
    df = pd.read_csv(raw / "ETTh1.csv", parse_dates=["date"], index_col="date")
    return df.asfreq("1h"), "OT"                         # already hourly; OT = oil temperature


def load_ecl(raw=RAW, n_clients=ECL_N_CLIENTS):
    df = pd.read_csv(raw / "LD2011_2014.txt", sep=";", decimal=",",
                     index_col=0, parse_dates=True)
    df = df.resample("1h").mean()                        # 15-min -> hourly
    df = df.loc["2012-01-01":]                           # earlier years: many clients not yet active
    df = df.loc[:, (df == 0).mean() < 0.05]              # drop clients that are mostly zero
    df = df.iloc[:, :n_clients]
    return df, df.columns[-1]                            # last kept client = forecast target


LOADERS = {"uci": load_uci, "ett": load_ett, "ecl": load_ecl}


def fill_missing(df):
    n_missing = int(df.isna().sum().sum())
    df = df.interpolate(method="time").ffill().bfill()
    assert not df.isna().any().any(), "NaNs remain after filling"
    return df, n_missing


def prepare(name):
    df, target = LOADERS[name]()
    df, n_missing = fill_missing(df)
    n = len(df)
    n_train, n_val = int(n * TRAIN_FRAC), int(n * VAL_FRAC)

    train = df.iloc[:n_train]
    mean = train.mean()
    std = train.std().replace(0, 1.0)                    # scaler fitted on TRAIN ONLY (no leakage)
    scaled = ((df - mean) / std).astype(np.float32)

    PROC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROC / f"{name}.npz",
                        data=scaled.values,
                        timestamps=df.index.values.astype("datetime64[s]"))
    meta = {
        "name": name, "features": list(df.columns), "target": target,
        "target_idx": list(df.columns).index(target),
        "n_rows": n, "n_missing_filled": n_missing,
        "borders": {"train": [0, n_train], "val": [n_train, n_train + n_val],
                    "test": [n_train + n_val, n]},
        "mean": mean.tolist(), "std": std.tolist(),
    }
    (PROC / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] rows={n} features={df.shape[1]} filled={n_missing} "
          f"train/val/test={n_train}/{n_val}/{n - n_train - n_val} target={target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    for name in ap.parse_args().datasets:
        prepare(name)


if __name__ == "__main__":
    main()
