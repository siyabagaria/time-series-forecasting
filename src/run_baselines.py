"""Run both baselines on every dataset x horizon x seed and write results/baselines.csv
Columns match the team schema: dataset, model, horizon, seed, MAE, RMSE, time, params
(Metrics are on the SCALED target, same as the deep models should report.)

Usage: python -m src.run_baselines
Note: baselines are deterministic, so all seeds give identical numbers (std = 0).
"""
import argparse, time
from pathlib import Path
import pandas as pd
from .data_io import target_windows
from .baselines import seasonal_naive, LinearBaseline
from .metrics import mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uci", "ett", "ecl"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[24, 96])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--out", default="results/baselines.csv")
    a = ap.parse_args()

    rows = []
    for ds in a.datasets:
        for H in a.horizons:
            Xtr, Ytr = target_windows(ds, "train", a.lookback, H)
            Xte, Yte = target_windows(ds, "test", a.lookback, H)

            p = seasonal_naive(Xte, H)
            res_naive = dict(MAE=mae(Yte, p), RMSE=rmse(Yte, p), time=0.0, params=0)

            t0 = time.time()
            lin = LinearBaseline().fit(Xtr, Ytr)
            fit_t = time.time() - t0
            p = lin.predict(Xte)
            res_lin = dict(MAE=mae(Yte, p), RMSE=rmse(Yte, p), time=fit_t, params=lin.n_params)

            for seed in a.seeds:
                for model, r in (("SeasonalNaive", res_naive), ("Linear", res_lin)):
                    rows.append(dict(dataset=ds, model=model, horizon=H, seed=seed, **r))
            print(ds, H, "naive", round(res_naive["MAE"], 4), "linear", round(res_lin["MAE"], 4))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
