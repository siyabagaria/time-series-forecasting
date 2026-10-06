# Comparative Study: LSTM vs GRU vs Transformer for Time-Series Forecasting

Datasets: UCI Household Electricity (hourly), ETTh1, Electricity (ECL).
Models: LSTM, GRU, Transformer + baselines (Seasonal-naive, Linear).
Setup: 96h lookback -> 24h / 96h horizon, 3 seeds, metrics MAE / RMSE / train time / #params.

## Repo layout
```
data/raw/        raw downloads (not committed)
data/processed/  scaled arrays + json metadata (not committed)
src/             download_data, data_prep, data_io, windows, metrics, baselines, run_baselines
notebooks/       Colab notebooks
results/         results CSVs
```

## Reproduce the data + baseline part
```bash
pip install -r requirements.txt
python -m src.download_data          # -> data/raw/
python -m src.data_prep              # -> data/processed/
python -m src.run_baselines          # -> results/baselines.csv
```

## Data decisions (Person 1)
- UCI: minute -> hourly mean; `?` = missing -> time interpolation (then ffill/bfill). Target: `Global_active_power`.
- ETTh1: already hourly. Target: `OT`.
- ECL: 15-min -> hourly mean, from 2012-01-01, drop mostly-zero clients, keep first 20 clients; target = last kept client.
- Split chronologically 70/10/20. Scaler (z-score) fitted on the TRAIN rows only.
- Val/test windows start `lookback` rows early so the first window has a full input; all targets stay inside their own split.
- Metrics are computed on the scaled target (same for every model).

## Using the data in a model (for P2)
```python
from src.windows import get_dataloaders
loaders, meta = get_dataloaders("ett", lookback=96, horizon=24, batch_size=64)
x, y = next(iter(loaders["train"]))   # x [B,96,F], y [B,24]
```
Results CSV columns: `dataset, model, horizon, seed, MAE, RMSE, time, params`.
