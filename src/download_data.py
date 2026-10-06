"""Download the 3 raw datasets into data/raw/.

Usage:  python -m src.download_data
If a download fails (network/URL change), download the file manually from the
URL printed below and place it in data/raw/ with the expected name.
"""
import urllib.request, zipfile, io
from pathlib import Path

RAW = Path("data/raw")

UCI_HOUSEHOLD = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
ECL_UCI = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
ETTH1 = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def fetch(url):
    print("Downloading", url)
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / "ETTh1.csv").exists():
        (RAW / "ETTh1.csv").write_bytes(fetch(ETTH1))
    if not (RAW / "household_power_consumption.txt").exists():
        zipfile.ZipFile(io.BytesIO(fetch(UCI_HOUSEHOLD))).extractall(RAW)
    if not (RAW / "LD2011_2014.txt").exists():
        zipfile.ZipFile(io.BytesIO(fetch(ECL_UCI))).extractall(RAW)
    print("Files in data/raw:", sorted(p.name for p in RAW.iterdir()))


if __name__ == "__main__":
    main()
