import yfinance as yf
import pandas as pd
from pathlib import Path

START_DATE = "2021-01-01"
END_DATE = "2025-05-24"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assets = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD"
}

for asset, ticker in assets.items():

    print(f"Downloading {asset} ({ticker})...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df = df[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ]

    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    df["Return"] = df["Close"].pct_change()

    output_file = OUTPUT_DIR / f"{asset.lower()}_daily.csv"

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
    print()


print("Market data download completed.")
