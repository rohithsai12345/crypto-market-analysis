import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT"
}

START_TIMESTAMP = int(datetime(2021, 1, 1).timestamp() * 1000)
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_klines(symbol, start_ms):
    all_klines = []
    current_start = start_ms
    end_ms = int(datetime.now().timestamp() * 1000)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current_start,
            "limit": 1000
        }
        try:
            res = requests.get(BINANCE_KLINE_URL, params=params, timeout=10)
            if res.status_code != 200:
                print(f"Error fetching {symbol}: status {res.status_code}")
                break
            data = res.json()
            if not data:
                break
            all_klines.extend(data)
            last_open_time = data[-1][0]
            if last_open_time <= current_start:
                break
            current_start = last_open_time + 1
            time.sleep(0.2)
        except Exception as err:
            print(f"Failed fetching {symbol}: {err}")
            break

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])

    df["Date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in numeric_cols:
        df[c] = df[c].astype(float)

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Return"] = df["Close"].pct_change()
    return df


def download_all_market_data():
    print("=" * 60)
    print("DOWNLOADING EXTENDED REAL MARKET DATA FROM BINANCE (2021 - PRESENT)")
    print("=" * 60)

    for asset, symbol in ASSETS.items():
        print(f"Fetching {asset} ({symbol})...")
        df = fetch_binance_klines(symbol, START_TIMESTAMP)
        if not df.empty:
            out_file = OUTPUT_DIR / f"{asset.lower()}_daily.csv"
            df.to_csv(out_file, index=False)
            print(f"Saved: {out_file} | Rows: {len(df)} | Date range: {df['Date'].min()} → {df['Date'].max()}")
        else:
            print(f"Warning: No data received for {asset}")
    print("\nMarket data update complete.\n")


if __name__ == "__main__":
    download_all_market_data()
