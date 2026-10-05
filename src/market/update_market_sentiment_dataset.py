import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SENTIMENT_FILE = PROCESSED_DIR / "daily_sentiment.csv"
BTC_FILE = RAW_DIR / "btc_daily.csv"
ETH_FILE = RAW_DIR / "eth_daily.csv"
SOL_FILE = RAW_DIR / "sol_daily.csv"
BNB_FILE = RAW_DIR / "bnb_daily.csv"

OUTPUT_FILE = PROCESSED_DIR / "market_sentiment_analysis.csv"


def build_extended_market_sentiment_dataset():
    print("=" * 60)
    print("BUILDING EXTENDED MARKET-SENTIMENT DATASET (2021 - 2026)")
    print("=" * 60)

    # 1. Load Market Data
    btc = pd.read_csv(BTC_FILE)
    eth = pd.read_csv(ETH_FILE)
    sol = pd.read_csv(SOL_FILE)
    bnb = pd.read_csv(BNB_FILE)

    for df in [btc, eth, sol, bnb]:
        df["date"] = pd.to_datetime(df["Date"])

    # Rename asset-specific columns
    btc = btc[["date", "Open", "High", "Low", "Close", "Volume", "Return"]].rename(columns={
        "Open": "btc_open", "High": "btc_high", "Low": "btc_low", "Close": "btc_close",
        "Volume": "btc_volume", "Return": "btc_return"
    })

    eth = eth[["date", "Open", "High", "Low", "Close", "Volume", "Return"]].rename(columns={
        "Open": "eth_open", "High": "eth_high", "Low": "eth_low", "Close": "eth_close",
        "Volume": "eth_volume", "Return": "eth_return"
    })

    sol = sol[["date", "Close", "Volume", "Return"]].rename(columns={
        "Close": "sol_close", "Volume": "sol_volume", "Return": "sol_return"
    })

    bnb = bnb[["date", "Close", "Volume", "Return"]].rename(columns={
        "Close": "bnb_close", "Volume": "bnb_volume", "Return": "bnb_return"
    })

    # Merge Market Data on Date (Outer to ensure complete range)
    market_df = btc.merge(eth, on="date", how="outer")
    market_df = market_df.merge(sol, on="date", how="outer")
    market_df = market_df.merge(bnb, on="date", how="outer")
    market_df = market_df.sort_values("date").reset_index(drop=True)

    # Fill missing return values
    for col in ["btc_return", "eth_return", "sol_return", "bnb_return"]:
        market_df[col] = market_df[col].fillna(0.0)

    # 2. Load Daily Sentiment
    sentiment = pd.read_csv(SENTIMENT_FILE)
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # Merge Market with Sentiment
    merged = market_df.merge(sentiment, on="date", how="left")

    # Handle sentiment values for extended dates
    avg_s_mean = sentiment["avg_sentiment"].mean()
    news_count_median = int(sentiment["news_count"].median())

    merged["avg_sentiment"] = merged["avg_sentiment"].ffill().bfill()
    merged["positive_ratio"] = merged["positive_ratio"].fillna(0.33)
    merged["neutral_ratio"] = merged["neutral_ratio"].fillna(0.34)
    merged["negative_ratio"] = merged["negative_ratio"].fillna(0.33)
    merged["news_count"] = merged["news_count"].fillna(news_count_median).astype(int)

    # 3. Add Volatility & Range Features
    merged["btc_volatility"] = (merged["btc_high"] - merged["btc_low"]) / merged["btc_close"]
    merged["eth_volatility"] = (merged["eth_high"] - merged["eth_low"]) / merged["eth_close"]
    merged["btc_abs_return"] = merged["btc_return"].abs()
    merged["eth_abs_return"] = merged["eth_return"].abs()

    # 4. Save
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Extended dataset created: {OUTPUT_FILE}")
    print(f"Total Rows: {len(merged)}")
    print(f"Date Range: {merged['date'].min().date()} → {merged['date'].max().date()}")
    print("=" * 60 + "\n")

    return merged


if __name__ == "__main__":
    build_extended_market_sentiment_dataset()
