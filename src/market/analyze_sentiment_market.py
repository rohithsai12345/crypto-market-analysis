import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SENTIMENT_FILE = Path("data/processed/daily_sentiment.csv")
BTC_FILE = Path("data/raw/btc_daily.csv")
ETH_FILE = Path("data/raw/eth_daily.csv")

OUTPUT_FILE = Path(
    "data/processed/market_sentiment_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CRYPTOCURRENCY MARKET–SENTIMENT ANALYSIS")
print("=" * 60)

sentiment = pd.read_csv(SENTIMENT_FILE)
btc = pd.read_csv(BTC_FILE)
eth = pd.read_csv(ETH_FILE)

sentiment["date"] = pd.to_datetime(sentiment["date"])
btc["date"] = pd.to_datetime(btc["Date"])
eth["date"] = pd.to_datetime(eth["Date"])

print("\nSentiment records:", len(sentiment))
print("BTC records:", len(btc))
print("ETH records:", len(eth))


# ============================================================
# SELECT MARKET VARIABLES
# ============================================================

btc = btc[
    [
        "date",
        "Close",
        "Volume",
        "Return"
    ]
].rename(
    columns={
        "Close": "btc_close",
        "Volume": "btc_volume",
        "Return": "btc_return"
    }
)

eth = eth[
    [
        "date",
        "Close",
        "Volume",
        "Return"
    ]
].rename(
    columns={
        "Close": "eth_close",
        "Volume": "eth_volume",
        "Return": "eth_return"
    }
)


# ============================================================
# MERGE SENTIMENT + MARKET DATA
# ============================================================

analysis = sentiment.merge(
    btc,
    on="date",
    how="inner"
)

analysis = analysis.merge(
    eth,
    on="date",
    how="inner"
)

analysis = analysis.sort_values("date").reset_index(drop=True)


print("\nMerged records:", len(analysis))

print(
    "Date range:",
    analysis["date"].min().date(),
    "to",
    analysis["date"].max().date()
)


# ============================================================
# LAGGED SENTIMENT VARIABLES
# ============================================================

# Previous-day sentiment
analysis["sentiment_lag_1"] = (
    analysis["avg_sentiment"].shift(1)
)

# Two-day lag
analysis["sentiment_lag_2"] = (
    analysis["avg_sentiment"].shift(2)
)

# Three-day lag
analysis["sentiment_lag_3"] = (
    analysis["avg_sentiment"].shift(3)
)

# Previous-day news volume
analysis["news_count_lag_1"] = (
    analysis["news_count"].shift(1)
)


# ============================================================
# FORWARD RETURNS
# ============================================================

# Next-day returns
analysis["btc_return_next_day"] = (
    analysis["btc_return"].shift(-1)
)

analysis["eth_return_next_day"] = (
    analysis["eth_return"].shift(-1)
)

# Two-day forward return
analysis["btc_return_2day"] = (
    analysis["btc_close"].shift(-2)
    / analysis["btc_close"]
    - 1
)

analysis["eth_return_2day"] = (
    analysis["eth_close"].shift(-2)
    / analysis["eth_close"]
    - 1
)


# ============================================================
# VOLATILITY MEASURES
# ============================================================

analysis["btc_abs_return"] = (
    analysis["btc_return"].abs()
)

analysis["eth_abs_return"] = (
    analysis["eth_return"].abs()
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)


def correlation(x, y):
    temp = analysis[[x, y]].dropna()

    if len(temp) < 2:
        return np.nan

    return temp[x].corr(temp[y])


results = []


# Same-day relationships
results.append(
    {
        "relationship": "Daily sentiment vs BTC return",
        "x": "avg_sentiment",
        "y": "btc_return",
        "correlation": correlation(
            "avg_sentiment",
            "btc_return"
        )
    }
)

results.append(
    {
        "relationship": "Daily sentiment vs ETH return",
        "x": "avg_sentiment",
        "y": "eth_return",
        "correlation": correlation(
            "avg_sentiment",
            "eth_return"
        )
    }
)

results.append(
    {
        "relationship": "Daily sentiment vs BTC volume",
        "x": "avg_sentiment",
        "y": "btc_volume",
        "correlation": correlation(
            "avg_sentiment",
            "btc_volume"
        )
    }
)

results.append(
    {
        "relationship": "Daily sentiment vs ETH volume",
        "x": "avg_sentiment",
        "y": "eth_volume",
        "correlation": correlation(
            "avg_sentiment",
            "eth_volume"
        )
    }
)


# Lagged relationships
results.append(
    {
        "relationship": "Previous-day sentiment vs BTC next-day return",
        "x": "sentiment_lag_1",
        "y": "btc_return_next_day",
        "correlation": correlation(
            "sentiment_lag_1",
            "btc_return_next_day"
        )
    }
)

results.append(
    {
        "relationship": "Previous-day sentiment vs ETH next-day return",
        "x": "sentiment_lag_1",
        "y": "eth_return_next_day",
        "correlation": correlation(
            "sentiment_lag_1",
            "eth_return_next_day"
        )
    }
)

results.append(
    {
        "relationship": "2-day lag sentiment vs BTC return",
        "x": "sentiment_lag_2",
        "y": "btc_return_next_day",
        "correlation": correlation(
            "sentiment_lag_2",
            "btc_return_next_day"
        )
    }
)

results.append(
    {
        "relationship": "3-day lag sentiment vs BTC return",
        "x": "sentiment_lag_3",
        "y": "btc_return_next_day",
        "correlation": correlation(
            "sentiment_lag_3",
            "btc_return_next_day"
        )
    }
)


# News volume relationships
results.append(
    {
        "relationship": "News volume vs BTC absolute return",
        "x": "news_count",
        "y": "btc_abs_return",
        "correlation": correlation(
            "news_count",
            "btc_abs_return"
        )
    }
)

results.append(
    {
        "relationship": "News volume vs ETH absolute return",
        "x": "news_count",
        "y": "eth_abs_return",
        "correlation": correlation(
            "news_count",
            "eth_abs_return"
        )
    }
)


correlation_results = pd.DataFrame(results)


# ============================================================
# PRINT CORRELATIONS
# ============================================================

print(
    correlation_results[
        [
            "relationship",
            "correlation"
        ]
    ].to_string(index=False)
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

matrix_columns = [
    "avg_sentiment",
    "news_count",
    "btc_return",
    "eth_return",
    "btc_volume",
    "eth_volume",
    "btc_abs_return",
    "eth_abs_return"
]

print("\nCorrelation matrix:")

print(
    analysis[matrix_columns]
    .corr()
    .round(4)
)


# ============================================================
# SAVE ANALYSIS DATA
# ============================================================

analysis.to_csv(
    OUTPUT_FILE,
    index=False
)

correlation_results.to_csv(
    "data/processed/correlation_results.csv",
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print("\nAnalysis shape:", analysis.shape)

print("\nMissing values:")
print(
    analysis[
        [
            "avg_sentiment",
            "btc_return",
            "eth_return"
        ]
    ]
    .isna()
    .sum()
)

print("\nSaved:")
print(OUTPUT_FILE)
print("data/processed/correlation_results.csv")

print("\n" + "=" * 60)
print("MARKET–SENTIMENT ANALYSIS COMPLETE")
print("=" * 60)
