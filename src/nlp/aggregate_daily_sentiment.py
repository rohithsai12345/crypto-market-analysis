import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/news_finbert.csv")
OUTPUT_FILE = Path("data/processed/daily_sentiment.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DAILY SENTIMENT AGGREGATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print("\nNews records:", len(df))
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())


# ============================================================
# CREATE SENTIMENT INDICATORS
# ============================================================

# FinBERT class indicators
df["is_positive"] = (
    df["finbert_label"] == "positive"
).astype(int)

df["is_neutral"] = (
    df["finbert_label"] == "neutral"
).astype(int)

df["is_negative"] = (
    df["finbert_label"] == "negative"
).astype(int)


# ============================================================
# DAILY AGGREGATION
# ============================================================

daily = (
    df.groupby("date")
    .agg(
        news_count=("finbert_label", "size"),

        avg_sentiment=("finbert_sentiment_score", "mean"),

        avg_finbert_confidence=("finbert_score", "mean"),

        avg_positive_probability=("finbert_positive", "mean"),

        avg_negative_probability=("finbert_negative", "mean"),

        avg_neutral_probability=("finbert_neutral", "mean"),

        positive_news=("is_positive", "sum"),

        neutral_news=("is_neutral", "sum"),

        negative_news=("is_negative", "sum"),

        avg_market_direction=("market_direction", "mean"),
    )
    .reset_index()
)


# ============================================================
# DAILY SENTIMENT PROPORTIONS
# ============================================================

daily["positive_ratio"] = (
    daily["positive_news"] / daily["news_count"]
)

daily["neutral_ratio"] = (
    daily["neutral_news"] / daily["news_count"]
)

daily["negative_ratio"] = (
    daily["negative_news"] / daily["news_count"]
)


# ============================================================
# SORT
# ============================================================

daily = daily.sort_values("date").reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print("\nDaily records:", len(daily))

print(
    "\nDaily date range:",
    daily["date"].min().date(),
    "to",
    daily["date"].max().date()
)

print("\nMissing values:")
print(
    daily.isna().sum()[daily.isna().sum() > 0]
)

print("\nSentiment statistics:")
print(
    daily[
        [
            "avg_sentiment",
            "positive_ratio",
            "neutral_ratio",
            "negative_ratio"
        ]
    ].describe()
)

print("\nHighest news-volume days:")
print(
    daily.nlargest(10, "news_count")[
        ["date", "news_count", "avg_sentiment"]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

daily.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("DAILY SENTIMENT AGGREGATION COMPLETE")
print("=" * 60)
