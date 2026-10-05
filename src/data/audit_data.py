import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

# ============================================================
# NLP DATASET AUDIT
# ============================================================

nlp_file = RAW / "train-00000-of-00001.parquet"
nlp = pd.read_parquet(nlp_file)

print("=" * 60)
print("NLP DATASET AUDIT")
print("=" * 60)

print("Shape:", nlp.shape)

print("\nColumns:")
print(nlp.columns.tolist())

print("\nData types:")
print(nlp.dtypes)

print("\nMissing values:")
print(nlp.isna().sum())

# Check duplicate records using stable scalar fields
duplicate_mask = nlp.duplicated(
    subset=["timestamp", "title", "url"],
    keep=False
)

print("\nDuplicate records (timestamp + title + URL):",
      duplicate_mask.sum())

print("\nDuplicate URLs:")
print(nlp["url"].duplicated().sum())

print("\nMarket direction distribution:")
print(nlp["market_direction"].value_counts().sort_index())

print("\nTimestamp range:")
print(nlp["timestamp"].min(), "→", nlp["timestamp"].max())

print("\nText length:")
print(nlp["text"].str.len().describe())

print("\nTotal tokens:")
print(nlp["total_tokens"].describe())


# ============================================================
# MARKET DATA AUDIT
# ============================================================

for asset in ["btc", "eth"]:

    file = RAW / f"{asset}_daily.csv"
    df = pd.read_csv(file)

    print("\n" + "=" * 60)
    print(f"{asset.upper()} MARKET DATA AUDIT")
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate dates:",
          df["Date"].duplicated().sum())

    print("\nDate range:")
    print(df["Date"].min(), "→", df["Date"].max())

    print("\nPrice statistics:")
    print(
        df[["Open", "High", "Low", "Close"]].describe()
    )

    print("\nVolume statistics:")
    print(df["Volume"].describe())

    print("\nReturn statistics:")
    print(df["Return"].describe())


print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
