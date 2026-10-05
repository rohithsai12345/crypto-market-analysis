import pandas as pd
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/raw/train-00000-of-00001.parquet")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "news_clean.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Conservative text preprocessing.

    We intentionally preserve:
    - numbers
    - financial terms
    - crypto tickers
    - negations
    - punctuation that may carry meaning
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 60)
print("NLP PREPROCESSING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Original records:", len(df))


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

print("\nProcessing timestamps...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

if df["timestamp"].isna().any():
    raise ValueError("Invalid timestamps detected.")

df["date"] = df["timestamp"].dt.date.astype(str)


# ============================================================
# DUPLICATE HANDLING
# ============================================================

print("\nChecking duplicate timestamp + URL records...")

before_dedup = len(df)

# Sort so that the record with the highest vote count
# is retained when duplicate timestamp + URL records exist.
df = df.sort_values(
    by=["timestamp", "url", "total_votes"],
    ascending=[True, True, False]
)

df = df.drop_duplicates(
    subset=["timestamp", "url"],
    keep="first"
)

after_dedup = len(df)

print("Records before deduplication:", before_dedup)
print("Records after deduplication:", after_dedup)
print("Records removed:", before_dedup - after_dedup)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

print("\nCleaning text...")

# Preserve original text
df["text_original"] = df["text"]

# Create cleaned text
df["clean_text"] = df["text"].apply(clean_text)

# Combine title and cleaned article text
df["model_text"] = (
    df["title"].astype(str).str.strip()
    + " "
    + df["clean_text"].astype(str).str.strip()
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    "timestamp",
    "date",
    "title",
    "description",
    "text_original",
    "clean_text",
    "model_text",
    "market_direction",
    "engagement_quality",
    "total_votes",
    "source_url",
    "url",
    "total_tokens"
]

df = df[output_columns]


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\nFinal validation...")

print("Final shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print(
    "\nDuplicate timestamp + URL:",
    df.duplicated(
        subset=["timestamp", "url"]
    ).sum()
)

print("\nDate range:")
print(df["date"].min(), "→", df["date"].max())

print("\nMarket direction distribution:")
print(
    df["market_direction"]
    .value_counts()
    .sort_index()
)

print("\nClean text length:")
print(df["clean_text"].str.len().describe())


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
