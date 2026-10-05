import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/news_clean.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CREATING CHRONOLOGICAL NLP SPLITS")
print("=" * 60)

# Load processed data
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)

# Chronological split
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

print(f"\nTotal records: {n}")
print(f"Training:     {len(train)}")
print(f"Validation:   {len(validation)}")
print(f"Test:         {len(test)}")

print("\nDate ranges:")

print(
    f"Train:      {train['timestamp'].min()} → "
    f"{train['timestamp'].max()}"
)

print(
    f"Validation: {validation['timestamp'].min()} → "
    f"{validation['timestamp'].max()}"
)

print(
    f"Test:       {test['timestamp'].min()} → "
    f"{test['timestamp'].max()}"
)

# Check for temporal overlap
assert train["timestamp"].max() < validation["timestamp"].min()
assert validation["timestamp"].max() < test["timestamp"].min()

# Label distributions
print("\nLabel distribution:")

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    print(f"\n{name}")
    print(
        data["market_direction"]
        .value_counts()
        .sort_index()
    )

# Save
train.to_csv(
    OUTPUT_DIR / "news_train.csv",
    index=False
)

validation.to_csv(
    OUTPUT_DIR / "news_validation.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR / "news_test.csv",
    index=False
)

print("\nFiles created:")
print("data/processed/news_train.csv")
print("data/processed/news_validation.csv")
print("data/processed/news_test.csv")

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)
