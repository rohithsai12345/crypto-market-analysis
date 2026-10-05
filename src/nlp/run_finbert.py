import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/news_clean.csv")
OUTPUT_FILE = Path("data/processed/news_finbert.csv")

MODEL_NAME = "ProsusAI/finbert"

BATCH_SIZE = 32
MAX_LENGTH = 128
CHECKPOINT_EVERY = 500


# ============================================================
# LOAD INPUT DATA
# ============================================================

print("=" * 60)
print("FINBERT SENTIMENT ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nTotal records:", len(df))


# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

if OUTPUT_FILE.exists():

    old_df = pd.read_csv(OUTPUT_FILE)

    required = [
        "finbert_label",
        "finbert_score",
        "finbert_positive",
        "finbert_negative",
        "finbert_neutral"
    ]

    if not all(col in old_df.columns for col in required):
        raise ValueError(
            "Existing FinBERT file does not contain all required columns."
        )

    completed = len(old_df)

    print("\nExisting checkpoint found.")
    print("Already processed:", completed)

    if completed > len(df):
        raise ValueError(
            "Checkpoint has more rows than the input dataset."
        )

    # Verify that checkpoint is the beginning of the same dataset
    if completed > 0:
        if (
            old_df.iloc[0]["timestamp"]
            != df.iloc[0]["timestamp"]
        ):
            raise ValueError(
                "Checkpoint does not match the current input dataset."
            )

else:

    old_df = None
    completed = 0

    print("\nNo existing checkpoint found.")


# ============================================================
# IF ALREADY COMPLETE
# ============================================================

if completed == len(df):

    print("\nAll records are already processed.")

    if "finbert_sentiment_score" not in old_df.columns:
        old_df["finbert_sentiment_score"] = (
            old_df["finbert_positive"]
            - old_df["finbert_negative"]
        )

    old_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:", OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("FINBERT ANALYSIS COMPLETE")
    print("=" * 60)

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading FinBERT...")

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1
)

print("FinBERT loaded successfully.")


# ============================================================
# REMAINING DATA
# ============================================================

remaining_df = df.iloc[completed:].copy()

texts = (
    remaining_df["model_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

print("\nRemaining records:", len(texts))


# ============================================================
# RESULT STORAGE
# ============================================================

new_labels = []
new_scores = []
new_positive = []
new_negative = []
new_neutral = []


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning inference...")

for start in tqdm(
    range(0, len(texts), BATCH_SIZE),
    desc="FinBERT"
):

    batch = texts[start:start + BATCH_SIZE]

    results = classifier(
        batch,
        truncation=True,
        max_length=MAX_LENGTH,
        top_k=None
    )

    for result in results:

        if isinstance(result, list):
            result_list = result
        else:
            result_list = [result]

        probabilities = {
            item["label"].lower(): item["score"]
            for item in result_list
        }

        positive = probabilities.get("positive", 0.0)
        negative = probabilities.get("negative", 0.0)
        neutral = probabilities.get("neutral", 0.0)

        label = max(
            probabilities,
            key=probabilities.get
        )

        score = probabilities[label]

        new_labels.append(label)
        new_scores.append(score)
        new_positive.append(positive)
        new_negative.append(negative)
        new_neutral.append(neutral)

    # ========================================================
    # CHECKPOINT
    # ========================================================

    processed_now = len(new_labels)

    if (
        processed_now % CHECKPOINT_EVERY < BATCH_SIZE
        or processed_now == len(texts)
    ):

        # Results generated during this run
        new_results = remaining_df.iloc[:processed_now].copy()

        new_results["finbert_label"] = new_labels
        new_results["finbert_score"] = new_scores
        new_results["finbert_positive"] = new_positive
        new_results["finbert_negative"] = new_negative
        new_results["finbert_neutral"] = new_neutral

        # Combine old checkpoint + new results
        if old_df is not None:
            checkpoint = pd.concat(
                [
                    old_df,
                    new_results
                ],
                ignore_index=True
            )
        else:
            checkpoint = new_results

        checkpoint.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"\nCheckpoint saved: "
            f"{len(checkpoint)}/{len(df)}"
        )


# ============================================================
# FINAL DATASET
# ============================================================

print("\nBuilding final dataset...")

new_results = remaining_df.copy()

new_results["finbert_label"] = new_labels
new_results["finbert_score"] = new_scores
new_results["finbert_positive"] = new_positive
new_results["finbert_negative"] = new_negative
new_results["finbert_neutral"] = new_neutral


if old_df is not None:

    final_df = pd.concat(
        [
            old_df,
            new_results
        ],
        ignore_index=True
    )

else:

    final_df = new_results


# ============================================================
# CONTINUOUS SENTIMENT SCORE
# ============================================================

final_df["finbert_sentiment_score"] = (
    final_df["finbert_positive"]
    - final_df["finbert_negative"]
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINBERT RESULTS")
print("=" * 60)

print("\nRecords:", len(final_df))

print("\nLabel distribution:")
print(
    final_df["finbert_label"].value_counts()
)

print("\nMissing values:")
print(
    final_df[
        [
            "finbert_label",
            "finbert_score",
            "finbert_positive",
            "finbert_negative",
            "finbert_neutral",
            "finbert_sentiment_score"
        ]
    ].isna().sum()
)

print("\nSentiment score statistics:")
print(
    final_df["finbert_sentiment_score"].describe()
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("FINBERT ANALYSIS COMPLETE")
print("=" * 60)
