import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# CONFIGURATION
# ============================================================

NEWS_FILE = Path("data/processed/news_clean.csv")
FINBERT_FILE = Path("data/processed/news_finbert.csv")

OUTPUT_FILE = Path(
    "data/processed/keyword_analysis.csv"
)

TOP_N = 30


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CRYPTOCURRENCY KEYWORD ANALYSIS")
print("=" * 60)

news = pd.read_csv(NEWS_FILE)

finbert = pd.read_csv(
    FINBERT_FILE,
    usecols=[
        "timestamp",
        "title",
        "finbert_label",
        "finbert_sentiment_score"
    ]
)

print("\nNews records:", len(news))
print("FinBERT records:", len(finbert))


# ============================================================
# VERIFY DATASET ALIGNMENT
# ============================================================

if len(news) != len(finbert):
    raise ValueError(
        f"Dataset length mismatch: "
        f"news={len(news)}, finbert={len(finbert)}"
    )

# Verify timestamp and title correspond row-by-row
timestamp_match = (
    news["timestamp"].astype(str).values
    == finbert["timestamp"].astype(str).values
)

title_match = (
    news["title"].fillna("").astype(str).values
    == finbert["title"].fillna("").astype(str).values
)

if not timestamp_match.all():
    raise ValueError(
        "Timestamp alignment check failed."
    )

if not title_match.all():
    raise ValueError(
        "Title alignment check failed."
    )

print("Dataset alignment: VERIFIED")


# ============================================================
# ATTACH FINBERT RESULTS
# ============================================================

df = news.copy()

df["finbert_label"] = (
    finbert["finbert_label"].values
)

df["finbert_sentiment_score"] = (
    finbert["finbert_sentiment_score"].values
)

df["model_text"] = (
    df["model_text"]
    .fillna("")
    .astype(str)
)

print("Merged records:", len(df))


# ============================================================
# TF-IDF KEYWORD FUNCTION
# ============================================================

def extract_keywords(texts, category):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.90,
        sublinear_tf=True,
        max_features=50000
    )

    matrix = vectorizer.fit_transform(texts)

    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    top_indices = scores.argsort()[::-1][:TOP_N]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        results.append(
            {
                "category": category,
                "rank": rank,
                "keyword": terms[index],
                "tfidf_score": scores[index]
            }
        )

    return results


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

results = []


# ------------------------------------------------------------
# Overall
# ------------------------------------------------------------

print("\nExtracting overall keywords...")

results.extend(
    extract_keywords(
        df["model_text"],
        "overall"
    )
)


# ------------------------------------------------------------
# FinBERT sentiment groups
# ------------------------------------------------------------

for sentiment in [
    "positive",
    "negative",
    "neutral"
]:

    subset = df[
        df["finbert_label"] == sentiment
    ]

    print(
        f"\nExtracting {sentiment} keywords..."
    )

    print(
        "Records:",
        len(subset)
    )

    if len(subset) == 0:
        continue

    results.extend(
        extract_keywords(
            subset["model_text"],
            sentiment
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TOP KEYWORDS")
print("=" * 60)

for category in [
    "overall",
    "positive",
    "negative",
    "neutral"
]:

    print(
        f"\n--- {category.upper()} ---"
    )

    category_df = results_df[
        results_df["category"] == category
    ]

    print(
        category_df[
            [
                "rank",
                "keyword",
                "tfidf_score"
            ]
        ].to_string(index=False)
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    "\nTotal keyword records:",
    len(results_df)
)

print("\nKeyword counts by category:")
print(
    results_df["category"].value_counts()
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("KEYWORD ANALYSIS COMPLETE")
print("=" * 60)
