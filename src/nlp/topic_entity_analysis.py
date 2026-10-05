import re
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NEWS_FILE = BASE_DIR / "data" / "processed" / "news_clean.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "topic_entity_analysis.csv"

# Cryptocurrency & Financial Entities to detect
CRYPTO_ENTITIES = [
    "Bitcoin", "Ethereum", "Solana", "Ripple", "XRP", "Cardano", "Binance", "Coinbase",
    "SEC", "ETF", "Fed", "Federal Reserve", "Inflation", "DeFi", "NFT", "Whale", "Regulation"
]


def extract_topics_and_entities(num_topics=5, num_words=6):
    """
    Extracts key topics via NMF topic modeling and counts key entity occurrences.
    """
    if not NEWS_FILE.exists():
        print(f"File not found: {NEWS_FILE}")
        return

    df = pd.read_csv(NEWS_FILE)
    texts = df["model_text"].fillna("").astype(str).tolist()

    print("Running NMF Topic Modeling...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5,
        max_df=0.85,
        sublinear_tf=True,
        max_features=10000
    )

    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=num_topics, random_state=42, init="nndsvda")
    nmf.fit(tfidf)

    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[:-num_words - 1:-1]
        top_words = [feature_names[i] for i in top_features_ind]
        topics.append({
            "topic_id": topic_idx + 1,
            "topic_words": ", ".join(top_words)
        })

    # Entity counts
    full_text = " ".join(texts)
    entity_counts = []
    for entity in CRYPTO_ENTITIES:
        pattern = r"\b" + re.escape(entity) + r"\b"
        matches = len(re.findall(pattern, full_text, flags=re.IGNORECASE))
        entity_counts.append({"entity": entity, "count": matches})

    # Save to CSV
    topics_df = pd.DataFrame(topics)
    entities_df = pd.DataFrame(entity_counts)

    topics_df.to_csv(OUTPUT_FILE.parent / "topics_analysis.csv", index=False)
    entities_df.to_csv(OUTPUT_FILE.parent / "entities_analysis.csv", index=False)

    print("Topics Extracted:")
    print(topics_df.to_string(index=False))

    print("\nTop Entities Detected:")
    print(entities_df.sort_values("count", ascending=False).head(10).to_string(index=False))

    return topics_df, entities_df


if __name__ == "__main__":
    extract_topics_and_entities()
