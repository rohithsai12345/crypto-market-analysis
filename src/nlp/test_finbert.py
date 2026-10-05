from transformers import pipeline


MODEL_NAME = "ProsusAI/finbert"

print("=" * 60)
print("FINBERT TEST")
print("=" * 60)

print("\nLoading FinBERT...")

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1
)

examples = [
    "Bitcoin adoption continues to increase among institutional investors.",
    "The cryptocurrency market crashed after a major security exploit.",
    "Bitcoin prices remained stable with low trading activity.",
    "Ethereum announced a major network upgrade."
]

print("\nPredictions:\n")

for text in examples:

    result = classifier(
        text,
        truncation=True
    )[0]

    print("Text:", text)
    print("Label:", result["label"])
    print("Score:", round(result["score"], 4))
    print("-" * 60)
