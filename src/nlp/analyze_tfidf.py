import pandas as pd
import joblib
import numpy as np

from pathlib import Path


MODEL_DIR = Path("models")

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

model = joblib.load(
    MODEL_DIR / "logistic_regression.joblib"
)

features = np.array(
    vectorizer.get_feature_names_out()
)

classes = model.classes_

print("=" * 60)
print("TF-IDF FEATURE ANALYSIS")
print("=" * 60)

for class_index, class_label in enumerate(classes):

    coefficients = model.coef_[class_index]

    # Highest positive coefficients
    top_indices = np.argsort(coefficients)[-20:][::-1]

    print("\n" + "=" * 60)

    if class_label == 0:
        name = "NEUTRAL"
    elif class_label == 1:
        name = "BEARISH"
    else:
        name = "BULLISH"

    print(f"TOP FEATURES FOR {name}")

    print("=" * 60)

    for index in top_indices:
        print(
            f"{features[index]:30s} "
            f"{coefficients[index]:.4f}"
        )
