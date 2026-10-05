import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("TF-IDF + LINEAR SVM")
print("=" * 60)


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train = pd.read_csv(DATA_DIR / "news_train.csv")
validation = pd.read_csv(DATA_DIR / "news_validation.csv")
test = pd.read_csv(DATA_DIR / "news_test.csv")

X_train = train["model_text"].fillna("")
y_train = train["market_direction"]

X_val = validation["model_text"].fillna("")
y_val = validation["market_direction"]

X_test = test["model_text"].fillna("")
y_test = test["market_direction"]


# ------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining matrix:", X_train_tfidf.shape)


# ------------------------------------------------------------
# Linear SVM
# ------------------------------------------------------------

print("\nTraining Linear SVM...")

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------

def evaluate(name, X, y):

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Neutral",
                "Bearish",
                "Bullish"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    return accuracy, precision, recall, f1


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

evaluate(
    "VALIDATION RESULTS",
    X_val_tfidf,
    y_val
)


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

evaluate(
    "TEST RESULTS",
    X_test_tfidf,
    y_test
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

joblib.dump(
    vectorizer,
    MODEL_DIR / "svm_tfidf_vectorizer.joblib"
)

joblib.dump(
    model,
    MODEL_DIR / "linear_svm.joblib"
)

print("\nModels saved:")
print("models/svm_tfidf_vectorizer.joblib")
print("models/linear_svm.joblib")

print("\n" + "=" * 60)
print("SVM TRAINING COMPLETE")
print("=" * 60)
