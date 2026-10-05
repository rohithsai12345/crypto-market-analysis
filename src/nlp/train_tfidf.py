import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

train = pd.read_csv(DATA_DIR / "news_train.csv")
validation = pd.read_csv(DATA_DIR / "news_validation.csv")
test = pd.read_csv(DATA_DIR / "news_test.csv")

X_train = train["model_text"].fillna("")
y_train = train["market_direction"]

X_val = validation["model_text"].fillna("")
y_val = validation["market_direction"]

X_test = test["model_text"].fillna("")
y_test = test["market_direction"]

print("\nDataset sizes:")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

# IMPORTANT:
# Validation and test are transformed using the vocabulary
# learned ONLY from the training data.
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("Training matrix:", X_train_tfidf.shape)
print("Validation matrix:", X_val_tfidf.shape)
print("Test matrix:", X_test_tfidf.shape)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

val_predictions = model.predict(X_val_tfidf)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision, val_recall, val_f1, _ = (
    precision_recall_fscore_support(
        y_val,
        val_predictions,
        average="weighted",
        zero_division=0
    )
)

print(f"\nAccuracy : {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall   : {val_recall:.4f}")
print(f"F1-score : {val_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        val_predictions,
        target_names=[
            "Neutral",
            "Bearish",
            "Bullish"
        ],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_val,
        val_predictions
    )
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_predictions = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision, test_recall, test_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0
    )
)

print(f"\nAccuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Neutral",
            "Bearish",
            "Bullish"
        ],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

joblib.dump(
    model,
    MODEL_DIR / "logistic_regression.joblib"
)

print("\nModels saved:")
print("models/tfidf_vectorizer.joblib")
print("models/logistic_regression.joblib")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
