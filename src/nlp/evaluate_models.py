import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


def evaluate_all_models():
    """
    Evaluates baseline ML models (Linear SVM & Logistic Regression) and FinBERT transformer model.
    Prints accuracy, precision, recall, f1-score, and confusion matrices.
    """
    print("=" * 70)
    print("NLP SENTIMENT MODEL EVALUATION REPORT")
    print("=" * 70)

    results = {}

    # 1. Load Data
    clean_news_path = DATA_DIR / "news_clean.csv"
    finbert_path = DATA_DIR / "news_finbert.csv"

    if not clean_news_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {clean_news_path}")

    df = pd.read_csv(clean_news_path)
    
    # Map numerical market direction to sentiment labels (-1: negative, 0: neutral, 1: positive)
    if "market_direction" in df.columns:
        direction_map = {-1: "negative", 0: "neutral", 1: "positive"}
        y_true_str = df["market_direction"].map(direction_map).fillna("neutral")
        y_true_num = df["market_direction"]
    else:
        y_true_str = pd.Series(["neutral"] * len(df))
        y_true_num = pd.Series([0] * len(df))

    texts = df["model_text"].fillna("").astype(str)

    # ------------------------------------------------------------
    # A. Linear SVM Model Evaluation
    # ------------------------------------------------------------
    svm_path = MODELS_DIR / "linear_svm.joblib"
    svm_vec_path = MODELS_DIR / "svm_tfidf_vectorizer.joblib"

    if svm_path.exists() and svm_vec_path.exists():
        print("\n" + "-" * 70)
        print("1. LINEAR SVM CLASSIFIER")
        print("-" * 70)

        svm = joblib.load(svm_path)
        svm_vec = joblib.load(svm_vec_path)

        X_svm = svm_vec.transform(texts)
        y_pred_svm = svm.predict(X_svm)

        # Convert numerical predictions if needed
        if isinstance(y_pred_svm[0], (int, np.integer)):
            y_pred_svm_str = pd.Series(y_pred_svm).map(direction_map).fillna("neutral")
        else:
            y_pred_svm_str = y_pred_svm

        acc_svm = accuracy_score(y_true_str, y_pred_svm_str)
        p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_true_str, y_pred_svm_str, average="weighted")

        print(f"Accuracy : {acc_svm * 100:.2f}%")
        print(f"Precision: {p_svm * 100:.2f}%")
        print(f"Recall   : {r_svm * 100:.2f}%")
        print(f"F1-Score : {f1_svm * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_true_str, y_pred_svm_str, digits=4))
        print("Confusion Matrix:")
        cm_svm = confusion_matrix(y_true_str, y_pred_svm_str, labels=["negative", "neutral", "positive"])
        print(pd.DataFrame(cm_svm, index=["True Neg", "True Neu", "True Pos"], columns=["Pred Neg", "Pred Neu", "Pred Pos"]))

        results["Linear SVM"] = {
            "Accuracy": acc_svm,
            "Precision": p_svm,
            "Recall": r_svm,
            "F1-Score": f1_svm
        }

    # ------------------------------------------------------------
    # B. Logistic Regression Model Evaluation
    # ------------------------------------------------------------
    lr_path = MODELS_DIR / "logistic_regression.joblib"
    lr_vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"

    if lr_path.exists() and lr_vec_path.exists():
        print("\n" + "-" * 70)
        print("2. LOGISTIC REGRESSION CLASSIFIER")
        print("-" * 70)

        lr = joblib.load(lr_path)
        lr_vec = joblib.load(lr_vec_path)

        X_lr = lr_vec.transform(texts)
        y_pred_lr = lr.predict(X_lr)

        if isinstance(y_pred_lr[0], (int, np.integer)):
            y_pred_lr_str = pd.Series(y_pred_lr).map(direction_map).fillna("neutral")
        else:
            y_pred_lr_str = y_pred_lr

        acc_lr = accuracy_score(y_true_str, y_pred_lr_str)
        p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(y_true_str, y_pred_lr_str, average="weighted")

        print(f"Accuracy : {acc_lr * 100:.2f}%")
        print(f"Precision: {p_lr * 100:.2f}%")
        print(f"Recall   : {r_lr * 100:.2f}%")
        print(f"F1-Score : {f1_lr * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_true_str, y_pred_lr_str, digits=4))

        results["Logistic Regression"] = {
            "Accuracy": acc_lr,
            "Precision": p_lr,
            "Recall": r_lr,
            "F1-Score": f1_lr
        }

    # ------------------------------------------------------------
    # C. FinBERT Financial Transformer Model Evaluation
    # ------------------------------------------------------------
    if finbert_path.exists():
        print("\n" + "-" * 70)
        print("3. FINBERT TRANSFORMER MODEL (ProsusAI/finbert)")
        print("-" * 70)

        finbert_df = pd.read_csv(finbert_path)
        y_pred_finbert = finbert_df["finbert_label"].fillna("neutral")

        acc_fb = accuracy_score(y_true_str, y_pred_finbert)
        p_fb, r_fb, f1_fb, _ = precision_recall_fscore_support(y_true_str, y_pred_finbert, average="weighted")

        print(f"Accuracy : {acc_fb * 100:.2f}%")
        print(f"Precision: {p_fb * 100:.2f}%")
        print(f"Recall   : {r_fb * 100:.2f}%")
        print(f"F1-Score : {f1_fb * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_true_str, y_pred_finbert, digits=4))
        print("Confusion Matrix:")
        cm_fb = confusion_matrix(y_true_str, y_pred_finbert, labels=["negative", "neutral", "positive"])
        print(pd.DataFrame(cm_fb, index=["True Neg", "True Neu", "True Pos"], columns=["Pred Neg", "Pred Neu", "Pred Pos"]))

        results["FinBERT Transformer"] = {
            "Accuracy": acc_fb,
            "Precision": p_fb,
            "Recall": r_fb,
            "F1-Score": f1_fb
        }

    # Summary Comparison Table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(results).T * 100
    summary_df = summary_df.round(2)
    print(summary_df.to_string())

    # Save metrics to CSV
    summary_df.to_csv(DATA_DIR / "model_evaluation_metrics.csv")
    print(f"\nSaved evaluation report to: {DATA_DIR / 'model_evaluation_metrics.csv'}")

    return summary_df


if __name__ == "__main__":
    evaluate_all_models()
