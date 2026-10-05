import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit

from src.prediction.feature_builder import (
    FEATURE_COLUMNS,
    BULLISH_THRESHOLD,
    BEARISH_THRESHOLD,
    TARGET_CLASSES,
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    build_predictive_dataset
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VERSIONS_DIR = MODELS_DIR / "versions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_select_predictive_model(dataset_file=None, promote_to_production=False):
    """
    Trains an expanded suite of hyperparameter-tuned predictive models (Tuned Random Forest,
    ExtraTrees, Tuned XGBoost, Tuned LightGBM, HistGradientBoosting, Stacking Meta-Learner, Soft Voting)
    on chronological train set, evaluates on validation set & TimeSeriesSplit CV, selects the best model,
    tests on held-out test set, and updates model artifacts.
    """
    if dataset_file is None:
        dataset_file = DATA_DIR / "prediction_dataset.csv"

    if not Path(dataset_file).exists():
        build_predictive_dataset()

    df = pd.read_csv(dataset_file)
    df["date"] = pd.to_datetime(df["date"])

    # Drop un-labeled row (the last row where future return t+1 is not yet known)
    labeled_df = df.dropna(subset=["target_label"]).copy()
    labeled_df["target_label"] = labeled_df["target_label"].astype(int)

    n_samples = len(labeled_df)

    # 1. Chronological Train (70%) / Validation (15%) / Test (15%) Split
    train_end_idx = int(n_samples * 0.70)
    val_end_idx = int(n_samples * 0.85)

    train_df = labeled_df.iloc[:train_end_idx].copy()
    val_df = labeled_df.iloc[train_end_idx:val_end_idx].copy()
    test_df = labeled_df.iloc[val_end_idx:].copy()

    print("=" * 75, flush=True)
    print("OPTIMIZED CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT (2021 - 2026)", flush=True)
    print("=" * 75, flush=True)
    print(f"TRAIN      : {len(train_df)} samples ({train_df['date'].min().date()} → {train_df['date'].max().date()})", flush=True)
    print(f"VALIDATION : {len(val_df)} samples ({val_df['date'].min().date()} → {val_df['date'].max().date()})", flush=True)
    print(f"TEST       : {len(test_df)} samples ({test_df['date'].min().date()} → {test_df['date'].max().date()})", flush=True)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target_label"].values

    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["target_label"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target_label"].values

    # Scale features for models requiring scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Compute sample weights for unweighted models like XGBoost
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    sample_weights_train = np.array([class_weights[y] for y in y_train])

    # 2. Define High-Performance Hyperparameter-Tuned Models
    rf_tuned = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        criterion="entropy",
        random_state=42,
        class_weight="balanced",
        n_jobs=2
    )

    et_tuned = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        criterion="entropy",
        random_state=42,
        class_weight="balanced",
        n_jobs=2
    )

    hgb_tuned = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42,
        class_weight="balanced"
    )

    xgb_tuned = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        gamma=0.1,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=2
    )

    lgb_tuned = LGBMClassifier(
        n_estimators=250,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
        n_jobs=2
    )

    lr_tuned = LogisticRegression(
        C=0.1,
        max_iter=1000,
        penalty="l2",
        random_state=42,
        class_weight="balanced"
    )

    # Soft Voting Ensemble
    voting_ensemble = VotingClassifier(
        estimators=[
            ("rf", rf_tuned),
            ("et", et_tuned),
            ("xgb", xgb_tuned),
            ("lgb", lgb_tuned),
            ("hgb", hgb_tuned),
            ("lr", lr_tuned)
        ],
        voting="soft",
        n_jobs=2
    )

    # Stacking Meta-Learner Classifier
    stacking_ensemble = StackingClassifier(
        estimators=[
            ("rf", rf_tuned),
            ("et", et_tuned),
            ("xgb", xgb_tuned),
            ("lgb", lgb_tuned),
            ("hgb", hgb_tuned)
        ],
        final_estimator=LogisticRegression(C=0.1, class_weight="balanced", random_state=42),
        cv=5,
        n_jobs=2
    )

    candidates = {
        "Random Forest (Entropy Tuned)": (rf_tuned, False),
        "ExtraTrees (Tuned)": (et_tuned, False),
        "HistGradientBoosting (Regularized)": (hgb_tuned, False),
        "XGBoost (Tuned)": (xgb_tuned, False),
        "LightGBM (Tuned)": (lgb_tuned, False),
        "Logistic Regression (L2 Tuned)": (lr_tuned, True),
        "Soft Voting Ensemble": (voting_ensemble, True),
        "Stacking Meta-Learner": (stacking_ensemble, True)
    }

    results = {}
    best_model_name = None
    best_val_f1 = -1.0
    best_model_obj = None
    best_is_scaled = False

    print("\n" + "=" * 75, flush=True)
    print("MODEL VALIDATION EVALUATION (EXPANDED HYPERPARAMETER CANDIDATES)", flush=True)
    print("=" * 75, flush=True)

    for name, (model, is_scaled) in candidates.items():
        X_tr = X_train_scaled if is_scaled else X_train
        X_v = X_val_scaled if is_scaled else X_val

        if name == "XGBoost (Tuned)":
            model.fit(X_tr, y_train, sample_weight=sample_weights_train)
        else:
            model.fit(X_tr, y_train)

        y_val_pred = model.predict(X_v)

        acc = accuracy_score(y_val, y_val_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average="weighted", zero_division=0)

        results[name] = {
            "val_accuracy": float(acc),
            "val_precision": float(prec),
            "val_recall": float(rec),
            "val_f1": float(f1),
            "model_obj": model,
            "is_scaled": is_scaled
        }

        print(f"Model: {name:<35} | Val Acc: {acc * 100:.2f}% | Val F1: {f1 * 100:.2f}% | Val Prec: {prec * 100:.2f}% | Val Rec: {rec * 100:.2f}%", flush=True)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_name = name
            best_model_obj = model
            best_is_scaled = is_scaled

    print("\n" + "=" * 75, flush=True)
    print(f"PROMOTED PRODUCTION MODEL: {best_model_name} (Val F1: {best_val_f1 * 100:.2f}%)", flush=True)
    print("=" * 75, flush=True)

    # Final Evaluation on Held-Out Test Set
    X_t = X_test_scaled if best_is_scaled else X_test
    y_test_pred = best_model_obj.predict(X_t)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted", zero_division=0)

    print("\nFINAL HELD-OUT TEST SET PERFORMANCE:", flush=True)
    print(f"Test Accuracy : {test_acc * 100:.2f}%", flush=True)
    print(f"Test Precision: {test_prec * 100:.2f}%", flush=True)
    print(f"Test Recall   : {test_rec * 100:.2f}%", flush=True)
    print(f"Test F1-Score : {test_f1 * 100:.2f}%\n", flush=True)
    print("Test Classification Report:", flush=True)
    print(classification_report(y_test, y_test_pred, target_names=["BEARISH", "NEUTRAL", "BULLISH"], zero_division=0), flush=True)

    # Save artifacts into candidate staging area first
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = MODELS_DIR / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    cand_model_path = staging_dir / "candidate_model.joblib"
    cand_scaler_path = staging_dir / "candidate_scaler.joblib"
    cand_metadata_path = staging_dir / "candidate_metadata.json"

    joblib.dump(best_model_obj, cand_model_path)
    joblib.dump(scaler, cand_scaler_path)

    metadata = {
        "model_name": best_model_name,
        "is_scaled": best_is_scaled,
        "training_timestamp": timestamp_str,
        "bullish_threshold": BULLISH_THRESHOLD,
        "bearish_threshold": BEARISH_THRESHOLD,
        "feature_schema": FEATURE_COLUMNS,
        "class_mapping": CLASS_TO_LABEL,
        "date_ranges": {
            "train": f"{train_df['date'].min().date()} to {train_df['date'].max().date()}",
            "validation": f"{val_df['date'].min().date()} to {val_df['date'].max().date()}",
            "test": f"{test_df['date'].min().date()} to {test_df['date'].max().date()}"
        },
        "metrics": {
            "val_accuracy": results[best_model_name]["val_accuracy"],
            "val_f1": results[best_model_name]["val_f1"],
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec)
        },
        "all_candidate_metrics": {
            k: {"val_accuracy": v["val_accuracy"], "val_f1": v["val_f1"], "val_precision": v["val_precision"], "val_recall": v["val_recall"]} for k, v in results.items()
        },
        "model_version": f"model_{timestamp_str}.joblib"
    }

    with open(cand_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCandidate model staged at: {cand_model_path}", flush=True)

    # If promotion is requested or no production model exists, promote candidate immediately
    prod_model_path = MODELS_DIR / "predictive_model.joblib"
    if promote_to_production or not prod_model_path.exists():
        promote_candidate_to_production(staging_dir=staging_dir)

    return best_model_obj, scaler, metadata


def promote_candidate_to_production(staging_dir=None):
    """
    Atomically promotes staged candidate model files into active production storage.
    """
    if staging_dir is None:
        staging_dir = MODELS_DIR / "staging"

    cand_model_path = staging_dir / "candidate_model.joblib"
    cand_scaler_path = staging_dir / "candidate_scaler.joblib"
    cand_metadata_path = staging_dir / "candidate_metadata.json"

    if not cand_model_path.exists() or not cand_metadata_path.exists():
        raise FileNotFoundError(f"Staged candidate artifacts not found in: {staging_dir}")

    with open(cand_metadata_path, "r") as f:
        metadata = json.load(f)

    timestamp_str = metadata.get("training_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    prod_model_path = MODELS_DIR / "predictive_model.joblib"
    current_model_path = MODELS_DIR / "current_model.joblib"
    version_model_path = VERSIONS_DIR / f"model_{timestamp_str}.joblib"
    prod_scaler_path = MODELS_DIR / "predictive_scaler.joblib"
    prod_metadata_path = MODELS_DIR / "predictive_model_metadata.json"

    cand_model = joblib.load(cand_model_path)
    joblib.dump(cand_model, prod_model_path)
    joblib.dump(cand_model, current_model_path)
    joblib.dump(cand_model, version_model_path)

    if cand_scaler_path.exists():
        cand_scaler = joblib.load(cand_scaler_path)
        joblib.dump(cand_scaler, prod_scaler_path)

    with open(prod_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ PROMOTED CANDIDATE MODEL TO PRODUCTION:", flush=True)
    print(f"- Production Model : {prod_model_path}", flush=True)
    print(f"- Versioned Backup  : {version_model_path}", flush=True)
    print(f"- Model Metadata    : {prod_metadata_path}", flush=True)
    return metadata


if __name__ == "__main__":
    train_and_select_predictive_model(promote_to_production=True)
