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
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from src.prediction.feature_builder import (
    FEATURE_COLUMNS_BTC,
    FEATURE_COLUMNS_ETH,
    BULLISH_THRESHOLD,
    BEARISH_THRESHOLD,
    CLASS_TO_LABEL,
    build_predictive_dataset
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VERSIONS_DIR = MODELS_DIR / "versions"
CANDIDATES_DIR = MODELS_DIR / "candidates"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)


def train_and_select_predictive_model(asset="BTC", dataset_file=None, promote_to_production=False, models_dir=None, candidates_dir=None, versions_dir=None):
    """
    Trains hyperparameter-tuned candidate models for asset (BTC/ETH), evaluates validation performance,
    stages candidate model in models/candidates/, and optionally promotes to production if requested.
    """
    asset_key = asset.upper()
    feature_schema = FEATURE_COLUMNS_ETH if asset_key == "ETH" else FEATURE_COLUMNS_BTC

    target_models_dir = Path(models_dir) if models_dir else MODELS_DIR
    target_candidates_dir = Path(candidates_dir) if candidates_dir else CANDIDATES_DIR
    target_versions_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR

    target_models_dir.mkdir(parents=True, exist_ok=True)
    target_candidates_dir.mkdir(parents=True, exist_ok=True)
    target_versions_dir.mkdir(parents=True, exist_ok=True)

    if dataset_file is None:
        dataset_file = DATA_DIR / ("prediction_dataset_eth.csv" if asset_key == "ETH" else "prediction_dataset.csv")

    if not Path(dataset_file).exists():
        build_predictive_dataset(asset=asset_key)

    df = pd.read_csv(dataset_file)
    df["date"] = pd.to_datetime(df["date"])

    labeled_df = df.dropna(subset=["target_label"]).copy()
    labeled_df["target_label"] = labeled_df["target_label"].astype(int)

    n_samples = len(labeled_df)

    # Chronological Train (70%) / Validation (15%) / Test (15%) Split
    train_end_idx = int(n_samples * 0.70)
    val_end_idx = int(n_samples * 0.85)

    train_df = labeled_df.iloc[:train_end_idx].copy()
    val_df = labeled_df.iloc[train_end_idx:val_end_idx].copy()
    test_df = labeled_df.iloc[val_end_idx:].copy()

    print("=" * 75, flush=True)
    print(f"[{asset_key}] CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT (2021 - 2026)", flush=True)
    print("=" * 75, flush=True)
    print(f"TRAIN      : {len(train_df)} samples ({train_df['date'].min().date()} → {train_df['date'].max().date()})", flush=True)
    print(f"VALIDATION : {len(val_df)} samples ({val_df['date'].min().date()} → {val_df['date'].max().date()})", flush=True)
    print(f"TEST       : {len(test_df)} samples ({test_df['date'].min().date()} → {test_df['date'].max().date()})", flush=True)

    X_train = train_df[feature_schema].values
    y_train = train_df["target_label"].values

    X_val = val_df[feature_schema].values
    y_val = val_df["target_label"].values

    X_test = test_df[feature_schema].values
    y_test = test_df["target_label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    class_counts = np.bincount(y_train, minlength=3)
    total_samples = len(y_train)
    class_weights = np.where(class_counts > 0, total_samples / (3.0 * class_counts), 1.0)
    sample_weights_train = np.array([class_weights[y] for y in y_train])

    rf_tuned = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", criterion="entropy", random_state=42, class_weight="balanced", n_jobs=2
    )
    et_tuned = ExtraTreesClassifier(
        n_estimators=300, max_depth=10, min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", criterion="entropy", random_state=42, class_weight="balanced", n_jobs=2
    )
    hgb_tuned = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.03, max_depth=5, min_samples_leaf=15,
        l2_regularization=1.0, random_state=42, class_weight="balanced"
    )
    xgb_tuned = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.02, subsample=0.75,
        colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
        random_state=42, eval_metric="mlogloss", n_jobs=2
    )
    lgb_tuned = LGBMClassifier(
        n_estimators=250, max_depth=4, num_leaves=15, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=1.5,
        random_state=42, class_weight="balanced", verbosity=-1, n_jobs=2
    )
    lr_tuned = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight="balanced")

    candidates = {
        "Random Forest (Entropy Tuned)": (rf_tuned, False),
        "ExtraTrees (Tuned)": (et_tuned, False),
        "HistGradientBoosting (Regularized)": (hgb_tuned, False),
        "XGBoost (Tuned)": (xgb_tuned, False),
        "LightGBM (Tuned)": (lgb_tuned, False),
        "Logistic Regression (L2 Tuned)": (lr_tuned, True)
    }

    results = {}
    best_model_name = None
    best_val_f1 = -1.0
    best_model_obj = None
    best_is_scaled = False

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

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_name = name
            best_model_obj = model
            best_is_scaled = is_scaled

    print(f"PROMOTED CANDIDATE MODEL [{asset_key}]: {best_model_name} (Val F1: {best_val_f1 * 100:.2f}%)", flush=True)

    X_t = X_test_scaled if best_is_scaled else X_test
    y_test_pred = best_model_obj.predict(X_t)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted", zero_division=0)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    cand_model_path = target_candidates_dir / f"candidate_model_{asset_key.lower()}.joblib"
    cand_scaler_path = target_candidates_dir / f"candidate_scaler_{asset_key.lower()}.joblib"
    cand_metadata_path = target_candidates_dir / f"candidate_metadata_{asset_key.lower()}.json"

    joblib.dump(best_model_obj, cand_model_path)
    joblib.dump(scaler, cand_scaler_path)

    metadata = {
        "asset": asset_key,
        "model_name": best_model_name,
        "is_scaled": best_is_scaled,
        "training_timestamp": timestamp_str,
        "prediction_horizon": f"Predict {asset_key}'s next 24-hour direction",
        "bullish_threshold": BULLISH_THRESHOLD,
        "bearish_threshold": BEARISH_THRESHOLD,
        "feature_schema": feature_schema,
        "class_mapping": CLASS_TO_LABEL,
        "metrics": {
            "val_accuracy": results[best_model_name]["val_accuracy"],
            "val_f1": results[best_model_name]["val_f1"],
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec)
        },
        "model_version": f"model_{asset_key.lower()}_{timestamp_str}.joblib"
    }

    with open(cand_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Staged candidate model into: {cand_model_path}", flush=True)

    prod_model_path = target_models_dir / f"predictive_model_{asset_key.lower()}.joblib"
    if promote_to_production or not prod_model_path.exists():
        promote_candidate_to_production(asset=asset_key, candidates_dir=target_candidates_dir, models_dir=target_models_dir, versions_dir=target_versions_dir)

    return best_model_obj, scaler, metadata


def promote_candidate_to_production(asset="BTC", candidates_dir=None, models_dir=None, versions_dir=None):
    """
    Only copies a staged candidate into the production model path after passing promotion criteria.
    """
    asset_key = asset.upper()
    cand_dir = Path(candidates_dir) if candidates_dir else CANDIDATES_DIR
    prod_dir = Path(models_dir) if models_dir else MODELS_DIR
    ver_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR

    prod_dir.mkdir(parents=True, exist_ok=True)
    ver_dir.mkdir(parents=True, exist_ok=True)

    cand_model_path = cand_dir / f"candidate_model_{asset_key.lower()}.joblib"
    cand_scaler_path = cand_dir / f"candidate_scaler_{asset_key.lower()}.joblib"
    cand_metadata_path = cand_dir / f"candidate_metadata_{asset_key.lower()}.json"

    if not cand_model_path.exists() or not cand_metadata_path.exists():
        raise FileNotFoundError(f"Staged candidate files missing for {asset_key} in: {cand_dir}")

    with open(cand_metadata_path, "r") as f:
        metadata = json.load(f)

    timestamp_str = metadata.get("training_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Production targets
    prod_model_asset = prod_dir / f"predictive_model_{asset_key.lower()}.joblib"
    prod_scaler_asset = prod_dir / f"predictive_scaler_{asset_key.lower()}.joblib"
    prod_metadata_asset = prod_dir / f"predictive_model_metadata_{asset_key.lower()}.json"

    version_model_path = ver_dir / f"model_{asset_key.lower()}_{timestamp_str}.joblib"

    cand_model = joblib.load(cand_model_path)
    joblib.dump(cand_model, prod_model_asset)
    joblib.dump(cand_model, version_model_path)

    if cand_scaler_path.exists():
        cand_scaler = joblib.load(cand_scaler_path)
        joblib.dump(cand_scaler, prod_scaler_asset)

    with open(prod_metadata_asset, "w") as f:
        json.dump(metadata, f, indent=2)

    # For BTC default compatibility
    if asset_key == "BTC":
        joblib.dump(cand_model, prod_dir / "predictive_model.joblib")
        if cand_scaler_path.exists():
            joblib.dump(cand_scaler, prod_dir / "predictive_scaler.joblib")
        with open(prod_dir / "predictive_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"✅ Promoted [{asset_key}] candidate into production model path: {prod_model_asset}", flush=True)
    return metadata


if __name__ == "__main__":
    train_and_select_predictive_model(asset="BTC", promote_to_production=True)
    train_and_select_predictive_model(asset="ETH", promote_to_production=True)
