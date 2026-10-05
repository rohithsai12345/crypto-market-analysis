import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.prediction.feature_builder import build_predictive_dataset
from src.prediction.train_predictive_model import (
    train_and_select_predictive_model,
    promote_candidate_to_production
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VERSIONS_DIR = MODELS_DIR / "versions"
CANDIDATES_DIR = MODELS_DIR / "candidates"


def run_continuous_retraining(asset="BTC", models_dir=None, candidates_dir=None, versions_dir=None, dataset_file=None, market_sentiment_file=None):
    """
    Executes periodic continuous retraining workflow for target asset:
    1. Re-builds predictive dataset with newly resolved labels.
    2. Trains candidate models in models/candidates/.
    3. Promotes candidate model ONLY if validation performance beats current production model.
    """
    asset_key = asset.upper()
    print("=" * 70)
    print(f"[{asset_key}] PERIODIC CONTINUOUS RETRAINING PIPELINE")
    print("=" * 70)

    target_models_dir = Path(models_dir) if models_dir else MODELS_DIR
    target_candidates_dir = Path(candidates_dir) if candidates_dir else CANDIDATES_DIR
    target_versions_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR

    target_models_dir.mkdir(parents=True, exist_ok=True)
    target_candidates_dir.mkdir(parents=True, exist_ok=True)
    target_versions_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rebuild feature dataset if requested or default
    print(f"Step 1: Rebuilding feature dataset for {asset_key}...")
    if dataset_file is not None or market_sentiment_file is not None:
        df_data = build_predictive_dataset(
            asset=asset_key,
            market_sentiment_file=market_sentiment_file if market_sentiment_file else DATA_DIR / "market_sentiment_analysis.csv",
            output_file=dataset_file
        )
    else:
        df_data = build_predictive_dataset(asset=asset_key)

    # 2. Check current production model metadata
    metadata_path = target_models_dir / (f"predictive_model_metadata_{asset_key.lower()}.json" if asset_key == "ETH" else "predictive_model_metadata.json")
    current_val_f1 = 0.0

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            current_val_f1 = float(meta.get("metrics", {}).get("val_f1", 0.0))

    print(f"Current Production Model Validation F1 [{asset_key}]: {current_val_f1 * 100:.2f}%")

    # 3. Train candidate model into models/candidates/
    print("Step 2: Training candidate model in models/candidates/...")
    candidate_model, candidate_scaler, candidate_meta = train_and_select_predictive_model(
        asset=asset_key,
        dataset_file=dataset_file,
        promote_to_production=False,
        models_dir=target_models_dir,
        candidates_dir=target_candidates_dir,
        versions_dir=target_versions_dir
    )
    candidate_val_f1 = float(candidate_meta.get("metrics", {}).get("val_f1", 0.0))

    # 4. Model Promotion Gate Rule
    print("\n" + "=" * 70)
    print(f"MODEL PROMOTION RULE EVALUATION [{asset_key}]")
    print("=" * 70)
    print(f"Candidate Model F1 : {candidate_val_f1 * 100:.2f}%")
    print(f"Current Model F1   : {current_val_f1 * 100:.2f}%")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if candidate_val_f1 >= current_val_f1 or current_val_f1 == 0.0:
        promote_candidate_to_production(
            asset=asset_key,
            candidates_dir=target_candidates_dir,
            models_dir=target_models_dir,
            versions_dir=target_versions_dir
        )
        print(f"RESULT: PROMOTED! Candidate model passed promotion criteria for {asset_key}.")
        promotion_status = "PROMOTED"
    else:
        rejected_path = target_versions_dir / f"rejected_candidate_{asset_key.lower()}_{timestamp_str}.joblib"
        joblib.dump(candidate_model, rejected_path)
        print(f"RESULT: REJECTED! Candidate model degraded validation performance ({candidate_val_f1 * 100:.2f}% vs {current_val_f1 * 100:.2f}%). Active production model retained.")
        promotion_status = "REJECTED_DEGRADATION"

    retrain_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asset": asset_key,
        "promotion_status": promotion_status,
        "current_val_f1": current_val_f1,
        "candidate_val_f1": candidate_val_f1,
        "promoted_model_version": candidate_meta.get("model_version") if promotion_status == "PROMOTED" else None
    }

    return retrain_summary


if __name__ == "__main__":
    run_continuous_retraining(asset="BTC")
    run_continuous_retraining(asset="ETH")
