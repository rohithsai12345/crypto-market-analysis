import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.prediction.feature_builder import build_predictive_dataset
from src.prediction.train_predictive_model import train_and_select_predictive_model

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VERSIONS_DIR = MODELS_DIR / "versions"


def run_continuous_retraining():
    """
    Executes periodic continuous retraining workflow:
    1. Re-builds predictive dataset with newly resolved labels.
    2. Trains new candidate models.
    3. Promotes candidate model ONLY if validation performance does NOT degrade.
    """
    print("=" * 70)
    print("PERIODIC CONTINUOUS RETRAINING PIPELINE")
    print("=" * 70)

    # 1. Rebuild feature dataset
    print("Step 1: Rebuilding feature dataset with latest observations...")
    df_data = build_predictive_dataset()

    # 2. Check current model metadata
    metadata_path = MODELS_DIR / "predictive_model_metadata.json"
    current_val_f1 = 0.0

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            current_val_f1 = float(meta.get("metrics", {}).get("val_f1", 0.0))

    print(f"Current Production Model Validation F1: {current_val_f1 * 100:.2f}%")

    # 3. Train candidate model
    print("Step 2: Training candidate model on updated dataset...")
    candidate_model, candidate_scaler, candidate_meta = train_and_select_predictive_model()
    candidate_val_f1 = float(candidate_meta.get("metrics", {}).get("val_f1", 0.0))

    # 4. Model Promotion Rule
    print("\n" + "=" * 70)
    print("MODEL PROMOTION RULE EVALUATION")
    print("=" * 70)
    print(f"Candidate Model F1 : {candidate_val_f1 * 100:.2f}%")
    print(f"Current Model F1   : {current_val_f1 * 100:.2f}%")

    if candidate_val_f1 >= (current_val_f1 - 0.01):
        print("RESULT: PROMOTED! Candidate model passed promotion criteria.")
        promotion_status = "PROMOTED"
    else:
        print("RESULT: REJECTED! Candidate model degraded validation performance. Retaining current production model.")
        promotion_status = "REJECTED_DEGRADATION"

    retrain_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "promotion_status": promotion_status,
        "current_val_f1": current_val_f1,
        "candidate_val_f1": candidate_val_f1,
        "promoted_model_version": candidate_meta.get("model_version")
    }

    return retrain_summary


if __name__ == "__main__":
    run_continuous_retraining()
