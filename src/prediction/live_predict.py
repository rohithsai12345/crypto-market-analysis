import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.prediction.feature_builder import (
    FEATURE_COLUMNS,
    BULLISH_THRESHOLD,
    BEARISH_THRESHOLD,
    LABEL_TO_CLASS,
    CLASS_TO_LABEL
)

try:
    from dashboard.services.market_api import get_live_market_data, get_live_coin_market
except ImportError:
    try:
        from services.market_api import get_live_market_data, get_live_coin_market
    except ImportError:
        get_live_market_data = None
        get_live_coin_market = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
PRED_DIR = BASE_DIR / "data" / "predictions"
MODELS_DIR = BASE_DIR / "models"

PRED_DIR.mkdir(parents=True, exist_ok=True)


def generate_live_prediction(asset="BTC", save_to_history=False):
    """
    Infers real-time market prediction using trained model and live data schema.
    Maintains a single active prediction row per asset for the active period.
    When 24-hour time completes, that row is updated with actual outcomes.
    """
    model_path = MODELS_DIR / "predictive_model.joblib"
    scaler_path = MODELS_DIR / "predictive_scaler.joblib"
    metadata_path = MODELS_DIR / "predictive_model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 1. Load latest processed dataset row to obtain historical feature baselines
    dataset_file = DATA_DIR / "prediction_dataset.csv"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file missing: {dataset_file}")

    history_df = pd.read_csv(dataset_file)
    latest_hist = history_df.iloc[-1].to_dict()

    # 2. Ingest real-time live platform data (Asset specific)
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    asset_key = asset.upper()
    if asset_key == "ETH":
        hist_prefix = "eth_"
        coingecko_key = "ethereum"
        default_price = 3300.0
    else:
        hist_prefix = "btc_"
        coingecko_key = "bitcoin"
        default_price = 107287.80

    live_price = float(latest_hist.get(f"{hist_prefix}close", default_price))
    live_return = float(latest_hist.get(f"{hist_prefix}return", 0.0))
    live_vol = float(latest_hist.get(f"{hist_prefix}volume", 15000000000))
    is_live = False

    if callable(get_live_market_data):
        try:
            mkt = get_live_market_data()
            if mkt.get(coingecko_key, {}).get("usd"):
                live_price = float(mkt[coingecko_key]["usd"])
                if mkt[coingecko_key].get("usd_24h_change"):
                    live_return = float(mkt[coingecko_key]["usd_24h_change"]) / 100.0
                if mkt[coingecko_key].get("usd_24h_vol"):
                    live_vol = float(mkt[coingecko_key]["usd_24h_vol"])
                is_live = True
        except Exception:
            pass

    # 3. Construct live feature vector strictly matching FEATURE_COLUMNS
    feature_dict = latest_hist.copy()
    if asset_key == "ETH":
        feature_dict["eth_close"] = live_price
        feature_dict["eth_return"] = live_return
        feature_dict["eth_volume"] = live_vol
        feature_dict["eth_return_lag_1"] = live_return
        if feature_dict.get("btc_close", 0) > 0:
            ratio = feature_dict["btc_close"] / (live_price + 1e-9)
            feature_dict["btc_eth_ratio_return"] = live_return
    else:
        feature_dict["btc_close"] = live_price
        feature_dict["btc_return"] = live_return
        feature_dict["btc_volume"] = live_vol
        feature_dict["btc_volatility"] = abs(live_return)

    # Extract feature values array matching the exact feature schema of the trained model
    expected_cols = metadata.get("feature_schema", FEATURE_COLUMNS)
    feature_vector = np.array([[feature_dict.get(col, 0.0) for col in expected_cols]])

    # Scale if required by model
    if metadata.get("is_scaled", True) and scaler is not None:
        feature_vector_scaled = scaler.transform(feature_vector)
    else:
        feature_vector_scaled = feature_vector

    # 4. Predict probabilities & class
    pred_label = int(model.predict(feature_vector_scaled)[0])
    predicted_direction = LABEL_TO_CLASS.get(pred_label, "NEUTRAL")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector_scaled)[0]
        # Standardize probability mapping order: BEARISH (0), NEUTRAL (1), BULLISH (2)
        if len(probabilities) == 3:
            prob_bearish = float(probabilities[0])
            prob_neutral = float(probabilities[1])
            prob_bullish = float(probabilities[2])
        else:
            prob_bearish, prob_neutral, prob_bullish = 0.33, 0.34, 0.33
    else:
        prob_bearish = 1.0 if predicted_direction == "BEARISH" else 0.0
        prob_neutral = 1.0 if predicted_direction == "NEUTRAL" else 0.0
        prob_bullish = 1.0 if predicted_direction == "BULLISH" else 0.0

    confidence = max(prob_bullish, prob_neutral, prob_bearish)

    # Conviction tier mapping
    if confidence >= 0.45:
        conviction_level = "HIGH (Strong Signal)"
    elif confidence >= 0.38:
        conviction_level = "MODERATE (Clear Trend)"
    else:
        conviction_level = "LOW (Market Noise)"

    pred_id = f"{asset}_{now_dt.strftime('%Y%m%d_%H%M%S')}"

    prediction_record = {
        "prediction_id": pred_id,
        "timestamp": now_str,
        "asset": asset_key,
        "prediction_horizon": "Next-Day (24H Close-to-Close)",
        "current_price": round(live_price, 2),
        "predicted_direction": predicted_direction,
        "prob_bullish": round(prob_bullish, 4),
        "prob_neutral": round(prob_neutral, 4),
        "prob_bearish": round(prob_bearish, 4),
        "confidence": round(confidence, 4),
        "conviction_level": conviction_level,
        "model_version": metadata.get("model_version", "v1.0"),
        "model_name": metadata.get("model_name", "ExtraTrees (Tuned)"),
        "data_status": "LIVE" if is_live else "LAST_AVAILABLE",
        "status": "PENDING",
        "actual_price": np.nan,
        "actual_return": np.nan,
        "actual_direction": np.nan,
        "correct": np.nan,
        "resolution_timestamp": np.nan
    }

    # Only mutate disk files when save_to_history is explicitly enabled
    if save_to_history:
        live_df = pd.DataFrame([prediction_record])
        live_df.to_csv(PRED_DIR / "live_predictions.csv", index=False)

        history_file = PRED_DIR / "prediction_history.csv"
        if history_file.exists():
            existing_hist = pd.read_csv(history_file)

            # Ensure necessary object columns exist
            for c in ["actual_direction", "actual_price", "actual_return", "correct", "resolution_timestamp", "conviction_level"]:
                if c in existing_hist.columns:
                    existing_hist[c] = existing_hist[c].astype(object)

            # Check if an active PENDING prediction exists for this asset
            pending_mask = (existing_hist["asset"] == asset_key) & (existing_hist["status"] == "PENDING")
            if pending_mask.any():
                # Update single active pending row in-place with latest price & prediction
                last_pending_idx = existing_hist[pending_mask].index[-1]
                for key, val in prediction_record.items():
                    existing_hist.at[last_pending_idx, key] = val
                combined_hist = existing_hist
            else:
                # Active prediction was resolved; append new PENDING prediction row for next period
                combined_hist = pd.concat([existing_hist, live_df], ignore_index=True)
        else:
            combined_hist = live_df

        combined_hist.to_csv(history_file, index=False)

    return prediction_record, feature_dict


if __name__ == "__main__":
    generate_live_prediction(save_to_history=True)
