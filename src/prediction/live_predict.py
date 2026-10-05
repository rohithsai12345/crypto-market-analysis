import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.prediction.feature_builder import (
    FEATURE_COLUMNS_BTC,
    FEATURE_COLUMNS_ETH,
    LABEL_TO_CLASS
)

try:
    from dashboard.services.market_api import get_live_market_data
except ImportError:
    try:
        from services.market_api import get_live_market_data
    except ImportError:
        get_live_market_data = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
PRED_DIR = BASE_DIR / "data" / "predictions"
MODELS_DIR = BASE_DIR / "models"

PRED_DIR.mkdir(parents=True, exist_ok=True)


def generate_live_prediction(asset="BTC"):
    """
    Pure live inference function. Fetches market data, builds feature vector, runs model prediction,
    and returns (prediction_record, feature_snapshot) without writing any files to disk.
    """
    asset_key = asset.upper()

    # Asset specific model selection
    model_path = MODELS_DIR / f"predictive_model_{asset_key.lower()}.joblib"
    scaler_path = MODELS_DIR / f"predictive_scaler_{asset_key.lower()}.joblib"
    metadata_path = MODELS_DIR / f"predictive_model_metadata_{asset_key.lower()}.json"

    # Fallback to default single model if asset model does not exist
    if not model_path.exists():
        model_path = MODELS_DIR / "predictive_model.joblib"
        scaler_path = MODELS_DIR / "predictive_scaler.joblib"
        metadata_path = MODELS_DIR / "predictive_model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Train model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # 1. Load latest processed dataset row
    dataset_file = DATA_DIR / (f"prediction_dataset_{asset_key.lower()}.csv" if (DATA_DIR / f"prediction_dataset_{asset_key.lower()}.csv").exists() else "prediction_dataset.csv")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file missing: {dataset_file}")

    history_df = pd.read_csv(dataset_file)
    latest_hist = history_df.iloc[-1].to_dict()

    # 2. Ingest real-time live market data
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    if asset_key == "ETH":
        coingecko_key = "ethereum"
        hist_prefix = "eth_"
        default_price = 3300.0
    else:
        coingecko_key = "bitcoin"
        hist_prefix = "btc_"
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

    # 3. Construct live feature dictionary
    feature_dict = latest_hist.copy()
    if asset_key == "ETH":
        feature_dict["eth_close"] = live_price
        feature_dict["eth_return"] = live_return
        feature_dict["eth_volume"] = live_vol
        feature_dict["eth_volatility"] = abs(live_return)
        feature_dict["eth_return_lag_1"] = live_return
        if feature_dict.get("btc_close", 0) > 0:
            feature_dict["btc_eth_ratio_return"] = live_return
    else:
        feature_dict["btc_close"] = live_price
        feature_dict["btc_return"] = live_return
        feature_dict["btc_volume"] = live_vol
        feature_dict["btc_volatility"] = abs(live_return)

    default_schema = FEATURE_COLUMNS_ETH if asset_key == "ETH" else FEATURE_COLUMNS_BTC
    expected_cols = metadata.get("feature_schema", default_schema)

    feature_vector = np.array([[feature_dict.get(col, 0.0) for col in expected_cols]])

    if metadata.get("is_scaled", True) and scaler is not None:
        feature_vector_scaled = scaler.transform(feature_vector)
    else:
        feature_vector_scaled = feature_vector

    # 4. Predict probabilities & class
    pred_label = int(model.predict(feature_vector_scaled)[0])
    predicted_direction = LABEL_TO_CLASS.get(pred_label, "NEUTRAL")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector_scaled)[0]
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

    if confidence >= 0.45:
        conviction_level = "HIGH (Strong Signal)"
    elif confidence >= 0.38:
        conviction_level = "MODERATE (Clear Trend)"
    else:
        conviction_level = "LOW (Market Noise)"

    pred_id = f"{asset_key}_{now_dt.strftime('%Y%m%d_%H%M%S')}"

    prediction_record = {
        "prediction_id": pred_id,
        "timestamp": now_str,
        "asset": asset_key,
        "prediction_horizon": f"Predict {asset_key}'s next 24-hour direction",
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

    # Zero side-effect file writes in generate_live_prediction
    return prediction_record, feature_dict


def save_prediction(prediction_record, history_file=None, live_file=None):
    """
    Explicitly logs a prediction record to live_predictions.csv and prediction_history.csv.
    Called ONLY when the user clicks 'Log prediction'.
    """
    if history_file is None:
        history_file = PRED_DIR / "prediction_history.csv"
    if live_file is None:
        live_file = PRED_DIR / "live_predictions.csv"

    live_df = pd.DataFrame([prediction_record])
    live_df.to_csv(live_file, index=False)

    if Path(history_file).exists():
        existing_hist = pd.read_csv(history_file)
        for c in ["actual_direction", "actual_price", "actual_return", "correct", "resolution_timestamp", "conviction_level"]:
            if c in existing_hist.columns:
                existing_hist[c] = existing_hist[c].astype(object)

        asset_key = prediction_record["asset"]
        pending_mask = (existing_hist["asset"] == asset_key) & (existing_hist["status"] == "PENDING")
        if pending_mask.any():
            last_pending_idx = existing_hist[pending_mask].index[-1]
            for key, val in prediction_record.items():
                existing_hist.at[last_pending_idx, key] = val
            combined_hist = existing_hist
        else:
            combined_hist = pd.concat([existing_hist, live_df], ignore_index=True)
    else:
        combined_hist = live_df

    combined_hist.to_csv(history_file, index=False)
    print(f"Logged prediction to: {history_file}")
    return combined_hist


if __name__ == "__main__":
    rec, feat = generate_live_prediction(asset="BTC")
    save_prediction(rec)
