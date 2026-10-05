import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.prediction.feature_builder import classify_return, BULLISH_THRESHOLD, BEARISH_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRED_DIR = BASE_DIR / "data" / "predictions"
DATA_DIR = BASE_DIR / "data" / "processed"


def resolve_pending_predictions(
    history_file=None,
    dataset_file=None,
    bullish_thresh=BULLISH_THRESHOLD,
    bearish_thresh=BEARISH_THRESHOLD
):
    """
    Periodically checks pending predictions in prediction_history.csv,
    retrieves actual future market prices from data/processed/prediction_dataset.csv or market data,
    calculates actual return, converts to actual direction, marks correctness, and saves.
    """
    if history_file is None:
        history_file = PRED_DIR / "prediction_history.csv"
    if dataset_file is None:
        dataset_file = DATA_DIR / "prediction_dataset.csv"

    if not Path(history_file).exists():
        print(f"No prediction history file found at: {history_file}")
        return None

    df_pred = pd.read_csv(history_file)
    if df_pred.empty:
        return df_pred

    # Ensure correct object/flexible dtypes for outcome columns
    df_pred["actual_direction"] = df_pred["actual_direction"].astype(object)
    df_pred["actual_price"] = df_pred["actual_price"].astype(object)
    df_pred["actual_return"] = df_pred["actual_return"].astype(object)
    df_pred["correct"] = df_pred["correct"].astype(object)
    df_pred["resolution_timestamp"] = df_pred["resolution_timestamp"].astype(object)

    if not Path(dataset_file).exists():
        print("Dataset file missing for resolving outcomes.")
        return df_pred

    df_data = pd.read_csv(dataset_file)
    df_data["date"] = pd.to_datetime(df_data["date"])

    resolved_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in df_pred.iterrows():
        if row.get("status") == "PENDING" or pd.isna(row.get("status")):
            pred_time = pd.to_datetime(row["timestamp"])
            curr_price = float(row["current_price"])

            # Find data row matching prediction date or next available market day
            match_data = df_data[df_data["date"] >= pred_time.floor("D")].sort_values("date")

            if len(match_data) >= 2:
                # Actual next day price
                actual_price = float(match_data.iloc[1]["btc_close"])
                actual_return = (actual_price - curr_price) / curr_price if curr_price > 0 else 0.0

                actual_direction = classify_return(actual_return, bullish_thresh, bearish_thresh)
                pred_direction = str(row["predicted_direction"])

                is_correct = bool(actual_direction == pred_direction)

                # Update row
                df_pred.at[idx, "actual_price"] = round(actual_price, 2)
                df_pred.at[idx, "actual_return"] = round(actual_return * 100.0, 2)  # In percentage
                df_pred.at[idx, "actual_direction"] = actual_direction
                df_pred.at[idx, "correct"] = is_correct
                df_pred.at[idx, "status"] = "RESOLVED"
                df_pred.at[idx, "resolution_timestamp"] = now_str

                resolved_count += 1

    if resolved_count > 0:
        df_pred.to_csv(history_file, index=False)
        print(f"Successfully resolved {resolved_count} pending predictions.")
    else:
        print("No pending predictions ready for resolution.")

    return df_pred


if __name__ == "__main__":
    resolve_pending_predictions()
