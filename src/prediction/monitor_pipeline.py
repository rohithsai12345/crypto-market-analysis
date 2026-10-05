import os
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
PRED_DIR = BASE_DIR / "data" / "predictions"
MODELS_DIR = BASE_DIR / "models"


def check_market_feed_health(timeout_sec=5):
    """
    Checks HTTP connectivity to CoinGecko and Binance public APIs.
    Returns status dict with boolean availability and latency metrics.
    """
    endpoints = {
        "coingecko": "https://api.coingecko.com/api/v3/ping",
        "binance": "https://api.binance.com/api/v3/ping"
    }
    feed_status = {}

    for name, url in endpoints.items():
        try:
            resp = requests.get(url, timeout=timeout_sec)
            if resp.status_code == 200:
                feed_status[name] = {"status": "HEALTHY", "http_code": 200, "latency_ms": round(resp.elapsed.total_seconds() * 1000, 2)}
            else:
                feed_status[name] = {"status": "DEGRADED", "http_code": resp.status_code, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            feed_status[name] = {"status": "UNREACHABLE", "error": str(e)}

    return feed_status


def check_dataset_freshness(max_age_hours=24):
    """
    Checks if datasets have recent market data within max_age_hours.
    """
    files_to_check = {
        "market_sentiment": DATA_DIR / "market_sentiment_analysis.csv",
        "btc_dataset": DATA_DIR / "prediction_dataset.csv",
        "eth_dataset": DATA_DIR / "prediction_dataset_eth.csv"
    }

    freshness = {}

    for name, filepath in files_to_check.items():
        if not filepath.exists():
            freshness[name] = {"status": "MISSING", "filepath": str(filepath)}
            continue

        try:
            df = pd.read_csv(filepath)
            if "date" not in df.columns or df.empty:
                freshness[name] = {"status": "EMPTY_OR_INVALID"}
                continue

            latest_date_str = str(df["date"].iloc[-1])
            latest_dt = pd.to_datetime(latest_date_str)

            now = datetime.now()
            age_hours = (now - latest_dt).total_seconds() / 3600.0

            # Daily closes allow up to 48h accounting for date boundaries
            is_stale = age_hours > (max_age_hours + 24)

            freshness[name] = {
                "status": "STALE" if is_stale else "FRESH",
                "latest_date": latest_date_str,
                "age_hours": round(age_hours, 1)
            }
        except Exception as e:
            freshness[name] = {"status": "ERROR", "error": str(e)}

    return freshness


def check_model_drift(history_file=None):
    """
    Evaluates rolling prediction accuracy on resolved live predictions to monitor model drift.
    """
    if history_file is None:
        history_file = PRED_DIR / "prediction_history.csv"

    if not Path(history_file).exists():
        return {"status": "NO_PREDICTION_HISTORY", "resolved_count": 0}

    try:
        df = pd.read_csv(history_file)
        resolved_df = df[df["status"] == "RESOLVED"].copy()

        if len(resolved_df) < 5:
            return {
                "status": "INSUFFICIENT_DATA",
                "total_predictions": len(df),
                "resolved_count": len(resolved_df),
                "message": "At least 5 resolved predictions required for rolling drift tracking."
            }

        resolved_df["correct"] = resolved_df["predicted_direction"] == resolved_df["actual_direction"]
        recent_acc = resolved_df["correct"].mean()

        neutral_baseline_acc = (resolved_df["actual_direction"] == "NEUTRAL").mean()

        drift_report = {
            "status": "MONITORING",
            "total_resolved": len(resolved_df),
            "rolling_accuracy": round(float(recent_acc), 4),
            "neutral_baseline": round(float(neutral_baseline_acc), 4),
            "drift_detected": bool(recent_acc < neutral_baseline_acc - 0.05),
            "summary": f"Rolling accuracy: {recent_acc*100:.1f}% vs NEUTRAL baseline: {neutral_baseline_acc*100:.1f}%"
        }
        return drift_report
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def check_pipeline_health():
    """
    Aggregates market feed health, dataset freshness, and model drift into a unified health summary.
    """
    feed_status = check_market_feed_health()
    freshness = check_dataset_freshness()
    drift = check_model_drift()

    all_feeds_ok = all(v.get("status") == "HEALTHY" for v in feed_status.values())
    all_datasets_fresh = all(v.get("status") in ["FRESH", "MISSING"] for v in freshness.values())

    overall_status = "HEALTHY" if (all_feeds_ok and all_datasets_fresh) else "WARNING"

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall_status,
        "market_feeds": feed_status,
        "dataset_freshness": freshness,
        "model_drift": drift
    }

    return summary


if __name__ == "__main__":
    health = check_pipeline_health()
    print(json.dumps(health, indent=2))
