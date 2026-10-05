import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    log_loss
)

from src.prediction.feature_builder import (
    FEATURE_COLUMNS_BTC,
    FEATURE_COLUMNS_ETH,
    LABEL_TO_CLASS,
    CLASS_TO_LABEL,
    classify_return
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def evaluate_trading_and_investment_performance(
    asset="BTC",
    dataset_file=None,
    model_path=None,
    scaler_path=None,
    metadata_path=None,
    test_ratio=0.15,
    fee_rate=0.0010,  # 0.10% transaction fee + slippage (10 bps)
    confidence_gate=0.45
):
    """
    Evaluates financial usefulness and compares model performance against 3 explicit baselines:
    1. Always predict NEUTRAL baseline
    2. Previous day's direction baseline (Naive Persistence)
    3. Simple momentum rule baseline (5-Day ROC)
    Also computes walk-forward backtest equity curves after fees & slippage.
    """
    asset_key = asset.upper()

    if dataset_file is None:
        dataset_file = DATA_DIR / (f"prediction_dataset_{asset_key.lower()}.csv" if asset_key == "ETH" else "prediction_dataset.csv")
    if model_path is None:
        model_path = MODELS_DIR / (f"predictive_model_{asset_key.lower()}.joblib" if asset_key == "ETH" else "predictive_model.joblib")
    if scaler_path is None:
        scaler_path = MODELS_DIR / (f"predictive_scaler_{asset_key.lower()}.joblib" if asset_key == "ETH" else "predictive_scaler.joblib")
    if metadata_path is None:
        metadata_path = MODELS_DIR / (f"predictive_model_metadata_{asset_key.lower()}.json" if asset_key == "ETH" else "predictive_model_metadata.json")

    if not Path(dataset_file).exists() or not Path(model_path).exists():
        return {"error": f"Dataset or model file missing for {asset_key}."}

    df = pd.read_csv(dataset_file)
    df["date"] = pd.to_datetime(df["date"])
    
    labeled_df = df.dropna(subset=["target_label", "target_return"]).copy()
    labeled_df["target_label"] = labeled_df["target_label"].astype(int)

    n_samples = len(labeled_df)
    test_start_idx = int(n_samples * (1 - test_ratio))
    test_df = labeled_df.iloc[test_start_idx:].copy().reset_index(drop=True)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    default_schema = FEATURE_COLUMNS_ETH if asset_key == "ETH" else FEATURE_COLUMNS_BTC
    expected_cols = meta.get("feature_schema", default_schema)

    X_test = test_df[expected_cols].values
    y_true = test_df["target_label"].values
    target_returns = test_df["target_return"].values

    if meta.get("is_scaled", True) and scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_pred = model.predict(X_test_scaled)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_scaled)
    else:
        probabilities = np.zeros((len(y_pred), 3))
        for i, p in enumerate(y_pred):
            probabilities[i, p] = 1.0

    confidences = np.max(probabilities, axis=1)

    # 1. Model Accuracy & Per-Class Metrics
    overall_acc = accuracy_score(y_true, y_pred)
    prec_array, rec_array, f1_array, support_array = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    per_class_metrics = {}
    for idx, name in enumerate(["BEARISH", "NEUTRAL", "BULLISH"]):
        per_class_metrics[name] = {
            "precision": round(float(prec_array[idx]), 4),
            "recall": round(float(rec_array[idx]), 4),
            "f1_score": round(float(f1_array[idx]), 4),
            "support": int(support_array[idx])
        }

    # 2. Benchmark Baselines Calculation
    # Baseline 1: Always predict NEUTRAL (Label 1)
    neutral_pred = np.ones_like(y_true)
    neutral_acc = accuracy_score(y_true, neutral_pred)
    neutral_returns = np.zeros_like(target_returns)  # Cash position

    # Baseline 2: Previous day's direction (Naive Persistence)
    asset_ret_col = "eth_return" if asset_key == "ETH" else "btc_return"
    prev_returns = test_df[asset_ret_col].values
    prev_dirs = [classify_return(r) for r in prev_returns]
    prev_pred = np.array([CLASS_TO_LABEL.get(d, 1) for d in prev_dirs])
    prev_acc = accuracy_score(y_true, prev_pred)
    prev_pos = np.where(prev_pred == 2, 1.0, np.where(prev_pred == 0, -1.0, 0.0))
    prev_pos_changes = np.abs(np.diff(np.insert(prev_pos, 0, 0.0)))
    prev_strat_returns = prev_pos * target_returns - (prev_pos_changes * fee_rate)

    # Baseline 3: Simple Momentum Rule (5-Day ROC Direction)
    roc_col = "eth_roc_5d" if asset_key == "ETH" else "btc_roc_5d"
    roc_vals = test_df[roc_col].values if roc_col in test_df.columns else prev_returns
    mom_dirs = [classify_return(r) for r in roc_vals]
    mom_pred = np.array([CLASS_TO_LABEL.get(d, 1) for d in mom_dirs])
    mom_acc = accuracy_score(y_true, mom_pred)
    mom_pos = np.where(mom_pred == 2, 1.0, np.where(mom_pred == 0, -1.0, 0.0))
    mom_pos_changes = np.abs(np.diff(np.insert(mom_pos, 0, 0.0)))
    mom_strat_returns = mom_pos * target_returns - (mom_pos_changes * fee_rate)

    # 3. Model Walk-Forward Backtest
    raw_positions = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
    gated_positions = np.where(confidences >= confidence_gate, raw_positions, 0.0)

    bnh_returns = target_returns
    raw_pos_changes = np.abs(np.diff(np.insert(raw_positions, 0, 0.0)))
    raw_strat_returns = raw_positions * target_returns - (raw_pos_changes * fee_rate)

    gated_pos_changes = np.abs(np.diff(np.insert(gated_positions, 0, 0.0)))
    gated_strat_returns = gated_positions * target_returns - (gated_pos_changes * fee_rate)

    # Cumulative Returns & Risk Metrics
    bnh_equity = np.cumprod(1 + bnh_returns)
    raw_equity = np.cumprod(1 + raw_strat_returns)
    gated_equity = np.cumprod(1 + gated_strat_returns)
    prev_equity = np.cumprod(1 + prev_strat_returns)
    mom_equity = np.cumprod(1 + mom_strat_returns)

    def calc_sharpe(ret_series):
        std_val = np.std(ret_series)
        if std_val < 1e-9:
            return 0.0
        return float((np.mean(ret_series) / std_val) * np.sqrt(365))

    def calc_max_drawdown(equity_curve):
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / (running_max + 1e-9)
        return float(np.min(drawdowns))

    results_summary = {
        "asset": asset_key,
        "test_sample_count": len(test_df),
        "test_date_range": f"{test_df['date'].min().date()} → {test_df['date'].max().date()}",
        "accuracy_model": round(float(overall_acc), 4),
        "baselines_accuracy": {
            "always_neutral": round(float(neutral_acc), 4),
            "previous_day_persistence": round(float(prev_acc), 4),
            "simple_momentum_5d_roc": round(float(mom_acc), 4)
        },
        "per_class_performance": per_class_metrics,
        "trading_simulation_after_fees": {
            "transaction_fee_rate": fee_rate,
            "buy_and_hold_return_pct": round(float(bnh_equity[-1] - 1.0) * 100, 2),
            "buy_and_hold_sharpe": round(calc_sharpe(bnh_returns), 2),
            "buy_and_hold_mdd_pct": round(calc_max_drawdown(bnh_equity) * 100, 2),
            "naive_persistence_return_pct": round(float(prev_equity[-1] - 1.0) * 100, 2),
            "naive_persistence_sharpe": round(calc_sharpe(prev_strat_returns), 2),
            "simple_momentum_return_pct": round(float(mom_equity[-1] - 1.0) * 100, 2),
            "simple_momentum_sharpe": round(calc_sharpe(mom_strat_returns), 2),
            "raw_model_return_pct": round(float(raw_equity[-1] - 1.0) * 100, 2),
            "raw_model_sharpe": round(calc_sharpe(raw_strat_returns), 2),
            "raw_model_mdd_pct": round(calc_max_drawdown(raw_equity) * 100, 2),
            "gated_strategy_return_pct": round(float(gated_equity[-1] - 1.0) * 100, 2),
            "gated_strategy_sharpe": round(calc_sharpe(gated_strat_returns), 2),
            "gated_strategy_mdd_pct": round(calc_max_drawdown(gated_equity) * 100, 2)
        }
    }

    eval_file = DATA_DIR / f"trading_evaluation_report_{asset_key.lower()}.json"
    with open(eval_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Also save to main trading_evaluation_report.json for default BTC
    if asset_key == "BTC":
        with open(DATA_DIR / "trading_evaluation_report.json", "w") as f:
            json.dump(results_summary, f, indent=2)

    return results_summary


if __name__ == "__main__":
    rep_btc = evaluate_trading_and_investment_performance(asset="BTC")
    print("BTC Performance:")
    print(json.dumps(rep_btc, indent=2))

    rep_eth = evaluate_trading_and_investment_performance(asset="ETH")
    print("ETH Performance:")
    print(json.dumps(rep_eth, indent=2))
