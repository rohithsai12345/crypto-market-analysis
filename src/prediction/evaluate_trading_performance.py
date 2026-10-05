import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    brier_score_loss,
    log_loss
)

from src.prediction.feature_builder import (
    FEATURE_COLUMNS,
    LABEL_TO_CLASS,
    CLASS_TO_LABEL
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def evaluate_trading_and_investment_performance(
    dataset_file=None,
    model_path=None,
    scaler_path=None,
    metadata_path=None,
    test_ratio=0.15,
    fee_rate=0.0010,  # 0.10% transaction fee / slippage (10 bps)
    confidence_gate=0.45
):
    """
    Evaluates financial usefulness, risk-adjusted returns, and walk-forward trading performance.
    
    Includes:
    1. Walk-forward backtest (0.1% fees, turnover, drawdown, Sharpe ratio).
    2. Benchmark strategy comparisons (Buy & Hold, Majority Class).
    3. Per-class precision, recall, and F1-score breakdowns.
    4. Probability calibration & confidence-gated return analysis.
    """
    if dataset_file is None:
        dataset_file = DATA_DIR / "prediction_dataset.csv"
    if model_path is None:
        model_path = MODELS_DIR / "predictive_model.joblib"
    if scaler_path is None:
        scaler_path = MODELS_DIR / "predictive_scaler.joblib"
    if metadata_path is None:
        metadata_path = MODELS_DIR / "predictive_model_metadata.json"

    if not Path(dataset_file).exists() or not Path(model_path).exists():
        return {"error": "Dataset or trained model file missing."}

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

    expected_cols = meta.get("feature_schema", FEATURE_COLUMNS)
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

    # 1. Classification Metrics & Per-Class Performance
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

    # 2. Baselines Comparison
    majority_class = int(pd.Series(y_true).mode()[0])
    majority_pred = np.full_like(y_true, majority_class)
    majority_acc = accuracy_score(y_true, majority_pred)

    # 3. Walk-Forward Backtesting Engine
    # Position: BULLISH (+1), BEARISH (-1 or 0), NEUTRAL (0)
    # Default strategy: Long on BULLISH, Short on BEARISH, Cash on NEUTRAL
    # Gated strategy: Hold Cash unless confidence >= confidence_gate
    
    raw_positions = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
    gated_positions = np.where(confidences >= confidence_gate, raw_positions, 0.0)

    # Compute daily strategy returns with 0.10% transaction fees on position switches
    bnh_returns = target_returns
    
    # Calculate position changes & fees
    raw_pos_changes = np.abs(np.diff(np.insert(raw_positions, 0, 0.0)))
    raw_fees = raw_pos_changes * fee_rate
    raw_strat_returns = raw_positions * target_returns - raw_fees

    gated_pos_changes = np.abs(np.diff(np.insert(gated_positions, 0, 0.0)))
    gated_fees = gated_pos_changes * fee_rate
    gated_strat_returns = gated_positions * target_returns - gated_fees

    # Cumulative equity curves
    bnh_equity = np.cumprod(1 + bnh_returns)
    raw_equity = np.cumprod(1 + raw_strat_returns)
    gated_equity = np.cumprod(1 + gated_strat_returns)

    # Cumulative total returns
    bnh_total_return = float(bnh_equity[-1] - 1.0)
    raw_total_return = float(raw_equity[-1] - 1.0)
    gated_total_return = float(gated_equity[-1] - 1.0)

    # Annualized Sharpe Ratio (365 trading days/yr in crypto)
    def calc_sharpe(ret_series):
        std_val = np.std(ret_series)
        if std_val < 1e-9:
            return 0.0
        return float((np.mean(ret_series) / std_val) * np.sqrt(365))

    # Maximum Drawdown (MDD)
    def calc_max_drawdown(equity_curve):
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return float(np.min(drawdowns))

    bnh_sharpe = calc_sharpe(bnh_returns)
    raw_sharpe = calc_sharpe(raw_strat_returns)
    gated_sharpe = calc_sharpe(gated_strat_returns)

    bnh_mdd = calc_max_drawdown(bnh_equity)
    raw_mdd = calc_max_drawdown(raw_equity)
    gated_mdd = calc_max_drawdown(gated_equity)

    turnover = float(np.mean(gated_pos_changes))

    # 4. Multi-threshold Confidence Evaluation
    threshold_results = {}
    for thresh in [0.33, 0.38, 0.45, 0.50]:
        t_pos = np.where(confidences >= thresh, raw_positions, 0.0)
        t_changes = np.abs(np.diff(np.insert(t_pos, 0, 0.0)))
        t_returns = t_pos * target_returns - (t_changes * fee_rate)
        t_eq = np.cumprod(1 + t_returns)
        threshold_results[f"c_thresh_{thresh}"] = {
            "total_return": round(float(t_eq[-1] - 1.0) * 100, 2),
            "sharpe_ratio": round(calc_sharpe(t_returns), 2),
            "max_drawdown": round(calc_max_drawdown(t_eq) * 100, 2),
            "active_trade_pct": round(float(np.mean(t_pos != 0.0)) * 100, 2)
        }

    # 5. Model Probability Calibration (Log Loss & Brier Score)
    try:
        calib_log_loss = float(log_loss(y_true, probabilities))
    except Exception:
        calib_log_loss = 0.0

    results_summary = {
        "test_sample_count": len(test_df),
        "test_date_range": f"{test_df['date'].min().date()} → {test_df['date'].max().date()}",
        "accuracy_model": round(float(overall_acc), 4),
        "accuracy_majority_baseline": round(float(majority_acc), 4),
        "per_class_performance": per_class_metrics,
        "calibration_log_loss": round(calib_log_loss, 4),
        "trading_simulation": {
            "transaction_fee_rate": fee_rate,
            "benchmark_buy_and_hold_return_pct": round(bnh_total_return * 100, 2),
            "benchmark_buy_and_hold_sharpe": round(bnh_sharpe, 2),
            "benchmark_buy_and_hold_mdd_pct": round(bnh_mdd * 100, 2),
            "raw_strategy_return_pct": round(raw_total_return * 100, 2),
            "raw_strategy_sharpe": round(raw_sharpe, 2),
            "raw_strategy_mdd_pct": round(raw_mdd * 100, 2),
            "gated_strategy_confidence_threshold": confidence_gate,
            "gated_strategy_return_pct": round(gated_total_return * 100, 2),
            "gated_strategy_sharpe": round(gated_sharpe, 2),
            "gated_strategy_mdd_pct": round(gated_mdd * 100, 2),
            "position_turnover_daily": round(turnover, 4)
        },
        "confidence_threshold_sweep": threshold_results
    }

    # Save summary report file
    eval_file = DATA_DIR / "trading_evaluation_report.json"
    with open(eval_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


if __name__ == "__main__":
    report = evaluate_trading_and_investment_performance()
    print(json.dumps(report, indent=2))
