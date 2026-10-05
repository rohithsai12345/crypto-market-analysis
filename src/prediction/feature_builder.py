import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Configurable Target Thresholds
BULLISH_THRESHOLD = 0.01   # +1.0% return
BEARISH_THRESHOLD = -0.01  # -1.0% return

# Enhanced Feature Schema List (BTC)
FEATURE_COLUMNS_BTC = [
    # Base Market Features (Bitcoin & Ethereum)
    "btc_close",
    "btc_return",
    "btc_volume",
    "btc_volatility",
    "eth_close",
    "eth_return",
    "eth_volume",
    "btc_eth_ratio_return",
    # Technical Indicators (BTC)
    "btc_rsi_14",
    "btc_macd",
    "btc_macd_signal",
    "btc_macd_hist",
    "btc_ema_ratio_7_25",
    "btc_ema_ratio_25_90",
    "btc_bb_percent_b",
    "btc_high_low_ratio",
    "btc_parkinson_vol",
    "btc_volume_zscore_20d",
    "btc_stoch_k",
    "btc_stoch_d",
    "btc_sma_ratio_50_200",
    "btc_roc_5d",
    "btc_roc_14d",
    # Sentiment Features
    "avg_sentiment",
    "positive_ratio",
    "neutral_ratio",
    "negative_ratio",
    "pos_neg_sentiment_ratio",
    "news_count",
    "avg_finbert_confidence",
    "sentiment_zscore_14d",
    "sentiment_return_interaction",
    # Sentiment Lags & Momentum
    "sentiment_lag_1",
    "sentiment_lag_2",
    "sentiment_lag_3",
    "sentiment_momentum_3d",
    "rolling_sentiment_3d",
    "rolling_sentiment_7d",
    "rolling_sentiment_14d",
    "sentiment_std_7d",
    # Return Lags & Volatility
    "btc_return_lag_1",
    "btc_return_lag_2",
    "btc_return_lag_3",
    "eth_return_lag_1",
    "news_count_lag_1",
    "rolling_return_3d",
    "rolling_return_7d",
    "rolling_return_14d",
    "rolling_volatility_7d",
    "rolling_volatility_14d",
    "news_volume_change_7d",
    "btc_volume_change_7d"
]

# Alias for backward compatibility
FEATURE_COLUMNS = FEATURE_COLUMNS_BTC

# Feature Schema List (ETH)
FEATURE_COLUMNS_ETH = [
    # Base Market Features
    "eth_close",
    "eth_return",
    "eth_volume",
    "eth_volatility",
    "btc_close",
    "btc_return",
    "btc_volume",
    "btc_eth_ratio_return",
    # Technical Indicators (ETH)
    "eth_rsi_14",
    "eth_macd",
    "eth_macd_signal",
    "eth_macd_hist",
    "eth_ema_ratio_7_25",
    "eth_ema_ratio_25_90",
    "eth_bb_percent_b",
    "eth_high_low_ratio",
    "eth_parkinson_vol",
    "eth_volume_zscore_20d",
    "eth_stoch_k",
    "eth_stoch_d",
    "eth_sma_ratio_50_200",
    "eth_roc_5d",
    "eth_roc_14d",
    # Sentiment Features
    "avg_sentiment",
    "positive_ratio",
    "neutral_ratio",
    "negative_ratio",
    "pos_neg_sentiment_ratio",
    "news_count",
    "avg_finbert_confidence",
    "sentiment_zscore_14d",
    "sentiment_return_interaction",
    # Sentiment Lags & Momentum
    "sentiment_lag_1",
    "sentiment_lag_2",
    "sentiment_lag_3",
    "sentiment_momentum_3d",
    "rolling_sentiment_3d",
    "rolling_sentiment_7d",
    "rolling_sentiment_14d",
    "sentiment_std_7d",
    # Return Lags & Volatility
    "eth_return_lag_1",
    "eth_return_lag_2",
    "eth_return_lag_3",
    "btc_return_lag_1",
    "news_count_lag_1",
    "rolling_eth_return_3d",
    "rolling_eth_return_7d",
    "rolling_eth_return_14d",
    "rolling_eth_volatility_7d",
    "rolling_eth_volatility_14d",
    "news_volume_change_7d",
    "eth_volume_change_7d"
]

TARGET_CLASSES = ["BEARISH", "NEUTRAL", "BULLISH"]
CLASS_TO_LABEL = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
LABEL_TO_CLASS = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}


def classify_return(ret_val, bullish_thresh=BULLISH_THRESHOLD, bearish_thresh=BEARISH_THRESHOLD):
    """
    Maps return value to target direction class strictly using configured thresholds.
    """
    if pd.isna(ret_val):
        return np.nan
    if ret_val > bullish_thresh:
        return "BULLISH"
    elif ret_val < bearish_thresh:
        return "BEARISH"
    else:
        return "NEUTRAL"


def compute_technical_indicators(df):
    """
    Computes technical indicators (RSI, MACD, EMAs, Bollinger Bands, Parkinson Volatility,
    Stochastic Oscillator, Volume Z-scores, ROC) strictly backward-looking up to t for both BTC and ETH.
    """
    for asset in ["btc", "eth"]:
        close_col = f"{asset}_close"
        volume_col = f"{asset}_volume"
        return_col = f"{asset}_return"

        if close_col not in df.columns:
            continue

        close = df[close_col]
        
        # 1. RSI-14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f"{asset}_rsi_14"] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9) normalized by Close
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df[f"{asset}_macd"] = (ema12 - ema26) / (close + 1e-9)
        df[f"{asset}_macd_signal"] = df[f"{asset}_macd"].ewm(span=9, adjust=False).mean()
        df[f"{asset}_macd_hist"] = df[f"{asset}_macd"] - df[f"{asset}_macd_signal"]

        # 3. EMA Ratios & SMA 50/200 Trend Ratio
        ema7 = close.ewm(span=7, adjust=False).mean()
        ema25 = close.ewm(span=25, adjust=False).mean()
        ema90 = close.ewm(span=90, adjust=False).mean()
        sma50 = close.rolling(50, min_periods=1).mean()
        sma200 = close.rolling(200, min_periods=1).mean()

        df[f"{asset}_ema_ratio_7_25"] = (ema7 / (ema25 + 1e-9)) - 1.0
        df[f"{asset}_ema_ratio_25_90"] = (ema25 / (ema90 + 1e-9)) - 1.0
        df[f"{asset}_sma_ratio_50_200"] = (sma50 / (sma200 + 1e-9)) - 1.0

        # 4. Bollinger Bands %B (20-day)
        ma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std().fillna(0)
        upper_b = ma20 + (2 * std20)
        lower_b = ma20 - (2 * std20)
        df[f"{asset}_bb_percent_b"] = (close - lower_b) / (upper_b - lower_b + 1e-9)

        # 5. High-Low Price Range Ratio & Parkinson Volatility Proxy
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        if high_col in df.columns and low_col in df.columns:
            df[f"{asset}_high_low_ratio"] = (df[high_col] - df[low_col]) / (close + 1e-9)
            high_low_ratio = (df[high_col] / (df[low_col] + 1e-9)).clip(lower=1.0)
            df[f"{asset}_parkinson_vol"] = np.sqrt(((np.log(high_low_ratio)) ** 2) / (4 * np.log(2)))
            
            # Stochastic Oscillator %K & %D (14-day)
            low14 = df[low_col].rolling(14, min_periods=1).min()
            high14 = df[high_col].rolling(14, min_periods=1).max()
            df[f"{asset}_stoch_k"] = (close - low14) / (high14 - low14 + 1e-9) * 100.0
            df[f"{asset}_stoch_d"] = df[f"{asset}_stoch_k"].rolling(3, min_periods=1).mean()
        else:
            df[f"{asset}_high_low_ratio"] = df[return_col].abs()
            df[f"{asset}_parkinson_vol"] = df[return_col].abs()
            df[f"{asset}_stoch_k"] = 50.0
            df[f"{asset}_stoch_d"] = 50.0

        # 6. Volume Z-Score (20-day)
        vol_ma20 = df[volume_col].rolling(20, min_periods=1).mean()
        vol_std20 = df[volume_col].rolling(20, min_periods=1).std().fillna(1.0)
        df[f"{asset}_volume_zscore_20d"] = (df[volume_col] - vol_ma20) / (vol_std20 + 1e-5)

        # 7. Rate of Change (ROC-5 & ROC-14)
        df[f"{asset}_roc_5d"] = close.pct_change(5).fillna(0.0)
        df[f"{asset}_roc_14d"] = close.pct_change(14).fillna(0.0)

    # Multi-asset ratio feature (BTC vs ETH)
    btc_eth_ratio = df["btc_close"] / (df["eth_close"] + 1e-9)
    df["btc_eth_ratio_return"] = btc_eth_ratio.pct_change().fillna(0.0)

    # Sentiment Z-Score & Pos-Neg Ratio
    sent_ma14 = df["avg_sentiment"].rolling(14, min_periods=1).mean()
    sent_std14 = df["avg_sentiment"].rolling(14, min_periods=1).std().fillna(0.1)
    df["sentiment_zscore_14d"] = (df["avg_sentiment"] - sent_ma14) / (sent_std14 + 1e-5)
    df["pos_neg_sentiment_ratio"] = df["positive_ratio"] / (df["negative_ratio"] + 1e-5)

    return df


def build_predictive_dataset(
    asset="BTC",
    market_sentiment_file=None,
    output_file=None,
    bullish_thresh=BULLISH_THRESHOLD,
    bearish_thresh=BEARISH_THRESHOLD
):
    """
    Constructs a unified chronological predictive dataset with zero temporal leakage for asset (BTC/ETH).
    Every row represents observation at time t. Target represents actual market return at t+1.
    """
    asset_key = asset.upper()
    if market_sentiment_file is None:
        market_sentiment_file = DATA_DIR / "market_sentiment_analysis.csv"
    if output_file is None:
        if asset_key == "ETH":
            output_file = DATA_DIR / "prediction_dataset_eth.csv"
        else:
            output_file = DATA_DIR / "prediction_dataset.csv"

    if not Path(market_sentiment_file).exists():
        raise FileNotFoundError(f"Input file not found: {market_sentiment_file}")

    df = pd.read_csv(market_sentiment_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Base Market & NLP Features
    df["btc_volatility"] = df["btc_return"].abs()
    df["eth_volatility"] = df["eth_return"].abs()

    if "avg_finbert_confidence" not in df.columns:
        df["avg_finbert_confidence"] = 0.80

    # Technical indicators for both BTC & ETH
    df = compute_technical_indicators(df)

    # Lag Features (Chronological backward shift only)
    df["sentiment_lag_1"] = df["avg_sentiment"].shift(1)
    df["sentiment_lag_2"] = df["avg_sentiment"].shift(2)
    df["sentiment_lag_3"] = df["avg_sentiment"].shift(3)
    df["sentiment_momentum_3d"] = df["avg_sentiment"] - df["sentiment_lag_3"]

    df["btc_return_lag_1"] = df["btc_return"].shift(1)
    df["btc_return_lag_2"] = df["btc_return"].shift(2)
    df["btc_return_lag_3"] = df["btc_return"].shift(3)

    df["eth_return_lag_1"] = df["eth_return"].shift(1)
    df["eth_return_lag_2"] = df["eth_return"].shift(2)
    df["eth_return_lag_3"] = df["eth_return"].shift(3)

    df["news_count_lag_1"] = df["news_count"].shift(1)

    df["sentiment_return_interaction"] = df["avg_sentiment"] * df[f"{asset_key.lower()}_return_lag_1"].fillna(0.0)

    # Rolling Features
    df["rolling_sentiment_3d"] = df["avg_sentiment"].rolling(window=3, min_periods=1).mean()
    df["rolling_sentiment_7d"] = df["avg_sentiment"].rolling(window=7, min_periods=1).mean()
    df["rolling_sentiment_14d"] = df["avg_sentiment"].rolling(window=14, min_periods=1).mean()
    df["sentiment_std_7d"] = df["avg_sentiment"].rolling(window=7, min_periods=1).std().fillna(0)

    # Asset specific rolling features
    df["rolling_return_3d"] = (1 + df["btc_return"]).rolling(window=3, min_periods=1).apply(np.prod, raw=True) - 1
    df["rolling_return_7d"] = (1 + df["btc_return"]).rolling(window=7, min_periods=1).apply(np.prod, raw=True) - 1
    df["rolling_return_14d"] = (1 + df["btc_return"]).rolling(window=14, min_periods=1).apply(np.prod, raw=True) - 1

    df["rolling_eth_return_3d"] = (1 + df["eth_return"]).rolling(window=3, min_periods=1).apply(np.prod, raw=True) - 1
    df["rolling_eth_return_7d"] = (1 + df["eth_return"]).rolling(window=7, min_periods=1).apply(np.prod, raw=True) - 1
    df["rolling_eth_return_14d"] = (1 + df["eth_return"]).rolling(window=14, min_periods=1).apply(np.prod, raw=True) - 1

    df["rolling_volatility_7d"] = df["btc_return"].rolling(window=7, min_periods=1).std().fillna(0)
    df["rolling_volatility_14d"] = df["btc_return"].rolling(window=7, min_periods=1).std().fillna(0)

    df["rolling_eth_volatility_7d"] = df["eth_return"].rolling(window=7, min_periods=1).std().fillna(0)
    df["rolling_eth_volatility_14d"] = df["eth_return"].rolling(window=7, min_periods=1).std().fillna(0)

    df["news_volume_change_7d"] = df["news_count"].pct_change(7).fillna(0).replace([np.inf, -np.inf], 0)
    df["btc_volume_change_7d"] = df["btc_volume"].pct_change(7).fillna(0).replace([np.inf, -np.inf], 0)
    df["eth_volume_change_7d"] = df["eth_volume"].pct_change(7).fillna(0).replace([np.inf, -np.inf], 0)

    # Target Generation for Asset (Next-Day t+1 Return)
    target_asset_return = df[f"{asset_key.lower()}_return"].shift(-1)
    df["target_return"] = target_asset_return
    df["target_direction"] = df["target_return"].apply(
        lambda r: classify_return(r, bullish_thresh, bearish_thresh)
    )
    df["target_label"] = df["target_direction"].map(CLASS_TO_LABEL)

    feature_schema = FEATURE_COLUMNS_ETH if asset_key == "ETH" else FEATURE_COLUMNS_BTC
    feature_cols = ["date", "btc_close", "eth_close", "target_return", "target_direction", "target_label"] + feature_schema
    feature_df = df[feature_cols].copy()

    feature_clean_df = feature_df.dropna(subset=feature_schema + ["target_direction"]).copy()

    feature_clean_df.to_csv(output_file, index=False)

    print(f"[{asset_key}] Predictive dataset saved to: {output_file}")
    print(f"Total rows: {len(feature_clean_df)}, Features: {len(feature_schema)}")
    print(f"Date range: {feature_clean_df['date'].min().date()} → {feature_clean_df['date'].max().date()}")

    return feature_clean_df


if __name__ == "__main__":
    build_predictive_dataset(asset="BTC")
    build_predictive_dataset(asset="ETH")
