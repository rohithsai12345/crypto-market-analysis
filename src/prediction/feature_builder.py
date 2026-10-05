import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Configurable Target Thresholds
BULLISH_THRESHOLD = 0.01   # +1.0% return
BEARISH_THRESHOLD = -0.01  # -1.0% return

# Enhanced Feature Schema List (Used for training and live inference)
FEATURE_COLUMNS = [
    # Base Market Features (Bitcoin & Ethereum only)
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
    Stochastic Oscillator, Volume Z-scores) strictly backward-looking up to t.
    """
    close = df["btc_close"]
    
    # 1. RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["btc_rsi_14"] = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9) normalized by Close
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["btc_macd"] = (ema12 - ema26) / (close + 1e-9)
    df["btc_macd_signal"] = df["btc_macd"].ewm(span=9, adjust=False).mean()
    df["btc_macd_hist"] = df["btc_macd"] - df["btc_macd_signal"]

    # 3. EMA Ratios & SMA 50/200 Trend Ratio
    ema7 = close.ewm(span=7, adjust=False).mean()
    ema25 = close.ewm(span=25, adjust=False).mean()
    ema90 = close.ewm(span=90, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()

    df["btc_ema_ratio_7_25"] = (ema7 / (ema25 + 1e-9)) - 1.0
    df["btc_ema_ratio_25_90"] = (ema25 / (ema90 + 1e-9)) - 1.0
    df["btc_sma_ratio_50_200"] = (sma50 / (sma200 + 1e-9)) - 1.0

    # 4. Bollinger Bands %B (20-day)
    ma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0)
    upper_b = ma20 + (2 * std20)
    lower_b = ma20 - (2 * std20)
    df["btc_bb_percent_b"] = (close - lower_b) / (upper_b - lower_b + 1e-9)

    # 5. High-Low Price Range Ratio & Parkinson Volatility Proxy
    if "btc_high" in df.columns and "btc_low" in df.columns:
        df["btc_high_low_ratio"] = (df["btc_high"] - df["btc_low"]) / (close + 1e-9)
        high_low_ratio = (df["btc_high"] / (df["btc_low"] + 1e-9)).clip(lower=1.0)
        df["btc_parkinson_vol"] = np.sqrt(((np.log(high_low_ratio)) ** 2) / (4 * np.log(2)))
        
        # 6. Stochastic Oscillator %K & %D (14-day)
        low14 = df["btc_low"].rolling(14, min_periods=1).min()
        high14 = df["btc_high"].rolling(14, min_periods=1).max()
        df["btc_stoch_k"] = (close - low14) / (high14 - low14 + 1e-9) * 100.0
        df["btc_stoch_d"] = df["btc_stoch_k"].rolling(3, min_periods=1).mean()
    else:
        df["btc_high_low_ratio"] = df["btc_return"].abs()
        df["btc_parkinson_vol"] = df["btc_return"].abs()
        df["btc_stoch_k"] = 50.0
        df["btc_stoch_d"] = 50.0

    # 7. Volume Z-Score (20-day)
    vol_ma20 = df["btc_volume"].rolling(20, min_periods=1).mean()
    vol_std20 = df["btc_volume"].rolling(20, min_periods=1).std().fillna(1.0)
    df["btc_volume_zscore_20d"] = (df["btc_volume"] - vol_ma20) / (vol_std20 + 1e-5)

    # 8. Rate of Change (ROC-5 & ROC-14)
    df["btc_roc_5d"] = close.pct_change(5).fillna(0.0)
    df["btc_roc_14d"] = close.pct_change(14).fillna(0.0)

    # 9. Multi-asset ratio feature (BTC vs ETH)
    btc_eth_ratio = df["btc_close"] / (df["eth_close"] + 1e-9)
    df["btc_eth_ratio_return"] = btc_eth_ratio.pct_change().fillna(0.0)

    # 10. Sentiment Z-Score & Pos-Neg Ratio
    sent_ma14 = df["avg_sentiment"].rolling(14, min_periods=1).mean()
    sent_std14 = df["avg_sentiment"].rolling(14, min_periods=1).std().fillna(0.1)
    df["sentiment_zscore_14d"] = (df["avg_sentiment"] - sent_ma14) / (sent_std14 + 1e-5)
    df["pos_neg_sentiment_ratio"] = df["positive_ratio"] / (df["negative_ratio"] + 1e-5)

    return df


def build_predictive_dataset(
    market_sentiment_file=None,
    output_file=None,
    bullish_thresh=BULLISH_THRESHOLD,
    bearish_thresh=BEARISH_THRESHOLD
):
    """
    Constructs a unified chronological predictive dataset with zero temporal leakage.
    Every row represents observation at time t. Target represents actual market return at t+1.
    """
    if market_sentiment_file is None:
        market_sentiment_file = DATA_DIR / "market_sentiment_analysis.csv"
    if output_file is None:
        output_file = DATA_DIR / "prediction_dataset.csv"

    if not Path(market_sentiment_file).exists():
        raise FileNotFoundError(f"Input file not found: {market_sentiment_file}")

    df = pd.read_csv(market_sentiment_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Base Market & NLP Features
    df["btc_volatility"] = df["btc_return"].abs()

    if "avg_finbert_confidence" not in df.columns:
        df["avg_finbert_confidence"] = 0.80

    # Technical indicators
    df = compute_technical_indicators(df)

    # 2. Lag Features (Chronological backward shift only)
    df["sentiment_lag_1"] = df["avg_sentiment"].shift(1)
    df["sentiment_lag_2"] = df["avg_sentiment"].shift(2)
    df["sentiment_lag_3"] = df["avg_sentiment"].shift(3)
    df["sentiment_momentum_3d"] = df["avg_sentiment"] - df["sentiment_lag_3"]

    df["btc_return_lag_1"] = df["btc_return"].shift(1)
    df["btc_return_lag_2"] = df["btc_return"].shift(2)
    df["btc_return_lag_3"] = df["btc_return"].shift(3)

    df["eth_return_lag_1"] = df["eth_return"].shift(1)
    df["news_count_lag_1"] = df["news_count"].shift(1)

    df["sentiment_return_interaction"] = df["avg_sentiment"] * df["btc_return_lag_1"].fillna(0.0)

    # 3. Rolling Features (Computed strictly backward up to t)
    df["rolling_sentiment_3d"] = df["avg_sentiment"].rolling(window=3, min_periods=1).mean()
    df["rolling_sentiment_7d"] = df["avg_sentiment"].rolling(window=7, min_periods=1).mean()
    df["rolling_sentiment_14d"] = df["avg_sentiment"].rolling(window=14, min_periods=1).mean()
    df["sentiment_std_7d"] = df["avg_sentiment"].rolling(window=7, min_periods=1).std().fillna(0)

    # Rolling Return (Cumulative return over 3d, 7d, 14d)
    df["rolling_return_3d"] = (1 + df["btc_return"]).rolling(window=3, min_periods=1).apply(np.prod, raw=True) - 1
    df["rolling_return_7d"] = (1 + df["btc_return"]).rolling(window=7, min_periods=1).apply(np.prod, raw=True) - 1
    df["rolling_return_14d"] = (1 + df["btc_return"]).rolling(window=14, min_periods=1).apply(np.prod, raw=True) - 1

    df["rolling_volatility_7d"] = df["btc_return"].rolling(window=7, min_periods=1).std().fillna(0)
    df["rolling_volatility_14d"] = df["btc_return"].rolling(window=14, min_periods=1).std().fillna(0)

    # Volume change percentages over 7d
    df["news_volume_change_7d"] = df["news_count"].pct_change(7).fillna(0).replace([np.inf, -np.inf], 0)
    df["btc_volume_change_7d"] = df["btc_volume"].pct_change(7).fillna(0).replace([np.inf, -np.inf], 0)

    # 4. Target Generation (Next-Day t+1 Return)
    # Target return is the market return on date t+1 relative to observation at date t
    df["target_return"] = df["btc_return"].shift(-1)
    df["target_direction"] = df["target_return"].apply(
        lambda r: classify_return(r, bullish_thresh, bearish_thresh)
    )
    df["target_label"] = df["target_direction"].map(CLASS_TO_LABEL)

    # Feature matrix at time t uses features up to t to predict target return at t+1
    feature_cols = ["date", "btc_close", "eth_close", "target_return", "target_direction", "target_label"] + FEATURE_COLUMNS
    feature_df = df[feature_cols].copy()

    # Drop initial rows missing lag/rolling features, and last row missing t+1 label
    feature_clean_df = feature_df.dropna(subset=FEATURE_COLUMNS + ["target_direction"]).copy()

    # Save complete dataset
    feature_clean_df.to_csv(output_file, index=False)

    print(f"Enhanced predictive dataset built and saved to: {output_file}")
    print(f"Total rows: {len(feature_clean_df)}")
    print(f"Total features: {len(FEATURE_COLUMNS)}")
    print(f"Date range: {feature_clean_df['date'].min().date()} → {feature_clean_df['date'].max().date()}")
    print("\nTarget Class Distribution:")
    print(feature_clean_df["target_direction"].value_counts(dropna=False))

    return feature_clean_df


if __name__ == "__main__":
    build_predictive_dataset()
