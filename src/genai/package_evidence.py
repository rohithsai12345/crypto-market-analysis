import json
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

try:
    from dashboard.services.market_api import get_live_coin_market
except ImportError:
    try:
        from services.market_api import get_live_coin_market
    except ImportError:
        get_live_coin_market = None


def build_evidence_package(start_date=None, end_date=None, asset="BTC"):
    """
    Assembles structured evidence JSON from analytical pipeline outputs
    strictly adhering to Section 6 (GenAI Prompt & Data Design).
    """
    daily_path = DATA_DIR / "daily_sentiment.csv"
    mkt_path = DATA_DIR / "market_sentiment_analysis.csv"
    kw_path = DATA_DIR / "keyword_analysis.csv"

    if not daily_path.exists() or not mkt_path.exists():
        return {}

    daily_df = pd.read_csv(daily_path, parse_dates=["date"])
    mkt_df = pd.read_csv(mkt_path, parse_dates=["date"])
    kw_df = pd.read_csv(kw_path) if kw_path.exists() else pd.DataFrame()

    # Filter date range
    if start_date:
        daily_df = daily_df[daily_df["date"] >= pd.to_datetime(start_date)]
        mkt_df = mkt_df[mkt_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        daily_df = daily_df[daily_df["date"] <= pd.to_datetime(end_date)]
        mkt_df = mkt_df[mkt_df["date"] <= pd.to_datetime(end_date)]

    if len(daily_df) == 0 or len(mkt_df) == 0:
        return {}

    # Market price and volume stats
    price_col = f"{asset.lower()}_close"
    volume_col = f"{asset.lower()}_volume"

    start_price = float(mkt_df[price_col].iloc[0]) if price_col in mkt_df else 0.0
    end_price = float(mkt_df[price_col].iloc[-1]) if price_col in mkt_df else 0.0
    total_price_change_pct = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0.0

    avg_volume = float(mkt_df[volume_col].mean()) if volume_col in mkt_df else 0.0

    # Sentiment distribution
    pos_pct = float(daily_df["positive_ratio"].mean() * 100)
    neu_pct = float(daily_df["neutral_ratio"].mean() * 100)
    neg_pct = float(daily_df["negative_ratio"].mean() * 100)
    avg_sent = float(daily_df["avg_sentiment"].mean())

    # Dominant sentiment classification
    if pos_pct > neg_pct and pos_pct > neu_pct:
        dominant_sentiment = "Positive / Bullish"
    elif neg_pct > pos_pct and neg_pct > neu_pct:
        dominant_sentiment = "Negative / Bearish"
    else:
        dominant_sentiment = "Neutral / Mixed"

    # Top keywords
    top_overall_keywords = []
    top_pos_keywords = []
    top_neg_keywords = []

    if not kw_df.empty:
        top_overall_keywords = kw_df[kw_df["category"] == "overall"]["keyword"].head(8).tolist()
        top_pos_keywords = kw_df[kw_df["category"] == "positive"]["keyword"].head(5).tolist()
        top_neg_keywords = kw_df[kw_df["category"] == "negative"]["keyword"].head(5).tolist()

    # High news volume day
    highest_vol_row = daily_df.nlargest(1, "news_count").iloc[0]
    high_volume_event = {
        "date": str(highest_vol_row["date"].date()),
        "news_count": int(highest_vol_row["news_count"]),
        "avg_sentiment": round(float(highest_vol_row["avg_sentiment"]), 3)
    }

    evidence = {
        "asset": asset.upper(),
        "period": f"{daily_df['date'].min().date()} to {daily_df['date'].max().date()}",
        "days_count": len(daily_df),
        "total_news_records": int(daily_df["news_count"].sum()),
        "price_summary": {
            "start_price_usd": round(start_price, 2),
            "end_price_usd": round(end_price, 2),
            "price_change_percentage": round(total_price_change_pct, 2),
            "trend_direction": "Upward" if total_price_change_pct >= 0 else "Downward"
        },
        "volume_summary": {
            "average_daily_volume_usd": round(avg_volume, 2)
        },
        "sentiment_distribution": {
            "avg_sentiment_score": round(avg_sent, 4),
            "dominant_sentiment": dominant_sentiment,
            "positive_percentage": round(pos_pct, 1),
            "neutral_percentage": round(neu_pct, 1),
            "negative_percentage": round(neg_pct, 1)
        },
        "top_keywords": {
            "overall": top_overall_keywords,
            "positive": top_pos_keywords,
            "negative": top_neg_keywords
        },
        "notable_high_activity_day": high_volume_event
    }

    return evidence


def build_live_evidence_package(live_coins=None, live_prices=None):
    """
    Builds real-time evidence package for Today's Market, integrating:
    - Real-time live prices & 24h changes (Binance Spot / CoinGecko)
    - Latest NLP sentiment signals
    - Real-time market regime classification
    """
    if live_coins is None and callable(get_live_coin_market):
        try:
            live_coins = get_live_coin_market()
        except Exception:
            live_coins = []

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract BTC & ETH live metrics
    btc_info = {}
    eth_info = {}
    top_movers = []

    if live_coins:
        for coin in live_coins:
            sym = str(coin.get("symbol", "")).upper()
            if sym == "BTC":
                btc_info = {
                    "price_usd": coin.get("current_price"),
                    "change_24h_pct": coin.get("price_change_percentage_24h"),
                    "volume_24h_usd": coin.get("total_volume"),
                    "market_cap_usd": coin.get("market_cap")
                }
            elif sym == "ETH":
                eth_info = {
                    "price_usd": coin.get("current_price"),
                    "change_24h_pct": coin.get("price_change_percentage_24h"),
                    "volume_24h_usd": coin.get("total_volume"),
                    "market_cap_usd": coin.get("market_cap")
                }
            
            if sym in ["BTC", "ETH"]:
                top_movers.append({
                    "rank": coin.get("market_cap_rank"),
                    "name": coin.get("name"),
                    "symbol": sym,
                    "price": coin.get("current_price"),
                    "change_24h_pct": coin.get("price_change_percentage_24h")
                })

    # Fallback if websocket prices are active
    if live_prices:
        if live_prices.get("BTC") and not btc_info.get("price_usd"):
            btc_info["price_usd"] = live_prices["BTC"]
        if live_prices.get("ETH") and not eth_info.get("price_usd"):
            eth_info["price_usd"] = live_prices["ETH"]

    # Pull latest sentiment baseline
    sentiment_baseline = {}
    daily_path = DATA_DIR / "daily_sentiment.csv"
    if daily_path.exists():
        daily_df = pd.read_csv(daily_path)
        latest_row = daily_df.iloc[-1]
        sentiment_baseline = {
            "avg_sentiment_score": round(float(latest_row["avg_sentiment"]), 4),
            "positive_ratio_pct": round(float(latest_row["positive_ratio"]) * 100, 1),
            "neutral_ratio_pct": round(float(latest_row["neutral_ratio"]) * 100, 1),
            "negative_ratio_pct": round(float(latest_row["negative_ratio"]) * 100, 1),
        }

    # Classify Today's Live Market Regime
    btc_chg = btc_info.get("change_24h_pct", 0) or 0
    eth_chg = eth_info.get("change_24h_pct", 0) or 0

    if btc_chg >= 2.0 and eth_chg >= 2.0:
        regime = "Bullish Expansion"
    elif btc_chg <= -2.0 and eth_chg <= -2.0:
        regime = "Bearish Pullback"
    elif abs(btc_chg) > 4.0 or abs(eth_chg) > 4.0:
        regime = "High Volatility"
    else:
        regime = "Consolidation / Mixed"

    live_evidence = {
        "timestamp": today_str,
        "market_regime": regime,
        "bitcoin_live": btc_info,
        "ethereum_live": eth_info,
        "sentiment_baseline": sentiment_baseline,
        "top_market_movers": top_movers[:5]
    }

    return live_evidence


def build_prediction_evidence_package(pred_record, feature_dict):
    """
    Constructs structured evidence JSON for LLM to explain ML directional predictions (Section 19 & 20).
    """
    prediction_evidence = {
        "asset": pred_record.get("asset", "BTC"),
        "timestamp": pred_record.get("timestamp"),
        "current_price": pred_record.get("current_price"),
        "prediction_horizon": pred_record.get("prediction_horizon", "1D"),
        "model_name": pred_record.get("model_name"),
        "model_version": pred_record.get("model_version"),
        "predicted_direction": pred_record.get("predicted_direction"),
        "probabilities": {
            "bullish": pred_record.get("prob_bullish"),
            "neutral": pred_record.get("prob_neutral"),
            "bearish": pred_record.get("prob_bearish")
        },
        "model_confidence": pred_record.get("confidence"),
        "supporting_evidence": {
            "current_sentiment_score": round(float(feature_dict.get("avg_sentiment", 0.0)), 4),
            "positive_news_ratio_pct": round(float(feature_dict.get("positive_ratio", 0.0)) * 100, 1),
            "negative_news_ratio_pct": round(float(feature_dict.get("negative_ratio", 0.0)) * 100, 1),
            "rolling_7d_sentiment": round(float(feature_dict.get("rolling_sentiment_7d", 0.0)), 4),
            "btc_daily_return_pct": round(float(feature_dict.get("btc_return", 0.0)) * 100, 2),
            "btc_7d_rolling_return_pct": round(float(feature_dict.get("rolling_return_7d", 0.0)) * 100, 2),
            "btc_7d_volatility_pct": round(float(feature_dict.get("rolling_volatility_7d", 0.0)) * 100, 2)
        }
    }
    return prediction_evidence


if __name__ == "__main__":
    live_pkg = build_live_evidence_package()
    print(json.dumps(live_pkg, indent=2))
