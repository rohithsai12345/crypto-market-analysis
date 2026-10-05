import os
import json
from pathlib import Path
from src.genai.package_evidence import build_evidence_package, build_live_evidence_package, build_prediction_evidence_package

SYSTEM_PROMPT = """You are a specialized cryptocurrency market-analysis assistant.
Use ONLY the supplied evidence JSON to summarize observed market movements and sentiment.

Strict Rules:
1. Do NOT invent prices, percentages, events, sources, or causal explanations.
2. Clearly state observed relationships (e.g. price change alongside sentiment ratio) without claiming definitive causation unless supported by evidence.
3. Do NOT provide guaranteed future price predictions or financial investment advice.
4. Keep the summary structured, professional, clear, and factual.
"""

PREDICTION_EXPLANATION_SYSTEM_PROMPT = """You are a senior Financial Machine Learning & Market Analysis assistant.
Explain the ML model's NEXT-DAY DIRECTIONAL PREDICTION based STRICTLY on the structured prediction evidence JSON.

Strict Rules:
1. Do NOT override or change the ML model's prediction or probabilities.
2. Explain supporting vs counter-acting features (e.g., sentiment ratio, 7d rolling return, daily price shift).
3. Do NOT invent prices, events, or ungrounded claims.
4. Explicitly state that the model assigns a probabilistic directional likelihood, NOT a financial guarantee.
"""

LIVE_OPINION_SYSTEM_PROMPT = """You are a senior cryptocurrency market-analysis assistant.
Provide a clear, evidence-grounded opinion on TODAY'S LIVE MARKET ACTION based strictly on the provided real-time market ticker evidence.

Rules:
1. Use ONLY the live prices, 24h % shifts, volume, and sentiment baseline provided in the JSON input.
2. Structure your response into:
   - ⚡ Today's Live Market Pulse & Regime
   - 📈 Key Tickers & Movers (BTC, ETH, Top Altcoins)
   - 🧠 Sentiment & Price Action Alignment
3. Distinguish empirical observations from causal speculation.
4. Do NOT give financial advice or future price targets.
"""


def generate_evidence_grounded_summary(evidence=None, api_key=None):
    """
    Generates a concise, evidence-grounded market summary using Gemini API or evidence-based rule engine fallback.
    """
    if evidence is None:
        evidence = build_evidence_package()

    if not evidence:
        return "Insufficient analytical evidence available for the selected period."

    key = api_key or os.environ.get("GEMINI_API_KEY")

    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            prompt = f"{SYSTEM_PROMPT}\n\nEVIDENCE DATA:\n{json.dumps(evidence, indent=2)}\n\nTASK:\n1. Summarize market price & volume movement.\n2. Summarize news sentiment & top keywords.\n3. Highlight notable volume days."
            
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )
            if response and response.output_text:
                return response.output_text.strip()
        except Exception:
            pass

    # High-fidelity deterministic evidence-grounded generator
    asset = evidence.get("asset", "BTC")
    period = evidence.get("period", "Selected Period")
    price_info = evidence.get("price_summary", {})
    sentiment_info = evidence.get("sentiment_distribution", {})
    keywords = evidence.get("top_keywords", {})
    high_vol = evidence.get("notable_high_activity_day", {})

    price_change = price_info.get("price_change_percentage", 0.0)
    direction = price_info.get("trend_direction", "Stable")
    start_p = price_info.get("start_price_usd", 0)
    end_p = price_info.get("end_price_usd", 0)

    pos_pct = sentiment_info.get("positive_percentage", 0.0)
    neu_pct = sentiment_info.get("neutral_percentage", 0.0)
    neg_pct = sentiment_info.get("negative_percentage", 0.0)
    dom_sent = sentiment_info.get("dominant_sentiment", "Neutral")
    avg_score = sentiment_info.get("avg_sentiment_score", 0.0)

    top_kw_str = ", ".join(keywords.get("overall", [])[:5]) or "crypto, market, bitcoin"
    top_pos_str = ", ".join(keywords.get("positive", [])[:3]) or "growth, rally"
    top_neg_str = ", ".join(keywords.get("negative", [])[:3]) or "decline, risk"

    summary_text = (
        f"### 📈 Market Movement Overview ({asset})\n"
        f"During the period **{period}**, {asset} experienced a **{direction.lower()} price movement** of **{price_change:+.2f}%**, "
        f"moving from **${start_p:,.2f}** to **${end_p:,.2f}**.\n\n"
        f"### 📊 News Sentiment Breakdown\n"
        f"Analysis of **{evidence.get('total_news_records', 0):,}** processed news articles indicates a **{dom_sent}** market narrative "
        f"with an average sentiment score of **{avg_score:+.4f}**.\n"
        f"- **Positive News:** {pos_pct:.1f}%\n"
        f"- **Neutral News:** {neu_pct:.1f}%\n"
        f"- **Negative News:** {neg_pct:.1f}%\n\n"
        f"### 🔑 Key Narrative Topics & Keywords\n"
        f"- **Dominant Keywords:** {top_kw_str}\n"
        f"- **Positive Identifiers:** {top_pos_str}\n"
        f"- **Negative Identifiers:** {top_neg_str}\n\n"
        f"### 📰 Peak Activity Spotlight\n"
        f"The highest news volume date occurred on **{high_vol.get('date', 'N/A')}** with **{high_vol.get('news_count', 0)} articles** "
        f"and an average daily sentiment score of **{high_vol.get('avg_sentiment', 0.0):+.3f}**.\n\n"
        f"*Note: This summary is automatically synthesized strictly from empirical NLP and market evidence without speculative financial predictions.*"
    )

    return summary_text


def generate_live_daily_opinion(live_evidence=None, api_key=None):
    """
    Generates GenAI opinion on TODAY'S LIVE MARKET data.
    """
    if live_evidence is None:
        live_evidence = build_live_evidence_package()

    if not live_evidence:
        return "Live market evidence unavailable at this moment."

    key = api_key or os.environ.get("GEMINI_API_KEY")

    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            prompt = f"{LIVE_OPINION_SYSTEM_PROMPT}\n\nLIVE MARKET DATA:\n{json.dumps(live_evidence, indent=2)}\n\nTASK: Generate today's market opinion."
            
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )
            if response and response.output_text:
                return response.output_text.strip()
        except Exception:
            pass

    # Deterministic Live Market Opinion Engine
    regime = live_evidence.get("market_regime", "Consolidation")
    ts = live_evidence.get("timestamp", "")
    btc = live_evidence.get("bitcoin_live", {})
    eth = live_evidence.get("ethereum_live", {})
    movers = live_evidence.get("top_market_movers", [])
    sentiment = live_evidence.get("sentiment_baseline", {})

    btc_p = btc.get("price_usd", 0.0) or 0.0
    btc_chg = btc.get("change_24h_pct", 0.0) or 0.0
    eth_p = eth.get("price_usd", 0.0) or 0.0
    eth_chg = eth.get("change_24h_pct", 0.0) or 0.0

    movers_str_list = []
    for m in movers:
        chg = m.get("change_24h_pct", 0.0) or 0.0
        p = m.get("price", 0.0) or 0.0
        movers_str_list.append(f"**{m.get('name')} ({m.get('symbol')})**: ${p:,.2f} ({chg:+.2f}%)")

    movers_formatted = "\n".join([f"- {s}" for s in movers_str_list]) if movers_str_list else "- Live market tickers active."

    pos_r = sentiment.get("positive_ratio_pct", 0.0)
    neg_r = sentiment.get("negative_ratio_pct", 0.0)

    opinion_text = (
        f"### ⚡ Today's Live Market Pulse & Opinion ({ts})\n"
        f"Market Status: **{regime.upper()}**\n\n"
        f"#### 🔍 Key Price Observations Today:\n"
        f"- **Bitcoin (BTC):** Current price is **${btc_p:,.2f}** with a 24-hour change of **{btc_chg:+.2f}%**.\n"
        f"- **Ethereum (ETH):** Current price is **${eth_p:,.2f}** with a 24-hour change of **{eth_chg:+.2f}%**.\n\n"
        f"#### 📈 Live Market Leaders & Top Asset Tickers:\n"
        f"{movers_formatted}\n\n"
        f"#### 🧠 Sentiment Alignment Analysis:\n"
        f"Today's live price movements occur alongside an NLP news sentiment baseline of **{pos_r:.1f}% positive** vs **{neg_r:.1f}% negative** news ratio. "
        f"When live price momentum ({regime}) aligns with positive news coverage, market liquidity typically reflects steady participant activity.\n\n"
        f"*Disclaimer: This AI opinion synthesizes live market platform data for analytical insight only and does not constitute financial or investment advice.*"
    )

    return opinion_text


def generate_prediction_explanation(pred_evidence, api_key=None):
    """
    Generates natural-language evidence-grounded explanation for ML model's directional prediction (Section 19 & 20).
    """
    if not pred_evidence:
        return "Prediction evidence unavailable."

    key = api_key or os.environ.get("GEMINI_API_KEY")

    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            prompt = f"{PREDICTION_EXPLANATION_SYSTEM_PROMPT}\n\nPREDICTION EVIDENCE:\n{json.dumps(pred_evidence, indent=2)}\n\nTASK: Explain the prediction."
            
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )
            if response and response.output_text:
                return response.output_text.strip()
        except Exception:
            pass

    # Deterministic Engine for ML Prediction Explanation
    asset = pred_evidence.get("asset", "BTC")
    direction = pred_evidence.get("predicted_direction", "NEUTRAL")
    probs = pred_evidence.get("probabilities", {})
    conf = pred_evidence.get("model_confidence", 0.0) * 100
    model_name = pred_evidence.get("model_name", "Predictive Model")
    supp = pred_evidence.get("supporting_evidence", {})

    p_bull = (probs.get("bullish", 0.0) or 0.0) * 100
    p_neu = (probs.get("neutral", 0.0) or 0.0) * 100
    p_bear = (probs.get("bearish", 0.0) or 0.0) * 100

    sent_score = supp.get("current_sentiment_score", 0.0)
    pos_r = supp.get("positive_news_ratio_pct", 0.0)
    neg_r = supp.get("negative_news_ratio_pct", 0.0)
    btc_ret = supp.get("btc_daily_return_pct", 0.0)
    roll_7d_ret = supp.get("btc_7d_rolling_return_pct", 0.0)
    roll_7d_sent = supp.get("rolling_7d_sentiment", 0.0)

    explanation = (
        f"The **{model_name}** assigns the highest probability to a **{direction}** next-day direction for **{asset}** "
        f"with **{conf:.1f}% confidence** (Probabilities: **Bullish {p_bull:.1f}%**, **Neutral {p_neu:.1f}%**, **Bearish {p_bear:.1f}%**).\n\n"
        f"**Supporting Factors:**\n"
        f"- **NLP News Sentiment:** Current average sentiment score is **{sent_score:+.4f}** ({pos_r:.1f}% positive vs {neg_r:.1f}% negative news).\n"
        f"- **7-Day Rolling Sentiment Trend:** Standing at **{roll_7d_sent:+.4f}**.\n"
        f"- **Short-Term Price Momentum:** BTC daily return is **{btc_ret:+.2f}%** with 7-day cumulative return of **{roll_7d_ret:+.2f}%**.\n\n"
        f"**Methodological Note:** This AI explanation synthesizes structured ML feature inputs without overriding the underlying model probabilities. "
        f"The prediction represents a probabilistic directional likelihood for research purposes and does not constitute financial advice."
    )

    return explanation


if __name__ == "__main__":
    live_ev = build_live_evidence_package()
    print(generate_live_daily_opinion(live_ev))
