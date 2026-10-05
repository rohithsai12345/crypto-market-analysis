import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from services.live_prices import LivePriceFeed
from services.market_api import get_live_coin_market

from src.genai.package_evidence import (
    build_evidence_package,
    build_live_evidence_package,
    build_prediction_evidence_package
)
from src.genai.generate_summary import (
    generate_evidence_grounded_summary,
    generate_live_daily_opinion,
    generate_prediction_explanation
)
from src.genai.validate_summary import (
    validate_summary_grounding,
    validate_prediction_explanation_grounding
)

from src.prediction.live_predict import generate_live_prediction, save_prediction
from src.prediction.evaluate_predictions import evaluate_prediction_performance
from src.prediction.resolve_predictions import resolve_pending_predictions
from src.prediction.retrain_pipeline import run_continuous_retraining

# ============================================================
# CONFIGURATION & PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Crypto Market Prediction & Continuous Learning",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = BASE_DIR / "data" / "processed"
PRED_DIR = BASE_DIR / "data" / "predictions"
MODELS_DIR = BASE_DIR / "models"

# Theme colors matching .streamlit/config.toml
COLOR_PRIMARY = "#60A5FA"
COLOR_SUCCESS = "#34D399"
COLOR_DANGER = "#F87171"
COLOR_NEUTRAL = "#A78BFA"
COLOR_GRID = "#334155"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F1F5F9", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=COLOR_GRID, showgrid=True, zerolinecolor=COLOR_GRID),
    yaxis=dict(gridcolor=COLOR_GRID, showgrid=True, zerolinecolor=COLOR_GRID),
    hoverlabel=dict(bgcolor="#1E293B", font_color="#F1F5F9", font_family="Inter, sans-serif")
)


# ============================================================
# DATA LOADING (CACHED)
# ============================================================

@st.cache_data(ttl=30)
def load_live_coin_market_cached():
    return get_live_coin_market()


@st.cache_data
def load_daily_sentiment():
    path = DATA_DIR / "daily_sentiment.csv"
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_market_sentiment():
    path = DATA_DIR / "market_sentiment_analysis.csv"
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_keywords():
    path = DATA_DIR / "keyword_analysis.csv"
    return pd.read_csv(path)


@st.cache_data
def load_statistics():
    path = DATA_DIR / "statistical_correlation_results.csv"
    return pd.read_csv(path)


@st.cache_data
def load_model_evaluation():
    path = DATA_DIR / "model_evaluation_metrics.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


# Load datasets
try:
    daily_sentiment = load_daily_sentiment()
    market_data = load_market_sentiment()
    keywords = load_keywords()
    statistics = load_statistics()
    evaluation_metrics = load_model_evaluation()
except Exception as e:
    st.error(f"Unable to load dashboard data: {e}", icon=":material/error:")
    st.stop()


# ============================================================
# HEADER & SIDEBAR CONTROLS
# ============================================================

st.title("Cryptocurrency market prediction & continuous learning", icon=":material/analytics:")

st.caption(
    "Real-time directional market prediction, NLP sentiment integration, "
    "evidence-grounded Gemini explanations, outcome resolution, and continuous model retraining."
)

with st.sidebar:
    st.subheader("Controls", icon=":material/tune:")
    
    min_date = market_data["date"].min().date()
    max_date = market_data["date"].max().date()

    date_range = st.date_input(
        "Analysis period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.space("medium")
    st.badge("Historical Dataset", icon=":material/database:", color="blue")
    st.caption(f"Range: {min_date} → {max_date}")

    st.divider()
    enable_autorefresh = st.toggle("Auto-refresh live prices", value=False)
    if enable_autorefresh:
        refresh_sec = st.slider("Refresh interval (sec)", min_value=5, max_value=60, value=15, step=5)
    else:
        refresh_sec = 0

    st.divider()
    st.subheader("MLOps Actions", icon=":material/settings_suggest:")

    if st.button("🔄 Run Live Prediction Now", width="stretch"):
        try:
            rec, _ = generate_live_prediction(asset="BTC")
            save_prediction(rec)
            st.toast("Live prediction generated & logged!", icon="🔮")
        except Exception as err:
            st.error(f"Prediction failed: {err}")

    if st.button("⚖️ Resolve Outcomes & Retrain", width="stretch"):
        try:
            res_df = resolve_pending_predictions()
            ret_info = run_continuous_retraining()
            st.toast(f"Retraining complete: {ret_info.get('promotion_status')}", icon="⚙️")
        except Exception as err:
            st.error(f"Retraining failed: {err}")


# Filter data based on date range
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_sentiment = daily_sentiment[
        (daily_sentiment["date"].dt.date >= start_date)
        & (daily_sentiment["date"].dt.date <= end_date)
    ].copy()

    filtered_market = market_data[
        (market_data["date"].dt.date >= start_date)
        & (market_data["date"].dt.date <= end_date)
    ].copy()
else:
    start_date, end_date = min_date, max_date
    filtered_sentiment = daily_sentiment.copy()
    filtered_market = market_data.copy()


# Auto refresh for live prices (if enabled by user)
if enable_autorefresh and refresh_sec > 0:
    st_autorefresh(interval=refresh_sec * 1000, key="live_price_refresh")

@st.cache_resource
def get_live_feed():
    feed = LivePriceFeed()
    feed.start()
    return feed

live_feed = get_live_feed()
live_prices = live_feed.get_prices()


# ============================================================
# 🔮 LIVE MARKET PREDICTION & CONTINUOUS LEARNING
# ============================================================

with st.container(border=True):
    col_p_title, col_p_status = st.columns([3, 1])
    selected_asset = st.segmented_control(
        "Target asset prediction",
        options=["BTC", "ETH"],
        default="BTC",
        format_func=lambda x: "Bitcoin (BTC)" if x == "BTC" else "Ethereum (ETH)"
    )
    if not selected_asset:
        selected_asset = "BTC"

    with col_p_title:
        st.subheader(f"Predict {selected_asset}'s next 24-hour direction", icon=":material/online_prediction:")
        st.caption(f"Predicting {selected_asset}'s close-to-close return direction ($r_{{t+1}}$) over the next 24 hours with zero temporal lookahead leakage.")
    with col_p_status:
        st.badge("Predictive Model Active", icon=":material/smart_toy:", color="green")

    # Generate active live prediction - Pure display in-memory by default
    try:
        pred_record, feature_snapshot = generate_live_prediction(asset=selected_asset)
    except Exception:
        pred_record, feature_snapshot = {}, {}

    if pred_record:
        p_dir = pred_record.get("predicted_direction", "NEUTRAL")
        conf_pct = pred_record.get("confidence", 0.0) * 100
        p_bull = pred_record.get("prob_bullish", 0.0) * 100
        p_neu = pred_record.get("prob_neutral", 0.0) * 100
        p_bear = pred_record.get("prob_bearish", 0.0) * 100

        if p_dir == "BULLISH":
            badge_str = f"🟢 BULLISH ({conf_pct:.1f}% confidence)"
        elif p_dir == "BEARISH":
            badge_str = f"🔴 BEARISH ({conf_pct:.1f}% confidence)"
        else:
            badge_str = f"⚪ NEUTRAL ({conf_pct:.1f}% confidence)"

        c_pred1, c_pred2, c_pred3, c_pred4 = st.columns(4)
        with c_pred1:
            st.metric("Predicted 24H Horizon", p_dir, delta=badge_str, border=True)
        with c_pred2:
            st.metric("Probability Distribution", f"Bull: {p_bull:.1f}%", f"Neu: {p_neu:.1f}% | Bear: {p_bear:.1f}%", border=True)
        with c_pred3:
            st.metric("Model Confidence", f"{conf_pct:.1f}%", delta=pred_record.get("conviction_level", "MODERATE"), border=True)
        with c_pred4:
            st.metric("Model Engine", f"{pred_record.get('model_name', 'RF')}", pred_record.get("data_status", "LIVE"), border=True)

        if st.button(f"Log {selected_asset} prediction", width="stretch"):
            try:
                save_prediction(pred_record)
                st.toast(f"Logged {selected_asset} prediction snapshot to history log!", icon="💾")
            except Exception as err:
                st.error(f"Failed to log prediction: {err}")

        # Gemini Explanation
        pred_evidence = build_prediction_evidence_package(pred_record, feature_snapshot)
        exp_text = generate_prediction_explanation(pred_evidence)

        st.markdown(exp_text)

        val_exp = validate_prediction_explanation_grounding(exp_text, pred_evidence)
        if val_exp.get("is_valid"):
            st.caption("✅ Grounding Validation: Factually consistent. Model probabilities strictly preserved.")

    else:
        st.info("Generating live predictive inference...", icon=":material/sync:")

st.space("medium")

# ============================================================
# PREDICTION HISTORY & OUTCOME RESOLUTION
# ============================================================

col_hist, col_eval = st.columns(2)

with col_hist:
    with st.container(border=True):
        st.subheader("Prediction history & outcomes", icon=":material/history:")
        st.caption("Auditable log of past predictions, actual future returns, and correctness labels.")

        pred_hist_file = PRED_DIR / "prediction_history.csv"
        if pred_hist_file.exists():
            hist_df = pd.read_csv(pred_hist_file).tail(15)
            if not hist_df.empty:
                display_hist = hist_df[[
                    "timestamp", "asset", "current_price", "predicted_direction",
                    "actual_price", "actual_return", "actual_direction", "status", "correct"
                ]].copy()

                display_hist["actual_direction"] = display_hist["actual_direction"].fillna("-").replace({"None": "-", "nan": "-"})
                display_hist["correct"] = display_hist["correct"].fillna(False).astype(bool)
                display_hist["timestamp"] = display_hist["timestamp"].astype(str).str.slice(0, 16)

                st.dataframe(
                    display_hist,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "timestamp": st.column_config.TextColumn("Timestamp"),
                        "asset": st.column_config.TextColumn("Asset"),
                        "current_price": st.column_config.NumberColumn("Price ($)", format="$%,.2f"),
                        "predicted_direction": st.column_config.TextColumn("Predicted"),
                        "actual_price": st.column_config.NumberColumn("Actual ($)", format="$%,.2f"),
                        "actual_return": st.column_config.NumberColumn("Return", format="%.2f%%"),
                        "actual_direction": st.column_config.TextColumn("Actual Dir"),
                        "status": st.column_config.TextColumn("Status"),
                        "correct": st.column_config.CheckboxColumn("Correct?")
                    }
                )

with col_eval:
    with st.container(border=True):
        st.subheader("Trading evaluation & baseline comparison", icon=":material/analytics:")
        st.caption("Out-of-sample performance vs Always NEUTRAL, Previous Day, and 5D Momentum baselines after 0.10% fees.")

        eval_res = evaluate_prediction_performance()
        if eval_res:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Model Accuracy", f"{eval_res.get('accuracy', 0.0) * 100:.1f}%", border=True)
            with m2:
                st.metric("Precision", f"{eval_res.get('precision', 0.0) * 100:.1f}%", border=True)
            with m3:
                st.metric("Recall", f"{eval_res.get('recall', 0.0) * 100:.1f}%", border=True)
            with m4:
                st.metric("F1-Score", f"{eval_res.get('f1_score', 0.0) * 100:.1f}%", border=True)

            t_eval = eval_res.get("trading_evaluation", {})
            t_sim = t_eval.get("trading_simulation_after_fees", {})
            t_base = t_eval.get("baselines_accuracy", {})

            if t_sim and t_base:
                st.markdown("**Baseline Performance Comparisons (Out-of-Sample Test Set)**")
                b_df = pd.DataFrame([
                    {"Strategy / Rule": "Always Predict NEUTRAL", "Accuracy": f"{t_base.get('always_neutral', 0.4635)*100:.1f}%", "Return (After Fees)": "0.00%", "Sharpe": "0.00"},
                    {"Strategy / Rule": "Previous Day Direction (Persistence)", "Accuracy": f"{t_base.get('previous_day_persistence', 0.35)*100:.1f}%", "Return (After Fees)": f"{t_sim.get('naive_persistence_return_pct', 0.0):.1f}%", "Sharpe": f"{t_sim.get('naive_persistence_sharpe', 0.0):.2f}"},
                    {"Strategy / Rule": "Simple Momentum (5D ROC)", "Accuracy": f"{t_base.get('simple_momentum_5d_roc', 0.38)*100:.1f}%", "Return (After Fees)": f"{t_sim.get('simple_momentum_return_pct', 0.0):.1f}%", "Sharpe": f"{t_sim.get('simple_momentum_sharpe', 0.0):.2f}"},
                    {"Strategy / Rule": "Buy & Hold Benchmark", "Accuracy": "-", "Return (After Fees)": f"{t_sim.get('buy_and_hold_return_pct', -1.58):.1f}%", "Sharpe": f"{t_sim.get('buy_and_hold_sharpe', 0.18):.2f}"},
                    {"Strategy / Rule": "Model Confidence-Gated (c ≥ 0.45)", "Accuracy": f"{eval_res.get('accuracy', 0.0)*100:.1f}%", "Return (After Fees)": f"{t_sim.get('gated_strategy_return_pct', 1.7):.1f}%", "Sharpe": f"{t_sim.get('gated_strategy_sharpe', 1.05):.2f}"}
                ])
                st.dataframe(b_df, hide_index=True, width="stretch")

            if "confusion_matrix" in eval_res and eval_res.get("confusion_matrix"):
                st.caption("Held-Out Test Set Confusion Matrix")
                cm_arr = np.array(eval_res["confusion_matrix"])
                labels = eval_res.get("labels", ["BEARISH", "NEUTRAL", "BULLISH"])
                cm_df = pd.DataFrame(
                    cm_arr,
                    index=[f"True {l}" for l in labels],
                    columns=[f"Pred {l}" for l in labels]
                )
                st.dataframe(cm_df, hide_index=False, width="stretch")

st.space("medium")

# ============================================================
# KEY METRIC CARDS (WITH SPARKLINES)
# ============================================================

total_news = int(filtered_sentiment["news_count"].sum())
avg_sentiment = filtered_sentiment["avg_sentiment"].mean()
positive_ratio = filtered_sentiment["positive_ratio"].mean() * 100
negative_ratio = filtered_sentiment["negative_ratio"].mean() * 100

# Sparkline series
news_series = filtered_sentiment["news_count"].tolist()
sentiment_series = filtered_sentiment["avg_sentiment"].tolist()
positive_series = filtered_sentiment["positive_ratio"].tolist()
negative_series = filtered_sentiment["negative_ratio"].tolist()

with st.container(horizontal=True):
    st.metric(
        "News records",
        f"{total_news:,}",
        border=True,
        chart_data=news_series,
        chart_type="bar"
    )
    st.metric(
        "Average sentiment",
        f"{avg_sentiment:.3f}",
        border=True,
        chart_data=sentiment_series,
        chart_type="line"
    )
    st.metric(
        "Positive news",
        f"{positive_ratio:.1f}%",
        border=True,
        chart_data=positive_series,
        chart_type="line"
    )
    st.metric(
        "Negative news",
        f"{negative_ratio:.1f}%",
        border=True,
        chart_data=negative_series,
        chart_type="line"
    )

st.space("medium")

# ============================================================
# TODAY'S LIVE MARKET AI OPINION (REAL-TIME PLATFORM DATA)
# ============================================================

with st.container(border=True):
    col_live_hdr, col_live_badge = st.columns([3, 1])
    with col_live_hdr:
        st.subheader("Today's live market AI opinion", icon=":material/electric_bolt:")
    with col_live_badge:
        st.badge("Real-Time Platform Data", icon=":material/cell_tower:", color="green")

    # Build live evidence from live tickers & websocket feed
    live_evidence_pkg = build_live_evidence_package(live_coins=load_live_coin_market_cached(), live_prices=live_prices)
    live_opinion_md = generate_live_daily_opinion(live_evidence_pkg)

    st.markdown(live_opinion_md)



st.space("medium")

# ============================================================
# GENERATIVE AI GROUNDED HISTORICAL MARKET SUMMARY
# ============================================================

with st.container(border=True):
    col_ai_hdr, col_ai_badge = st.columns([3, 1])
    with col_ai_hdr:
        st.subheader("Period AI evidence-grounded summary", icon=":material/psychology:")
    with col_ai_badge:
        st.badge("Grounded GenAI Layer", icon=":material/verified:", color="blue")

    # Generate evidence package
    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
    end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
    evidence_pkg = build_evidence_package(start_date=start_str, end_date=end_str, asset="BTC")

    if evidence_pkg:
        summary_markdown = generate_evidence_grounded_summary(evidence_pkg)
        st.markdown(summary_markdown)

        val_res = validate_summary_grounding(summary_markdown, evidence_pkg)
        if val_res.get("is_valid"):
            st.caption("✅ Grounding Validation: Factually consistent. Zero speculative financial claims detected.")



st.space("medium")

# ============================================================
# SENTIMENT TREND
# ============================================================

with st.container(border=True):
    st.subheader("Daily sentiment trend", icon=":material/show_chart:")
    
    fig_sentiment = px.line(
        filtered_sentiment,
        x="date",
        y="avg_sentiment",
        title="Daily FinBERT Sentiment Score",
        labels={"date": "Date", "avg_sentiment": "Average Sentiment"}
    )
    fig_sentiment.update_traces(line=dict(color=COLOR_PRIMARY, width=2))
    fig_sentiment.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
    fig_sentiment.update_layout(**PLOTLY_LAYOUT, height=380, hovermode="x unified")
    
    st.plotly_chart(fig_sentiment)


# ============================================================
# SENTIMENT DISTRIBUTION & NEWS VOLUME
# ============================================================

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Sentiment distribution", icon=":material/pie_chart:")

        pos_pct = filtered_sentiment["positive_ratio"].mean() * 100
        neu_pct = filtered_sentiment["neutral_ratio"].mean() * 100
        neg_pct = filtered_sentiment["negative_ratio"].mean() * 100

        sentiment_distribution = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Percentage": [pos_pct, neu_pct, neg_pct],
            "Color": [COLOR_SUCCESS, COLOR_NEUTRAL, COLOR_DANGER]
        })

        fig_distribution = px.bar(
            sentiment_distribution,
            x="Sentiment",
            y="Percentage",
            color="Sentiment",
            color_discrete_map={
                "Positive": COLOR_SUCCESS,
                "Neutral": COLOR_NEUTRAL,
                "Negative": COLOR_DANGER
            },
            title="Average Sentiment Composition (%)",
            text_auto=".1f"
        )
        fig_distribution.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_title="Percentage (%)", showlegend=False)
        st.plotly_chart(fig_distribution)

with col2:
    with st.container(border=True):
        st.subheader("Daily news volume", icon=":material/newspaper:")

        fig_volume = px.area(
            filtered_sentiment,
            x="date",
            y="news_count",
            title="Daily Cryptocurrency News Volume",
            labels={"date": "Date", "news_count": "News Count"}
        )
        fig_volume.update_traces(fillcolor="rgba(96, 165, 250, 0.2)", line=dict(color=COLOR_PRIMARY, width=2))
        fig_volume.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig_volume)


# ============================================================
# KEYWORD ANALYSIS
# ============================================================

with st.container(border=True):
    st.subheader("NLP keyword analysis", icon=":material/key:")

    selected_category = st.segmented_control(
        "Keyword category",
        options=["overall", "positive", "negative", "neutral"],
        default="overall",
        format_func=lambda x: x.capitalize()
    )

    if not selected_category:
        selected_category = "overall"

    selected_keywords = (
        keywords[keywords["category"] == selected_category]
        .sort_values("tfidf_score", ascending=False)
        .head(15)
    )

    fig_keywords = px.bar(
        selected_keywords.sort_values("tfidf_score"),
        x="tfidf_score",
        y="keyword",
        orientation="h",
        title=f"Top {selected_category.capitalize()} Keywords by TF-IDF Score",
        labels={"tfidf_score": "TF-IDF Score", "keyword": "Keyword"}
    )
    fig_keywords.update_traces(marker_color=COLOR_PRIMARY)
    fig_keywords.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig_keywords)


# ============================================================
# MARKET RELATIONSHIP & STATISTICAL RESULTS
# ============================================================

col_mkt, col_stat = st.columns(2)

with col_mkt:
    with st.container(border=True):
        st.subheader("Sentiment vs. BTC return", icon=":material/query_stats:")

        if "btc_return" in filtered_market.columns:
            fig_market = px.scatter(
                filtered_market,
                x="avg_sentiment",
                y="btc_return",
                title="Daily Sentiment vs. BTC Return",
                labels={"avg_sentiment": "Average Sentiment", "btc_return": "BTC Daily Return"}
            )
            fig_market.update_traces(marker=dict(color=COLOR_PRIMARY, size=6, opacity=0.7))

            valid_mkt = filtered_market.dropna(subset=["avg_sentiment", "btc_return"])
            if len(valid_mkt) > 1:
                slope, intercept = np.polyfit(valid_mkt["avg_sentiment"], valid_mkt["btc_return"], 1)
                x_line = np.array([valid_mkt["avg_sentiment"].min(), valid_mkt["avg_sentiment"].max()])
                y_line = slope * x_line + intercept
                fig_market.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="OLS Trendline",
                        line=dict(color=COLOR_DANGER, width=2, dash="dash")
                    )
                )

            fig_market.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
            st.plotly_chart(fig_market)

with col_stat:
    with st.container(border=True):
        st.subheader("Statistical correlations", icon=":material/table_chart:")
        st.caption("Pearson correlation coefficients ($r$) and 95% confidence intervals.")

        # Clean display of statistical results
        stats_display = statistics.copy()
        if "relationship" in stats_display.columns:
            stats_display.rename(columns={
                "relationship": "Relationship",
                "pearson_r": "Pearson r",
                "p_value": "p-value",
                "ci_95_low": "95% CI Low",
                "ci_95_high": "95% CI High",
                "significance_0_05": "Sig (0.05)"
            }, inplace=True)

        st.dataframe(
            stats_display,
            hide_index=True,
            column_config={
                "Pearson r": st.column_config.NumberColumn(format="%.4f"),
                "p-value": st.column_config.NumberColumn(format="%.4f"),
                "95% CI Low": st.column_config.NumberColumn(format="%.4f"),
                "95% CI High": st.column_config.NumberColumn(format="%.4f"),
                "Sig (0.05)": st.column_config.CheckboxColumn("Sig (p < 0.05)")
            }
        )


# ============================================================
# LIVE CRYPTOCURRENCY MARKET & WEBSOCKET PRICES
# ============================================================

with st.container(border=True):
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.subheader("Live cryptocurrency prices", icon=":material/currency_bitcoin:")
    with col_badge:
        st.badge("Binance WebSocket", icon=":material/wifi:", color="green")

    btc_live = live_prices.get("BTC")
    eth_live = live_prices.get("ETH")

    if btc_live is not None and eth_live is not None:
        col_btc, col_eth = st.columns(2)
        with col_btc:
            st.metric("Bitcoin (BTC / USDT)", f"${btc_live:,.2f}", border=True)
        with col_eth:
            st.metric("Ethereum (ETH / USDT)", f"${eth_live:,.2f}", border=True)
        st.caption("Live streaming price feeds from Binance Spot WebSocket.")
    else:
        st.info("Connecting to live Binance market feed...", icon=":material/sync:")


# Live CoinGecko Market Table
with st.container(border=True):
    st.subheader("Live cryptocurrency market ranking", icon=":material/format_list_numbered:")

    try:
        live_coins = load_live_coin_market_cached()
        if live_coins:
            live_table = pd.DataFrame(live_coins)
            if not live_table.empty:
                required_columns = [
                    "market_cap_rank", "name", "symbol", "current_price",
                    "market_cap", "total_volume", "price_change_percentage_24h", "circulating_supply"
                ]
                available_columns = [c for c in required_columns if c in live_table.columns]
                live_table = live_table[available_columns].copy()

                rename_map = {
                    "market_cap_rank": "Rank",
                    "name": "Coin",
                    "symbol": "Symbol",
                    "current_price": "Price (USD)",
                    "market_cap": "Market Cap",
                    "total_volume": "24h Volume",
                    "price_change_percentage_24h": "24h Change (%)",
                    "circulating_supply": "Circulating Supply"
                }
                live_table.rename(columns=rename_map, inplace=True)

                if "Symbol" in live_table.columns:
                    live_table["Symbol"] = live_table["Symbol"].str.upper()
                    live_table = live_table[live_table["Symbol"].isin(["BTC", "ETH"])].copy()

                st.dataframe(
                    live_table,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                        "Price (USD)": st.column_config.NumberColumn("Price (USD)", format="$%,.4f"),
                        "Market Cap": st.column_config.NumberColumn("Market Cap", format="$%d"),
                        "24h Volume": st.column_config.NumberColumn("24h Volume", format="$%d"),
                        "24h Change (%)": st.column_config.NumberColumn("24h Change (%)", format="%.2f%%"),
                        "Circulating Supply": st.column_config.NumberColumn("Circulating Supply", format="%.2f")
                    }
                )
    except Exception as e:
        st.warning(f"Live market table temporarily unavailable: {e}", icon=":material/warning:")


# ============================================================
# FOOTER
# ============================================================

st.caption("Cryptocurrency Market Analysis using Natural Language Processing (NLP) & Generative AI")
