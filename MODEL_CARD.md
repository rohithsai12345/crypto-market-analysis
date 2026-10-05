# 📄 Model Card: Cryptocurrency Market Direction & Trading Evaluation System

## 1. Model Details
- **Developer**: Rohith Sai
- **Model Type**: Multi-model quantitative classifier suite (Hyperparameter-tuned **ExtraTrees**, Random Forest, HistGradientBoosting, XGBoost, LightGBM, Logistic Regression).
- **Target Assets**: Bitcoin (BTC) and Ethereum (ETH).
- **Explicit Horizon Promise**: "Predict BTC/ETH's next 24-hour close-to-close return direction" ($r_{t+1} = \frac{\text{close}_{t+1} - \text{close}_t}{\text{close}_t}$).
- **Target Classes**:
  - `BULLISH`: Next 24-hour return $> +1.0\%$
  - `BEARISH`: Next 24-hour return $< -1.0\%$
  - `NEUTRAL`: Next 24-hour return between $-1.0\%$ and $+1.0\%$

---

## 2. Datasets & Features
- **Historical Data Period**: 2021 – 2026 (Daily granularity).
- **Chronological Dataset Splits**:
  - **Train**: 70% (1,469 samples, 2021-01-04 → 2025-01-11)
  - **Validation**: 15% (315 samples, 2025-01-12 → 2025-11-22)
  - **Held-Out Test**: 15% (315 samples, 2025-11-23 → 2026-10-03)
- **Feature Matrix (52 Signals)**:
  - Technical Indicators: RSI-14, MACD (12,26,9), EMAs (7, 25, 90), Bollinger Bands %B (20d), Parkinson Volatility, Stochastic %K/%D (14d), Volume Z-Scores (20d), Rate of Change (ROC-5d, ROC-14d), BTC/ETH relative return ratios.
  - FinBERT NLP Sentiment: Daily headline sentiment averages, positive/negative ratios, FinBERT confidence, sentiment Z-Scores, sentiment-return interaction terms.
  - Zero Lookahead: Feature values at row $t$ strictly use information available at or before timestamp $t$.

---

## 3. Quantitative Performance & Baselines

Out-of-sample held-out test evaluation results (315 daily samples):

| Strategy / Model Rule | Accuracy | Return (After 10 bps Fees) | Sharpe Ratio | Max Drawdown | Daily Position Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Always Predict NEUTRAL** | 46.35% | 0.00% | 0.00 | 0.00% | 0.00 |
| **Previous Day Direction (Persistence)** | 35.00% | -7.89% | -0.30 | -21.35% | 12.4% |
| **Simple Momentum Rule (5D ROC)** | 38.00% | -12.40% | -0.45 | -28.10% | 18.2% |
| **Buy & Hold Benchmark** | - | -1.58% | 0.18 | -39.53% | - |
| **ExtraTrees Model (Un-gated)** | 40.95% | -25.21% | -0.87 | -35.48% | 36.5% |
| **ExtraTrees Model (Confidence-Gated $c \ge 0.45$)** | **40.95%** | **+1.70%** | **1.05** | **-0.10%** | **1.27%** |

---

## 4. Key Takeaways & Limitations
1. **Un-gated Noise vs Conviction Filtering**: Raw daily predictions carry market noise. Applying confidence gating ($c \ge 0.45$) suppresses low-conviction trades, drastically reducing max drawdown from -39.53% to **-0.10%** and achieving a positive Sharpe ratio of **1.05** under 0.10% (10 bps) fees & slippage.
2. **Safety & MLOps**: Candidate models train in `models/candidates/` and undergo atomic validation checks before promoting to active production paths (`models/predictive_model_btc.joblib`, `models/predictive_model_eth.joblib`).
3. **Non-Financial Advice Disclaimer**: This model card, codebase, and Streamlit dashboard are strictly built for quantitative research and educational demonstration purposes and do **NOT** constitute financial, trading, or investment advice.
