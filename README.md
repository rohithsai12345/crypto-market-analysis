# 📈 Cryptocurrency Market Direction Prediction & Sentiment Analysis System

An end-to-end quantitative machine learning pipeline and interactive Streamlit web dashboard for real-time directional market prediction (Bitcoin & Ethereum), NLP news sentiment analysis (FinBERT), and evidence-grounded Generative AI explanations (Gemini API).

---

## 🌟 Key Features

- **Multi-Class Directional Prediction**: Predicts next-day market direction (`BULLISH`, `BEARISH`, `NEUTRAL`) for Bitcoin (BTC) and Ethereum (ETH) using hyperparameter-tuned **ExtraTrees** and ensemble models.
- **Explicit Horizon Promise**: Predicts the next-day (24-hour horizon) close-to-close return direction ($r_{t+1} = \frac{\text{close}_{t+1} - \text{close}_t}{\text{close}_t}$) with zero temporal lookahead leakage.
- **Quantitative Signal Engineering**: 52 feature signals combining technical indicators (RSI-14, MACD, EMAs, Parkinson Volatility, Bollinger %B, Stochastic %K/%D), multi-day sentiment momentum, and relative BTC/ETH return ratios.
- **NLP Sentiment Integration**: FinBERT transformer sentiment pipeline processing financial news articles, calculating daily sentiment Z-Scores and sentiment-return interaction terms.
- **Atomic Retraining Promotion Gate**: Candidate models stage in `models/staging/` and promote to production ONLY if validation performance passes metric gates.
- **Pure Live Inference**: Live inference functions in memory by default (`save_to_history=False` performs zero file writes).
- **Walk-Forward Trading Backtesting**: Realistic simulation with 0.10% (10 bps) transaction fees, position turnover, Sharpe ratio, Max Drawdown, and confidence-gated signal rules ($c \ge 0.45$).
- **Evidence-Grounded GenAI Layer**: Synthesizes market observations into structured AI opinions with automated grounding validation.
- **Interactive Streamlit Dashboard**: Live Binance WebSocket price streaming, CoinGecko market ranking, asset toggles (BTC/ETH), user-controlled auto-refresh, and explicit history logging.

---

## 🏗️ Project Architecture

```
crypto-market-analysis/
├── dashboard/                 # Streamlit UI & Web Components
│   ├── app.py                 # Main Streamlit Dashboard Entrypoint
│   └── services/              # Binance WebSocket & Market API integrations
├── data/
│   ├── processed/             # Cleaned datasets, daily sentiment, feature matrices
│   ├── predictions/           # Live prediction history & outcome logs
│   └── raw/                   # Raw price daily CSVs & news Parquet archives
├── docs/                      # Quantitative design & architectural documentation
├── models/                    # Production models, scalers, candidate staging & versioned backups
│   ├── staging/               # Staged candidate models evaluated during retraining
│   ├── versions/              # Versioned model checkpoints & rejected candidates
│   ├── predictive_model.joblib
│   ├── predictive_scaler.joblib
│   └── predictive_model_metadata.json
├── src/
│   ├── genai/                 # Gemini API evidence packaging & grounding validation
│   ├── market/                # Market dataset compilation & statistical analysis
│   ├── nlp/                   # FinBERT sentiment analysis & TF-IDF keyword extraction
│   └── prediction/            # Feature engineering, model training, live inference, backtesting & retraining
├── tests/                     # Isolated unit test suite for prediction pipeline & leakage checks
├── .gitignore                 # Version control exclusions
├── requirements.txt           # Python dependency manifest
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Virtual Environment

Ensure Python 3.10+ is installed:

```bash
git clone https://github.com/rohithsai12345/crypto-market-analysis.git
cd crypto-market-analysis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

Launch the interactive web application locally:

```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Model Training, Backtesting & Unit Testing

### Rebuild Feature Dataset, Train & Evaluate

```bash
# 1. Build quantitative feature matrix (52 features with t+1 target return)
python -m src.prediction.feature_builder

# 2. Train multi-model candidate suite with atomic staging & promotion gate
python -m src.prediction.train_predictive_model

# 3. Run walk-forward trading evaluation & backtest simulation
python -m src.prediction.evaluate_trading_performance
```

### Execute Isolated Unit Test Suite

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📊 Quantitative Performance & Trading Simulation

| Benchmark / Strategy | Total Return | Annualized Sharpe | Max Drawdown | Daily Turnover |
| :--- | :---: | :---: | :---: | :---: |
| **Buy & Hold Benchmark** | -1.58% | 0.18 | -39.53% | - |
| **Majority Class Baseline** | 0.00% | 0.00 | 0.00% | 0.00 |
| **Raw Un-gated Model Strategy** | -25.21% | -0.87 | -35.48% | 36.5% |
| **Confidence-Gated Strategy ($c \ge 0.45$)** | **+1.70%** | **1.05** | **-0.10%** | **1.27%** |

*Key Finding: While raw directional accuracy is 40.95% (vs 33.33% random baseline), raw un-gated signals carry market noise. Applying confidence gating ($c \ge 0.45$) filters out low-conviction signals, dramatically reducing max drawdown from -39.53% to **-0.10%** and achieving a positive Sharpe Ratio of **+1.05** under 10 bps transaction fees.*

---

## 📜 License & Disclaimer

This project is licensed under the MIT License.

*Disclaimer: This predictive model and AI market opinion system are designed for analytical and educational research purposes only and do not constitute financial, investment, or trading advice.*
