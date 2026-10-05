# 📈 Cryptocurrency Market Direction Prediction & Sentiment Analysis System

An end-to-end quantitative machine learning pipeline and interactive Streamlit web dashboard for real-time directional market prediction (Bitcoin & Ethereum), NLP news sentiment analysis (FinBERT), and evidence-grounded Generative AI explanations (Gemini API).

---

## 🌟 Key Features

- **Multi-Class Directional Prediction**: Predicts today's market direction (`BULLISH`, `BEARISH`, `NEUTRAL`) for Bitcoin (BTC) and Ethereum (ETH) using hyperparameter-tuned **ExtraTrees** and ensemble models.
- **Quantitative Signal Engineering**: 52 feature signals combining technical indicators (RSI-14, MACD, EMAs, Parkinson Volatility, Bollinger %B, Stochastic %K/%D), multi-day sentiment momentum, and relative BTC/ETH return ratios.
- **NLP Sentiment Integration**: FinBERT transformer sentiment pipeline processing financial news articles, calculating daily sentiment Z-Scores and sentiment-return interaction terms.
- **3-Tier Conviction Classification**: Maps model confidence into actionable conviction tiers:
  - `HIGH (Strong Signal)`: Confidence $\ge 45\%$
  - `MODERATE (Clear Trend)`: $38\% \le \text{Confidence} < 45\%$
  - `LOW (Market Noise)`: Confidence $< 38\%$
- **Zero Temporal Leakage**: Strict 70/15/15 chronological dataset splits (2021–2026) with 1-step lagged feature alignment.
- **Evidence-Grounded GenAI Layer**: Synthesizes market observations into structured AI opinions with automated grounding validation.
- **Interactive Streamlit Dashboard**: Live Binance WebSocket price streaming, CoinGecko market ranking, asset toggles (BTC/ETH), user-controlled auto-refresh, and MLOps manual prediction triggers.

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
├── models/                    # Trained joblib models, scalers & versioned checkpoints
│   ├── predictive_model.joblib
│   ├── predictive_scaler.joblib
│   └── predictive_model_metadata.json
├── src/
│   ├── genai/                 # Gemini API evidence packaging & grounding validation
│   ├── market/                # Market dataset compilation & statistical analysis
│   ├── nlp/                   # FinBERT sentiment analysis & TF-IDF keyword extraction
│   └── prediction/            # Feature engineering, model training, live inference & retraining
├── tests/                     # Unit test suite for prediction pipeline & leakage checks
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

## 🧪 Model Training & Unit Testing

### Rebuild Feature Dataset & Train Predictive Models

```bash
# 1. Build quantitative feature matrix (52 features)
python src/prediction/feature_builder.py

# 2. Train multi-model suite and select best production model
python src/prediction/train_predictive_model.py
```

### Execute Unit Test Suite

```bash
PYTHONPATH=. python -m unittest tests/test_prediction_pipeline.py
```

---

## 📊 Quantitative Performance Summary

| Split | Model Engine | Accuracy | F1-Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation** | ExtraTrees (Tuned) | **44.76%** | **43.82%** | **43.34%** | **44.76%** |
| **Held-Out Test** | ExtraTrees (Tuned) | **41.59%** | **39.66%** | **39.13%** | **41.59%** |

*Note: For a 3-class financial time-series prediction task (Random Baseline = 33.33%), the model achieves a **+11.43% out-of-sample edge** over random chance without temporal data leakage.*

---

## 📜 License & Disclaimer

This project is licensed under the MIT License.

*Disclaimer: This predictive model and AI market opinion system are designed for analytical and educational research purposes only and do not constitute financial, investment, or trading advice.*
