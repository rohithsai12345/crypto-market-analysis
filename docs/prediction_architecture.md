# Real-Time Cryptocurrency Market Prediction & Continuous Learning Architecture

## Academic & Disclaimer Notice

> **IMPORTANT:** This system provides **probabilistic directional predictions** for research and educational purposes. It **does not constitute financial or investment advice**. The system predicts the **NEXT-DAY BTC MARKET DIRECTION** based on historical market behavior, NLP news sentiment, and real-time live platform data. Correlation is explicitly distinguished from causation.

---

## 1. System Architecture Overview

```
Historical Data (Market + FinBERT Sentiment)
      ↓
Feature Engineering (src/prediction/feature_builder.py)
      ↓
Target Generation: Next-Day Return (BULLISH > +1%, BEARISH < -1%, NEUTRAL)
      ↓
Chronological Train (70%) / Validation (15%) / Test (15%) Split (No Shuffling)
      ↓
Model Training & Selection (src/prediction/train_predictive_model.py)
      ↓
Promoted Production Model (models/predictive_model.joblib)
      ↓
Live Platform Tickers (WebSocket / REST API) + Latest News Signals
      ↓
Live Feature Schema Mapping (src/prediction/live_predict.py)
      ↓
Live Prediction (P(BULLISH), P(NEUTRAL), P(BEARISH), Model Confidence)
      ↓
Gemini 3.8 Flash Evidence Explanation (src/genai/generate_summary.py)
      ↓
Prediction Log (data/predictions/prediction_history.csv)
      ↓
Actual Future Market Outcome Resolution (src/prediction/resolve_predictions.py)
      ↓
Continuous Learning Retraining & Promotion Rule (src/prediction/retrain_pipeline.py)
      ↓
Streamlit Interactive Dashboard (dashboard/app.py)
```

---

## 2. Prediction Target & Configurable Thresholds

- **Target Asset:** Bitcoin (`BTC`)
- **Prediction Horizon:** Next-Day ($t+1$)
- **Return Formula:** $R_{t+1} = \frac{\text{Close}_{t+1} - \text{Close}_t}{\text{Close}_t}$
- **Configurable Thresholds:**
  - `BULLISH_THRESHOLD = +0.01` ($+1.0\%$) $\rightarrow$ Class `BULLISH` (2)
  - `BEARISH_THRESHOLD = -0.01` ($-1.0\%$) $\rightarrow$ Class `BEARISH` (0)
  - Otherwise $\rightarrow$ Class `NEUTRAL` (1)

---

## 3. Zero Temporal Leakage Principle

To ensure true predictive validity and avoid data leakage:
1. Feature vector at time $t$ uses **ONLY** historical information available at or before time $t$.
2. Future return $R_{t+1}$ and target direction are shifted backward during feature generation and NEVER present in feature inputs (`FEATURE_COLUMNS`).
3. Dataset splitting is strictly **Chronological** ($Train < Validation < Test$). No random shuffling is performed.

---

## 4. Model Selection & Versioning

- **Evaluated Models:** Logistic Regression, Linear SVM, Random Forest, Gradient Boosting.
- **Model Promotion Rule:** Candidate models trained during continuous retraining are evaluated on a chronological validation window. A new candidate model is promoted to `models/current_model.joblib` ONLY if validation F1-score does not degrade and passes sanity checks.
- **Model Versioning:** All promoted models are versioned in `models/versions/model_<timestamp>.joblib` alongside metadata in `models/predictive_model_metadata.json`.

---

## 5. MLOps Continuous Learning Loop

1. **Inference:** Generate prediction at $t$ and log to `data/predictions/prediction_history.csv` with status `"PENDING"`.
2. **Resolution:** After horizon $t+1$ completes, `resolve_pending_predictions()` retrieves actual future market return, compares predicted vs actual direction, marks `correct = True/False`, and resolves status.
3. **Retraining:** `run_continuous_retraining()` appends newly labeled observations, retrains model, applies promotion rules, and updates active production model.
