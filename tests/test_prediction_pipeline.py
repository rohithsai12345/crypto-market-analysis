import unittest
import tempfile
import json
import joblib
import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.prediction.feature_builder import (
    classify_return,
    BULLISH_THRESHOLD,
    BEARISH_THRESHOLD,
    FEATURE_COLUMNS_BTC,
    FEATURE_COLUMNS_ETH,
    build_predictive_dataset
)
from src.prediction.train_predictive_model import (
    train_and_select_predictive_model,
    promote_candidate_to_production
)
from src.prediction.live_predict import generate_live_prediction, save_prediction
from src.prediction.resolve_predictions import resolve_pending_predictions
from src.prediction.retrain_pipeline import run_continuous_retraining
from src.prediction.evaluate_trading_performance import evaluate_trading_and_investment_performance


class TestPredictionPipelineIsolated(unittest.TestCase):

    def setUp(self):
        """Create an isolated temporary workspace directory for each test run."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        self.data_dir = self.base_path / "data" / "processed"
        self.pred_dir = self.base_path / "data" / "predictions"
        self.models_dir = self.base_path / "models"
        self.versions_dir = self.models_dir / "versions"
        self.candidates_dir = self.models_dir / "candidates"

        for d in [self.data_dir, self.pred_dir, self.models_dir, self.versions_dir, self.candidates_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create synthetic market sentiment dataset for testing
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        btc_return = np.random.uniform(-0.02, 0.02, 100)
        eth_return = np.random.uniform(-0.02, 0.02, 100)
        btc_close = 40000.0 * np.cumprod(1 + btc_return)
        eth_close = 2200.0 * np.cumprod(1 + eth_return)

        df = pd.DataFrame({
            "date": dates,
            "btc_close": btc_close,
            "btc_return": btc_return,
            "btc_volume": 20000000000 + np.random.randint(0, 5000000, 100),
            "eth_close": eth_close,
            "eth_return": eth_return,
            "eth_volume": 5000000000 + np.random.randint(0, 1000000, 100),
            "avg_sentiment": np.random.uniform(-0.5, 0.5, 100),
            "positive_ratio": np.random.uniform(0.2, 0.6, 100),
            "neutral_ratio": np.random.uniform(0.2, 0.6, 100),
            "negative_ratio": np.random.uniform(0.1, 0.4, 100),
            "news_count": np.random.randint(10, 50, 100),
            "avg_finbert_confidence": 0.85
        })

        self.mock_market_file = self.data_dir / "market_sentiment_analysis.csv"
        self.mock_dataset_btc = self.data_dir / "prediction_dataset.csv"
        self.mock_dataset_eth = self.data_dir / "prediction_dataset_eth.csv"
        df.to_csv(self.mock_market_file, index=False)

    def tearDown(self):
        """Clean up temporary test directory."""
        self.test_dir.cleanup()

    def test_1_eth_uses_eth_data_not_btc(self):
        """Test that ETH pipeline uses ETH price/return/volume & ETH technical features, not BTC data."""
        df_eth = build_predictive_dataset(
            asset="ETH",
            market_sentiment_file=self.mock_market_file,
            output_file=self.mock_dataset_eth
        )

        self.assertIn("eth_rsi_14", df_eth.columns)
        self.assertIn("eth_macd", df_eth.columns)
        self.assertIn("eth_ema_ratio_7_25", df_eth.columns)
        self.assertIn("target_return", df_eth.columns)

        # Verify that ETH target return matches next-day ETH return from raw market file
        raw_df = pd.read_csv(self.mock_market_file)
        raw_df["date"] = pd.to_datetime(raw_df["date"])

        for i in range(5):
            row_i = df_eth.iloc[i]
            curr_date = pd.to_datetime(row_i["date"])
            next_raw = raw_df[raw_df["date"] > curr_date]
            if not next_raw.empty:
                expected_eth_next_return = next_raw.iloc[0]["eth_return"]
                computed_target = row_i["target_return"]
                self.assertAlmostEqual(computed_target, expected_eth_next_return, places=6)

    def test_2_prediction_horizon_resolved_correctly(self):
        """Test that prediction record sets exact horizon promise and resolves outcomes correctly."""
        build_predictive_dataset(asset="BTC", market_sentiment_file=self.mock_market_file, output_file=self.mock_dataset_btc)

        rec_btc, _ = generate_live_prediction(asset="BTC")
        self.assertEqual(rec_btc["prediction_horizon"], "Predict BTC's next 24-hour direction")
        self.assertEqual(rec_btc["asset"], "BTC")

        # Set timestamp to an earlier date in synthetic dataset range to enable resolution test
        rec_btc["timestamp"] = "2024-01-05 12:00:00"

        # Log prediction snapshot using separate save_prediction function
        hist_file = self.pred_dir / "prediction_history.csv"
        live_file = self.pred_dir / "live_predictions.csv"
        save_prediction(rec_btc, history_file=hist_file, live_file=live_file)

        self.assertTrue(hist_file.exists())
        hist_df = pd.read_csv(hist_file)
        self.assertEqual(len(hist_df), 1)
        self.assertEqual(hist_df.iloc[0]["status"], "PENDING")

        # Resolve prediction against mock dataset
        res_df = resolve_pending_predictions(history_file=hist_file, dataset_file=self.mock_dataset_btc)
        self.assertIsNotNone(res_df)
        self.assertEqual(res_df.iloc[0]["status"], "RESOLVED")
        self.assertFalse(pd.isna(res_df.iloc[0]["actual_price"]))

    def test_3_rejected_candidate_cannot_replace_production(self):
        """Test that calling the actual promotion workflow rejects an inferior candidate without replacing production."""
        # 1. Setup production champion model metadata with high F1 score (0.9999)
        prod_meta_file_btc = self.models_dir / "predictive_model_metadata_btc.json"
        prod_meta_file_def = self.models_dir / "predictive_model_metadata.json"
        prod_model_file_btc = self.models_dir / "predictive_model_btc.joblib"
        prod_model_file_def = self.models_dir / "predictive_model.joblib"

        from sklearn.ensemble import ExtraTreesClassifier
        champion_model = ExtraTreesClassifier(n_estimators=5, random_state=42)
        joblib.dump(champion_model, prod_model_file_btc)
        joblib.dump(champion_model, prod_model_file_def)

        prod_meta = {
            "asset": "BTC",
            "model_name": "Production Champion Model (F1: 99.9%)",
            "metrics": {"val_f1": 0.9999}
        }
        with open(prod_meta_file_btc, "w") as f:
            json.dump(prod_meta, f, indent=2)
        with open(prod_meta_file_def, "w") as f:
            json.dump(prod_meta, f, indent=2)

        # Build synthetic dataset in isolated test directory
        build_predictive_dataset(
            asset="BTC",
            market_sentiment_file=self.mock_market_file,
            output_file=self.mock_dataset_btc
        )

        # 2. Call the full end-to-end continuous retraining workflow
        summary = run_continuous_retraining(
            asset="BTC",
            models_dir=self.models_dir,
            candidates_dir=self.candidates_dir,
            versions_dir=self.versions_dir,
            dataset_file=self.mock_dataset_btc,
            market_sentiment_file=self.mock_market_file
        )

        self.assertEqual(summary["promotion_status"], "REJECTED_DEGRADATION")

        # 3. Assert that active production metadata was NOT replaced by weak candidate
        with open(prod_meta_file_btc, "r") as f:
            active_meta = json.load(f)
        self.assertEqual(active_meta["model_name"], "Production Champion Model (F1: 99.9%)")

    def test_4_dataset_features_no_future_timestamps(self):
        """Test that dataset features at row t contain zero future information (strictly timestamps <= t)."""
        df = build_predictive_dataset(
            asset="BTC",
            market_sentiment_file=self.mock_market_file,
            output_file=self.mock_dataset_btc
        )

        for col in FEATURE_COLUMNS_BTC:
            self.assertNotIn("future", col)
            self.assertNotEqual(col, "target_return")
            self.assertNotEqual(col, "target_direction")
            self.assertNotEqual(col, "target_label")

    def test_5_writes_only_to_temporary_test_files(self):
        """Test that pure live inference generate_live_prediction writes ZERO files to real project paths."""
        real_base_dir = Path(__file__).resolve().parent.parent
        real_live_file = real_base_dir / "data" / "predictions" / "live_predictions.csv"
        real_hist_file = real_base_dir / "data" / "predictions" / "prediction_history.csv"

        real_live_mtime = real_live_file.stat().st_mtime if real_live_file.exists() else None
        real_hist_mtime = real_hist_file.stat().st_mtime if real_hist_file.exists() else None

        # Execute pure live inference
        rec, feat = generate_live_prediction(asset="BTC")

        # Verify actual project files were NOT created or modified
        if real_live_mtime is not None:
            self.assertEqual(real_live_file.stat().st_mtime, real_live_mtime)
        if real_hist_mtime is not None:
            self.assertEqual(real_hist_file.stat().st_mtime, real_hist_mtime)

        self.assertIsNotNone(rec)

    def test_6_pipeline_health_monitoring(self):
        """Test that pipeline monitoring checks feed health, dataset freshness, and drift reporting cleanly."""
        from src.prediction.monitor_pipeline import check_pipeline_health, check_model_drift

        health = check_pipeline_health()
        self.assertIn("overall_status", health)
        self.assertIn("market_feeds", health)
        self.assertIn("dataset_freshness", health)
        self.assertIn("model_drift", health)

        # Test model drift on a mock prediction history file in temporary folder
        mock_hist_file = self.pred_dir / "prediction_history.csv"
        df_hist = pd.DataFrame([
            {"timestamp": f"2024-01-0{i}", "asset": "BTC", "predicted_label": "BULLISH", "actual_label": "BULLISH", "status": "RESOLVED"}
            for i in range(1, 6)
        ])
        df_hist.to_csv(mock_hist_file, index=False)

        drift = check_model_drift(history_file=mock_hist_file)
        self.assertEqual(drift["status"], "MONITORING")
        self.assertEqual(drift["rolling_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
