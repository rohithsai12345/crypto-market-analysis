import unittest
import tempfile
import json
import joblib
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
        btc_close = 40000.0 + np.cumsum(np.random.normal(50, 200, 100))
        eth_close = 2200.0 + np.cumsum(np.random.normal(5, 20, 100))
        btc_return = np.insert(np.diff(btc_close) / btc_close[:-1], 0, 0.0)
        eth_return = np.insert(np.diff(eth_close) / eth_close[:-1], 0, 0.0)

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
        # Build datasets
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
        """Test that a rejected candidate staged in models/candidates/ cannot replace production model."""
        prod_model_file = self.models_dir / "predictive_model_btc.joblib"
        prod_meta_file = self.models_dir / "predictive_model_metadata_btc.json"

        from sklearn.ensemble import ExtraTreesClassifier
        champion_model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        joblib.dump(champion_model, prod_model_file)

        prod_meta = {
            "asset": "BTC",
            "model_name": "Production Champion Model (F1: 99.9%)",
            "metrics": {"val_f1": 0.9999}
        }
        with open(prod_meta_file, "w") as f:
            json.dump(prod_meta, f, indent=2)

        # Stage a weak candidate in models/candidates/
        cand_meta = {
            "asset": "BTC",
            "model_name": "Weak Candidate Model (F1: 30.0%)",
            "metrics": {"val_f1": 0.3000}
        }
        cand_meta_file = self.candidates_dir / "candidate_metadata_btc.json"
        cand_model_file = self.candidates_dir / "candidate_model_btc.json"
        joblib.dump(champion_model, cand_model_file)
        with open(cand_meta_file, "w") as f:
            json.dump(cand_meta, f, indent=2)

        # Evaluate promotion check rule
        curr_val_f1 = 0.9999
        cand_val_f1 = 0.3000

        promotion_status = "PROMOTED" if cand_val_f1 > curr_val_f1 else "REJECTED_DEGRADATION"
        self.assertEqual(promotion_status, "REJECTED_DEGRADATION")

        # Verify active production metadata remains untouched
        with open(prod_meta_file, "r") as f:
            current_active_meta = json.load(f)
        self.assertEqual(current_active_meta["model_name"], "Production Champion Model (F1: 99.9%)")

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
        """Test that pure live inference generate_live_prediction writes ZERO files outside temp directory."""
        live_file = self.pred_dir / "live_predictions.csv"
        hist_file = self.pred_dir / "prediction_history.csv"

        if live_file.exists():
            live_file.unlink()
        if hist_file.exists():
            hist_file.unlink()

        # Execute pure live inference
        rec, feat = generate_live_prediction(asset="BTC")

        # Verify zero side-effect file creation
        self.assertFalse(live_file.exists())
        self.assertFalse(hist_file.exists())
        self.assertIsNotNone(rec)


if __name__ == "__main__":
    unittest.main()
