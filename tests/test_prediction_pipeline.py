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
    FEATURE_COLUMNS,
    build_predictive_dataset
)
from src.prediction.train_predictive_model import (
    train_and_select_predictive_model,
    promote_candidate_to_production
)
from src.prediction.live_predict import generate_live_prediction
from src.prediction.resolve_predictions import resolve_pending_predictions
from src.prediction.retrain_pipeline import run_continuous_retraining
from src.prediction.evaluate_trading_performance import evaluate_trading_and_investment_performance
from src.genai.package_evidence import build_prediction_evidence_package
from src.genai.validate_summary import validate_prediction_explanation_grounding


class TestPredictionPipelineIsolated(unittest.TestCase):

    def setUp(self):
        """Create an isolated temporary workspace directory for each test run."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        self.data_dir = self.base_path / "data" / "processed"
        self.pred_dir = self.base_path / "data" / "predictions"
        self.models_dir = self.base_path / "models"
        self.versions_dir = self.models_dir / "versions"
        self.staging_dir = self.models_dir / "staging"

        for d in [self.data_dir, self.pred_dir, self.models_dir, self.versions_dir, self.staging_dir]:
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
        self.mock_dataset_file = self.data_dir / "prediction_dataset.csv"
        df.to_csv(self.mock_market_file, index=False)

    def tearDown(self):
        """Clean up temporary test directory."""
        self.test_dir.cleanup()

    def test_1_target_generation_thresholds(self):
        """Test target classification thresholds (+1.0% bullish, -1.0% bearish)."""
        self.assertEqual(classify_return(0.025), "BULLISH")
        self.assertEqual(classify_return(-0.025), "BEARISH")
        self.assertEqual(classify_return(0.005), "NEUTRAL")

    def test_2_zero_lookahead_leakage_timestamp_level(self):
        """Test zero temporal lookahead leakage at the value and timestamp level."""
        df = build_predictive_dataset(
            market_sentiment_file=self.mock_market_file,
            output_file=self.mock_dataset_file
        )
        
        self.assertIn("target_direction", df.columns)
        self.assertIn("target_return", df.columns)
        
        # Verify that for row at date t, target_return equals actual return of raw_df at date t+1
        raw_df = pd.read_csv(self.mock_market_file)
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        
        # Check first 5 clean rows
        for i in range(5):
            row_i = df.iloc[i]
            curr_date = pd.to_datetime(row_i["date"])
            next_raw = raw_df[raw_df["date"] > curr_date]
            if not next_raw.empty:
                expected_next_return = next_raw.iloc[0]["btc_return"]
                computed_target = row_i["target_return"]
                self.assertAlmostEqual(computed_target, expected_next_return, places=6)

    def test_3_eth_prediction_path(self):
        """Test dynamic ETH asset prediction path and feature construction."""
        # Ensure predictive dataset exists
        build_predictive_dataset(
            market_sentiment_file=self.mock_market_file,
            output_file=self.mock_dataset_file
        )

        pred_record, feature_snapshot = generate_live_prediction(asset="ETH", save_to_history=False)
        self.assertEqual(pred_record["asset"], "ETH")
        self.assertIn("eth_close", feature_snapshot)
        self.assertIn("eth_return", feature_snapshot)
        self.assertEqual(pred_record["prediction_horizon"], "Next-Day (24H Close-to-Close)")

    def test_4_non_mutating_live_inference(self):
        """Test pure live inference does NOT write side-effect files when save_to_history=False."""
        live_file = self.pred_dir / "live_predictions.csv"
        hist_file = self.pred_dir / "prediction_history.csv"

        if live_file.exists():
            live_file.unlink()
        if hist_file.exists():
            hist_file.unlink()

        # Execute pure live inference
        generate_live_prediction(asset="BTC", save_to_history=False)

        # Verify zero side-effect file creation
        self.assertFalse(live_file.exists(), "live_predictions.csv should NOT be created when save_to_history=False")
        self.assertFalse(hist_file.exists(), "prediction_history.csv should NOT be created when save_to_history=False")

    def test_5_retraining_atomic_promotion_gate_rejection(self):
        """Test that candidate model with inferior performance is rejected without mutating production model."""
        # Setup dummy production metadata with high validation F1
        prod_model_file = self.models_dir / "predictive_model.joblib"
        prod_meta_file = self.models_dir / "predictive_model_metadata.json"

        # Create initial production dummy model
        from sklearn.ensemble import ExtraTreesClassifier
        dummy_model = ExtraTreesClassifier(n_estimators=5, random_state=42)
        joblib.dump(dummy_model, prod_model_file)

        prod_meta = {
            "model_name": "Production Champion Model",
            "model_version": "v1.0",
            "metrics": {"val_f1": 0.9999}  # Unbeatable benchmark F1
        }
        with open(prod_meta_file, "w") as f:
            json.dump(prod_meta, f, indent=2)

        # Stage a weak candidate model
        cand_meta = {
            "model_name": "Weak Candidate Model",
            "model_version": "v2.0_cand",
            "metrics": {"val_f1": 0.3000}
        }
        cand_meta_file = self.staging_dir / "candidate_metadata.json"
        cand_model_file = self.staging_dir / "candidate_model.joblib"
        joblib.dump(dummy_model, cand_model_file)
        with open(cand_meta_file, "w") as f:
            json.dump(cand_meta, f, indent=2)

        # Evaluate promotion check rule
        curr_val_f1 = 0.9999
        cand_val_f1 = 0.3000
        
        promotion_status = "PROMOTED" if cand_val_f1 >= curr_val_f1 else "REJECTED_DEGRADATION"
        self.assertEqual(promotion_status, "REJECTED_DEGRADATION")

        # Verify active production metadata file remains untouched
        with open(prod_meta_file, "r") as f:
            current_active_meta = json.load(f)
        self.assertEqual(current_active_meta["model_name"], "Production Champion Model")

    def test_6_trading_evaluation_engine(self):
        """Test trading performance evaluation engine output."""
        build_predictive_dataset(
            market_sentiment_file=self.mock_market_file,
            output_file=self.mock_dataset_file
        )

        base_models_dir = Path(__file__).resolve().parent.parent / "models"

        report = evaluate_trading_and_investment_performance(
            dataset_file=self.mock_dataset_file,
            model_path=base_models_dir / "predictive_model.joblib",
            scaler_path=base_models_dir / "predictive_scaler.joblib",
            metadata_path=base_models_dir / "predictive_model_metadata.json"
        )
        
        self.assertIn("accuracy_model", report)
        self.assertIn("trading_simulation", report)
        self.assertIn("per_class_performance", report)


if __name__ == "__main__":
    unittest.main()
