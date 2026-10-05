import unittest
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
from src.prediction.train_predictive_model import train_and_select_predictive_model
from src.prediction.live_predict import generate_live_prediction
from src.prediction.resolve_predictions import resolve_pending_predictions
from src.prediction.retrain_pipeline import run_continuous_retraining
from src.genai.package_evidence import build_prediction_evidence_package
from src.genai.validate_summary import validate_prediction_explanation_grounding


class TestPredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure predictive dataset and model artifacts exist for test execution."""
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "predictive_model.joblib"
        scaler_path = models_dir / "predictive_scaler.joblib"
        metadata_path = models_dir / "predictive_model_metadata.json"

        dataset_file = base_dir / "data" / "processed" / "prediction_dataset.csv"
        if not dataset_file.exists():
            build_predictive_dataset()

        if not model_path.exists() or not scaler_path.exists():
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            df = pd.read_csv(dataset_file).dropna(subset=["target_label"])
            X = df[FEATURE_COLUMNS].values
            y = df["target_label"].astype(int).values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_scaled, y)

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)

            if not metadata_path.exists():
                import json
                meta = {
                    "model_name": "RandomForest (Test Fast)",
                    "is_scaled": True,
                    "training_timestamp": "test",
                    "feature_schema": FEATURE_COLUMNS,
                    "class_mapping": {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
                }
                with open(metadata_path, "w") as f:
                    json.dump(meta, f, indent=2)

    def test_1_target_generation(self):
        """Test target classification thresholds (+1.0% bullish, -1.0% bearish)."""
        self.assertEqual(classify_return(0.025), "BULLISH")
        self.assertEqual(classify_return(-0.025), "BEARISH")
        self.assertEqual(classify_return(0.005), "NEUTRAL")

    def test_2_no_lookahead_leakage(self):
        """Test that feature dataset contains zero temporal lookahead leakage."""
        df = build_predictive_dataset()
        self.assertIn("target_direction", df.columns)
        self.assertIn("target_return", df.columns)
        
        # Ensure target_return is NOT in FEATURE_COLUMNS schema
        for col in FEATURE_COLUMNS:
            self.assertNotEqual(col, "target_return")
            self.assertNotEqual(col, "target_direction")

    def test_3_feature_schema_consistency(self):
        """Test that live inference feature schema matches training feature schema."""
        pred_record, feature_snapshot = generate_live_prediction(asset="BTC")
        for col in FEATURE_COLUMNS:
            self.assertIn(col, feature_snapshot, f"Missing feature in live schema: {col}")

    def test_4_probability_bounds(self):
        """Test that predicted class probabilities are bounded between 0.0 and 1.0."""
        pred_record, _ = generate_live_prediction(asset="BTC")
        p_bull = pred_record["prob_bullish"]
        p_neu = pred_record["prob_neutral"]
        p_bear = pred_record["prob_bearish"]

        self.assertTrue(0.0 <= p_bull <= 1.0)
        self.assertTrue(0.0 <= p_neu <= 1.0)
        self.assertTrue(0.0 <= p_bear <= 1.0)
        self.assertAlmostEqual(p_bull + p_neu + p_bear, 1.0, places=2)

    def test_5_prediction_storage_and_resolution(self):
        """Test prediction storage logging and outcome resolution."""
        pred_record, _ = generate_live_prediction(asset="BTC")
        self.assertEqual(pred_record["status"], "PENDING")

        res_df = resolve_pending_predictions()
        self.assertIsNotNone(res_df)

    def test_6_gemini_evidence_validation(self):
        """Test GenAI prediction explanation grounding validator."""
        pred_record, feature_snapshot = generate_live_prediction(asset="BTC")
        ev_pkg = build_prediction_evidence_package(pred_record, feature_snapshot)
        
        explanation = f"The model predicts {pred_record['predicted_direction']} direction for BTC."
        val_res = validate_prediction_explanation_grounding(explanation, ev_pkg)
        self.assertTrue(val_res["is_valid"])


if __name__ == "__main__":
    unittest.main()
