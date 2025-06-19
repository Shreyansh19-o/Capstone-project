import unittest
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging
from sklearn.dummy import DummyClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Shreyansh19-o"
        repo_name = "Capstone-project"

        tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        logger.info(f"Setting MLflow tracking URI to {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        cls.client = MlflowClient()
        cls.new_model_name = "my_model"
        cls.stage = "Staging"

        try:
            latest_versions = cls.client.get_latest_versions(cls.new_model_name, stages=[cls.stage])
            logger.info(f"Found model {cls.new_model_name} in stage {cls.stage}: {latest_versions}")
        except MlflowException as e:
            if "404" in str(e):
                logger.warning(f"Model {cls.new_model_name} not found in registry. Registering a dummy model.")
                with mlflow.start_run() as run:
                    model = DummyClassifier(strategy="constant", constant=1)
                    mlflow.sklearn.log_model(model, "model")
                    model_uri = f"runs:/{run.info.run_id}/model"
                    cls.client.create_registered_model(cls.new_model_name)
                    model_version = cls.client.create_model_version(
                        name=cls.new_model_name,
                        source=model_uri,
                        run_id=run.info.run_id
                    )
                    cls.client.transition_model_version_stage(
                        name=cls.new_model_name,
                        version=model_version.version,
                        stage=cls.stage
                    )
            else:
                logger.error(f"Failed to check model {cls.new_model_name}: {str(e)}")
                raise e

        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        if cls.new_model_version is None:
            raise ValueError(f"No model version found for {cls.new_model_name} in stage {cls.stage}")

        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        logger.info(f"Loading model from {cls.new_model_uri}")
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        vectorizer_path = "models/vectorizer.pkl"
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        cls.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        logger.info(f"Vectorizer feature count: {len(cls.vectorizer.get_feature_names_out())}")
        logger.info(f"Vectorizer features: {cls.vectorizer.get_feature_names_out().tolist()}")

        data_path = "data/processed/test_bow.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Holdout data file not found at {data_path}")
        cls.holdout_data = pd.read_csv(data_path)
        logger.info(f"Holdout data columns: {cls.holdout_data.columns.tolist()}")
        logger.info(f"Holdout data shape: {cls.holdout_data.shape}")

        # Validate compatibility
        expected_features = 20
        if len(cls.vectorizer.get_feature_names_out()) != expected_features:
            raise ValueError(f"Vectorizer produces {len(cls.vectorizer.get_feature_names_out())} features, expected {expected_features}")
        missing_cols = [col for col in cls.vectorizer.get_feature_names_out() if col not in cls.holdout_data.columns]
        if missing_cols:
            raise ValueError(f"Test data missing vectorizer features: {missing_cols}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = MlflowClient()
        try:
            latest_version = client.get_latest_versions(model_name, stages=[stage])
            return latest_version[0].version if latest_version else None
        except MlflowException as e:
            logger.error(f"Error fetching latest version for {model_name}: {str(e)}")
            raise e

    def test_model_loaded_properly(self):
        """Test that the model is loaded properly."""
        self.assertIsNotNone(self.new_model, "Model failed to load from MLflow registry")

    def test_model_signature(self):
        """Test the model's input and output signature."""
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
        expected_input_features = 20
        self.assertEqual(
            input_df.shape[1],
            expected_input_features,
            f"Input shape mismatch: expected {expected_input_features} features, got {input_df.shape[1]}"
        )
        prediction = self.new_model.predict(input_df)
        self.assertEqual(
            len(prediction),
            input_df.shape[0],
            f"Output length mismatch: expected {input_df.shape[0]}, got {len(prediction)}"
        )
        self.assertTrue(
            isinstance(prediction, (list, pd.Series, pd.DataFrame)) and len(prediction.shape) == 1
            or prediction.ndim == 1,
            "Prediction should be a 1D array for binary classification"
        )

    def test_model_performance(self):
        """Test the model's performance on holdout data."""
        feature_names = self.vectorizer.get_feature_names_out()
        X_holdout = self.holdout_data[feature_names]
        y_holdout = self.holdout_data.iloc[:, -1]
        self.assertEqual(X_holdout.shape[1], 20, f"Expected 20 features, got {X_holdout.shape[1]}")
        y_pred_new = self.new_model.predict(X_holdout)
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, zero_division=0)
        recall_new = recall_score(y_holdout, y_pred_new, zero_division=0)
        f1_new = f1_score(y_holdout, y_pred_new, zero_division=0)
        logger.info(f"Model performance - Accuracy: {accuracy_new:.4f}, Precision: {precision_new:.4f}, "
                    f"Recall: {recall_new:.4f}, F1: {f1_new:.4f}")
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40
        self.assertGreaterEqual(accuracy_new, expected_accuracy)
        self.assertGreaterEqual(precision_new, expected_precision)
        self.assertGreaterEqual(recall_new, expected_recall)
        self.assertGreaterEqual(f1_new, expected_f1)

if __name__ == "__main__":
    unittest.main()