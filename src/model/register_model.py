import json
import mlflow
import logging
import os
import dagshub
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment setup for DagsHub/MLflow (uncomment for production use)
# -------------------------------------------------------------------------------------
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"
# mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# -------------------------------------------------------------------------------------

# Local setup for DagsHub/MLflow
# -------------------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/Shreyansh19-o/Capstone-project.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    dagshub.init(repo_owner="Shreyansh19-o", repo_name="Capstone-project", mlflow=True)
    logging.info("DagsHub initialized successfully")
except Exception as e:
    logging.error("Failed to initialize DagsHub: %s", e)
    raise
# -------------------------------------------------------------------------------------

def load_model_info(file_path: str) -> dict:
    """Load model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        # Validate required keys
        required_keys = ["run_id", "model_path"]
        if not all(key in model_info for key in required_keys):
            raise ValueError(f"Missing required keys in {file_path}: {required_keys}")
        logging.info("Model info loaded from %s: %s", file_path, model_info)
        return model_info
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except json.JSONDecodeError:
        logging.error("Invalid JSON format in %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error while loading model info: %s", e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and set an alias."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        client = MlflowClient()

        # Verify the run exists
        try:
            client.get_run(model_info["run_id"])
            logging.info("Run %s verified", model_info["run_id"])
        except MlflowException as e:
            logging.error("Invalid run_id %s: %s", model_info["run_id"], e)
            raise

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info("Model %s version %s registered", model_name, model_version.version)

        # Set alias instead of stage (for MLflow 2.0+ compatibility)
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )
        logging.info("Model %s version %s assigned alias 'staging'", model_name, model_version.version)

        # Optionally transition to stage (for older MLflow versions, if needed)
        # client.transition_model_version_stage(
        #     name=model_name,
        #     version=model_version.version,
        #     stage="Staging"
        # )
        # logging.info("Model %s version %s transitioned to Staging stage", model_name, model_version.version)

    except MlflowException as e:
        logging.error("MLflow error during model registration: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error during model registration: %s", e)
        raise

def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)
        model_name = "my_model"
        register_model(model_name, model_info)
        logging.info("Model registration completed successfully")
        print("Model registration completed successfully. Load model using 'models:/my_model@staging' or 'models:/my_model/1'")
    except Exception as e:
        logging.error("Failed to complete model registration: %s", e)
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()