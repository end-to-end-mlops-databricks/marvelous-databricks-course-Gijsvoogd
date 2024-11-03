import logging

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig


class HotelReservationsModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            # Make predictions using the loaded model
            predictions = self.model.predict(model_input)
            # You can add any post-processing logic here if needed
            return pd.DataFrame({"Prediction": predictions})
        else:
            raise ValueError("Input must be a pandas DataFrame.")


class CustomModelTrainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.spark = SparkSession.builder.getOrCreate()
        self.logger = logging.getLogger(__name__)
        self.experiment_name = "/Shared/hotel-reservations-custom-model"
        self.model_name = f"{config.catalog_name}.{config.schema_name}.hotel_reservations_custom_model"
        self.mlflow_tracking_uri = "databricks"
        self.mlflow_registry_uri = "databricks-uc"
        self.git_sha = "your_git_sha"
        self.branch = "week_2"
        self.run_id = None

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_registry_uri(self.mlflow_registry_uri)
        self.client = MlflowClient()

        self.train_set = None
        self.test_set = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def load_data(self):
        """Load training and testing sets from Databricks tables."""
        self.logger.info("Loading training and testing data from Databricks tables...")
        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name

        self.train_set = self.spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
        self.test_set = self.spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
        self.logger.info("Data loading completed.")

        features = self.config.num_features + self.config.cat_features + self.config.func_features

        missing_features_train = [f for f in self.config.func_features if f not in self.train_set.columns]
        missing_features_test = [f for f in self.config.func_features if f not in self.test_set.columns]

        if missing_features_train:
            raise KeyError(f"Functional features missing from train_set: {missing_features_train}")
        if missing_features_test:
            raise KeyError(f"Functional features missing from test_set: {missing_features_test}")

        self.X_train = self.train_set[features]
        self.y_train = self.train_set[self.config.target]

        self.X_test = self.test_set[features]
        self.y_test = self.test_set[self.config.target]

        self.logger.debug("Data types of X_train:")
        self.logger.debug(self.X_train.dtypes)

    def load_trained_model(self):
        """Load the trained model from MLflow."""
        self.logger.info("Loading the trained model from MLflow...")
        experiment_name = "/Shared/hotel-reservations-basic"

        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

        runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
        run_id = runs.iloc[0]["run_id"]
        self.logger.info(f"Loading model from run ID: {run_id}")

        model_uri = f"runs:/{run_id}/lgbm-pipeline-model"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.logger.info("Model loading completed.")

    def create_custom_model(self):
        """Wrap the loaded model in a custom PythonModel."""
        self.logger.info("Wrapping the model in a custom PythonModel...")
        self.wrapped_model = HotelReservationsModelWrapper(self.model)
        self.logger.info("Model wrapping completed.")

    def log_and_register_custom_model(self):
        """Log the custom model to MLflow and register it."""
        self.logger.info("Logging and registering the custom model to MLflow...")
        mlflow.set_experiment(experiment_name=self.experiment_name)

        with mlflow.start_run(
            tags={
                "git_sha": self.git_sha,
                "branch": self.branch,
            }
        ) as run:
            self.run_id = run.info.run_id

            example_input = self.X_test.iloc[0:1]
            example_prediction = self.wrapped_model.predict(context=None, model_input=example_input)

            signature = infer_signature(model_input=example_input, model_output=example_prediction)

            dataset = mlflow.data.from_pandas(
                df=self.train_set,
                source=f"{self.config.catalog_name}.{self.config.schema_name}.train_set",
                name="training_dataset",
            )
            mlflow.log_input(dataset, context="training")

            # Define the conda environment for the model (if needed)
            conda_env = _mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=[
                    # List any additional pip dependencies here
                    # For example: "scikit-learn==1.0.2"
                ],
                additional_conda_channels=None,
            )

            mlflow.pyfunc.log_model(
                python_model=self.wrapped_model,
                artifact_path="hotel-reservations-custom-model",
                signature=signature,
                input_example=example_input,
                conda_env=conda_env,
            )
            self.logger.info(f"Logged custom model with run ID '{self.run_id}'.")

        # Register the model
        self.logger.info("Registering the custom model in MLflow Model Registry...")
        model_uri = f"runs:/{self.run_id}/hotel-reservations-custom-model"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
            tags={"git_sha": self.git_sha},
        )
        self.logger.info(f"Custom model registered with version '{model_version.version}'.")

        # Set a model version alias
        model_version_alias = "what_a_model"
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=model_version_alias,
            version=model_version.version,
        )
        self.logger.info(f"Set alias '{model_version_alias}' for model version '{model_version.version}'.")

    def run(self):
        """Execute the entire custom model creation and registration process."""
        self.load_data()
        self.load_trained_model()
        self.create_custom_model()
        self.log_and_register_custom_model()
