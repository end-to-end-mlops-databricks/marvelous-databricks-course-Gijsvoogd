import logging

import mlflow
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservations.config import ProjectConfig


class ModelTrainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.spark = SparkSession.builder.getOrCreate()
        self.logger = logging.getLogger(__name__)
        self.experiment_name = "/Shared/hotel-reservations-basic"
        self.model_name = f"{config.catalog_name}.{config.schema_name}.hotel_reservations_model"
        self.mlflow_tracking_uri = "databricks"
        self.mlflow_registry_uri = "databricks-uc"
        self.git_sha = "ffa63b430205ff7"
        self.branch = "week_2"
        self.run_id = None

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_registry_uri(self.mlflow_registry_uri)

        self.train_set = None
        self.test_set = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.pipeline = None
        self.model_metrics = {}

    def load_data(self):
        """Load training and testing sets from Databricks tables."""
        self.logger.info("Loading training and testing data from Databricks tables...")
        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name

        self.train_set = self.spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
        self.test_set = self.spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
        self.logger.info("Data loading completed.")

        features = self.config.num_features + self.config.cat_features + self.config.func_features

        # Verify that functional features are present
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

        # Verify data types
        self.logger.debug("Data types of X_train:")
        self.logger.debug(self.X_train.dtypes)

    def build_pipeline(self):
        """Define the preprocessing and modeling pipeline."""
        self.logger.info("Building the preprocessing and modeling pipeline...")

        # Determine which func_features are numeric and which are categorical
        numeric_func_features = ["total_guests", "total_nights", "cancellation_rate", "special_request_ratio"]
        categorical_func_features = ["family_indicator"]

        numeric_features = self.config.num_features + numeric_func_features
        categorical_features = self.config.cat_features + categorical_func_features

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LGBMClassifier(**self.config.parameters)),
            ]
        )
        self.logger.info("Pipeline construction completed.")

    def train_model(self):
        """Train the model and evaluate its performance."""
        self.logger.info("Starting model training...")
        self.pipeline.fit(self.X_train, self.y_train)
        self.logger.info("Model training completed.")

        y_pred = self.pipeline.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

        self.model_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        self.logger.info(f"Model evaluation metrics: {self.model_metrics}")

    def log_and_register_model(self):
        """Log parameters, metrics, and register the model with MLflow."""
        self.logger.info("Logging parameters, metrics, and model to MLflow...")

        mlflow.set_experiment(experiment_name=self.experiment_name)

        with mlflow.start_run(
            tags={
                "git_sha": self.git_sha,
                "branch": self.branch,
            }
        ) as run:
            self.run_id = run.info.run_id

            mlflow.log_param("model_type", "LGBMClassifier with preprocessing")
            mlflow.log_params(self.config.parameters)
            mlflow.log_metrics(self.model_metrics)

            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lgbm-pipeline-model",
                signature=signature,
            )

            dataset = mlflow.data.from_pandas(
                df=self.train_set,
                source=f"{self.config.catalog_name}.{self.config.schema_name}.train_set",
                name="training_dataset",
            )
            mlflow.log_input(dataset, context="training")

            self.logger.info(f"Logged model with run ID '{self.run_id}'.")

        self.logger.info("Registering the model in MLflow Model Registry...")
        model_uri = f"runs:/{self.run_id}/lgbm-pipeline-model"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
            tags={"git_sha": self.git_sha},
        )
        self.logger.info(f"Model registered with version '{model_version.version}'.")

    def run(self):
        """Execute the entire model training and logging process."""
        self.load_data()
        self.build_pipeline()
        self.train_model()
        self.log_and_register_model()
