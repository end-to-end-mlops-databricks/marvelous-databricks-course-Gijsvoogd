import json
import logging

import mlflow


class ExperimentManager:
    def __init__(self, experiment_name, experiment_tags=None, tracking_uri="databricks"):
        """
        Initialize the ExperimentManager with the experiment name and tags.
        """
        self.logger = logging.getLogger(__name__)
        self.experiment_name = experiment_name
        self.experiment_tags = experiment_tags or {}
        self.tracking_uri = tracking_uri

        # Set the tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.logger.info(f"Set MLflow tracking URI to '{self.tracking_uri}'.")

    def set_experiment(self):
        """
        Set or create an MLflow experiment with the specified name and tags.
        """
        mlflow.set_experiment(experiment_name=self.experiment_name)
        mlflow.set_experiment_tags(self.experiment_tags)
        self.logger.info(f"Set MLflow experiment '{self.experiment_name}' with tags {self.experiment_tags}.")

    def start_run(self, run_name, run_tags=None, description=None, params=None, metrics=None):
        """
        Start an MLflow run and log parameters and metrics.
        """
        run_tags = run_tags or {}
        params = params or {}
        metrics = metrics or {}

        with mlflow.start_run(run_name=run_name, tags=run_tags, description=description) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            self.logger.info(f"Started MLflow run '{run_name}' with ID '{run.info.run_id}'.")
            self.logger.debug(f"Logged params: {params}")
            self.logger.debug(f"Logged metrics: {metrics}")
            return run.info.run_id

    def search_experiments(self, filter_string=None):
        """
        Search for MLflow experiments based on a filter string.
        """
        experiments = mlflow.search_experiments(filter_string=filter_string)
        self.logger.info(f"Found {len(experiments)} experiments with filter '{filter_string}'.")
        return experiments

    def save_experiment_info(self, experiments, filename="mlflow_experiment.json"):
        """
        Save experiment information to a JSON file.
        """
        experiment_dicts = [exp.__dict__ for exp in experiments]
        with open(filename, "w") as json_file:
            json.dump(experiment_dicts, json_file, indent=4)
        self.logger.info(f"Saved experiment information to '{filename}'.")

    def search_runs(self, experiment_names=None, filter_string=None):
        """
        Search for MLflow runs based on experiment names and filter string.
        """
        runs = mlflow.search_runs(experiment_names=experiment_names, filter_string=filter_string)
        self.logger.info(f"Found {len(runs)} runs with filter '{filter_string}' in experiments '{experiment_names}'.")
        return runs

    def get_run_info(self, run_id):
        """
        Get run information as a dictionary.
        """
        run_info = mlflow.get_run(run_id=run_id).to_dictionary()
        self.logger.info(f"Retrieved information for run ID '{run_id}'.")
        return run_info

    def save_run_info(self, run_info, filename="run_info.json"):
        """
        Save run information to a JSON file.
        """
        with open(filename, "w") as json_file:
            json.dump(run_info, json_file, indent=4)
        self.logger.info(f"Saved run information to '{filename}'.")

    def print_run_metrics(self, run_info):
        """
        Print the metrics from the run information.
        """
        metrics = run_info["data"]["metrics"]
        self.logger.info(f"Run metrics: {metrics}")
        print(metrics)

    def print_run_params(self, run_info):
        """
        Print the parameters from the run information.
        """
        params = run_info["data"]["params"]
        self.logger.info(f"Run parameters: {params}")
        print(params)
