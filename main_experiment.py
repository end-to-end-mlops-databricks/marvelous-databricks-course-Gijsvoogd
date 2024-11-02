import logging

from src.hotel_reservations.experiment import ExperimentManager


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Initialize the ExperimentManager
    experiment_name = "/Shared/hotel-reservations-basic"
    experiment_tags = {"repository_name": "hotel-reservations"}
    exp_manager = ExperimentManager(experiment_name=experiment_name, experiment_tags=experiment_tags)

    # Set or create the experiment
    exp_manager.set_experiment()

    # Search for experiments with a specific tag
    experiments = exp_manager.search_experiments(filter_string="tags.repository_name='hotel-reservations'")
    logger.info(f"Experiments found: {experiments}")

    # Save experiment information to a JSON file
    exp_manager.save_experiment_info(experiments)

    # Start a new MLflow run
    run_name = "demo-run"
    run_tags = {"git_sha": "ffa63b430205ff7", "branch": "main"}
    description = "Demo run"
    params = {"type": "demo"}
    metrics = {"metric1": 1.0, "metric2": 2.0}
    run_id = exp_manager.start_run(
        run_name=run_name, run_tags=run_tags, description=description, params=params, metrics=metrics
    )

    # Search for runs with a specific tag
    runs = exp_manager.search_runs(experiment_names=[experiment_name], filter_string="tags.git_sha='ffa63b430205ff7'")
    logger.info(f"Runs found: {runs}")

    # Get run information
    run_info = exp_manager.get_run_info(run_id=run_id)

    # Save run information to a JSON file
    exp_manager.save_run_info(run_info)

    # Print run metrics and parameters
    exp_manager.print_run_metrics(run_info)
    exp_manager.print_run_params(run_info)


if __name__ == "__main__":
    main()
