import logging

from hotel_reservations.config import ProjectConfig
from hotel_reservations.model_trainer import ModelTrainer


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Load configuration
    config_file_path = "project_config.yml"
    config = ProjectConfig.from_yaml(config_file_path)
    logger.info("Loaded project configuration.")

    # Initialize the ModelTrainer
    model_trainer = ModelTrainer(config=config)

    # Run the model training and logging process
    model_trainer.run()
    logger.info("Model training and registration process completed.")


if __name__ == "__main__":
    main()
