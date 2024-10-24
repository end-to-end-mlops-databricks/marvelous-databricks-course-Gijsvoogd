import logging

import yaml

from hotel_reservations.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_file_path):
    logger.info(f"Loading configuration from {config_file_path}")
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def process_data(file_path, config):
    logger.info(f"Loading and processing data from {file_path}")
    data_processor = DataProcessor(file_path, config)
    data_processor.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor.split_data()
    logger.info("Data processing completed")
    return X_train, X_test, y_train, y_test


def main():
    file_path = "/Volumes/cmo_talive/mlops/data/Hotel Reservations.csv"
    config_file_path = "project_config.yml"
    config = load_config(config_file_path)
    X_train, X_test, y_train, y_test = process_data(file_path, config)
    logger.info("Preview of the training data:")
    logger.info(f"X_train:\n{X_train.head()}")
    logger.info(f"y_train:\n{y_train.head()}")


if __name__ == "__main__":
    main()
