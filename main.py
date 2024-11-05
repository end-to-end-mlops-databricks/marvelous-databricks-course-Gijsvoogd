import logging

import yaml
from pyspark.sql import SparkSession

from hotel_reservations.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_file_path):
    logger.info(f"Loading configuration from {config_file_path}")
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def process_data(file_path, config, spark):
    logger.info(f"Loading and processing data from {file_path}")
    data_processor = DataProcessor(file_path, config)
    data_processor.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor.split_data()
    logger.info("Data processing completed")
    data_processor.save_to_catalog(X_train, X_test, y_train, y_test, spark)

    return X_train, X_test, y_train, y_test


def main():
    file_path = "/Volumes/cmo_talive/mlops/data/Hotel Reservations.csv"
    config_file_path = "project_config.yml"
    config = load_config(config_file_path)

    spark = SparkSession.builder.getOrCreate()

    X_train, X_test, y_train, y_test = process_data(file_path, config, spark)
    logger.info("Preview of the training data:")
    logger.info(f"X_train:\n{X_train.head()}")
    logger.info(f"y_train:\n{y_train.head()}")


if __name__ == "__main__":
    main()
