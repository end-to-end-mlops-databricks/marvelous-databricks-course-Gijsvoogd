import logging

from pyspark.sql import SparkSession

from src.hotel_reservations.config import ProjectConfig
from src.hotel_reservations.data_processor import DataProcessor
from src.hotel_reservations.model_trainer import ModelTrainer


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    spark = SparkSession.builder.appName("HotelReservations").getOrCreate()

    config_file_path = "project_config.yml"
    config = ProjectConfig.from_yaml(config_file_path)
    logger.info("Loaded project configuration.")

    data_file_path = "/Volumes/cmo_talive/mlops/data/Hotel Reservations.csv"
    data_processor = DataProcessor(file_path=data_file_path, config=config)

    data_processor.preprocess_data()
    logger.info("Data preprocessing completed.")

    X_train, X_test, y_train, y_test = data_processor.split_data()
    logger.info("Data splitting completed.")

    data_processor.save_to_catalog(X_train, X_test, y_train, y_test, spark)
    logger.info("Training and testing data saved to catalog.")

    train_set_spark = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
    test_set_spark = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    logger.info("Training and testing data loaded from catalog.")

    train_set_spark, test_set_spark = data_processor.feature_engineering(train_set_spark, test_set_spark, spark)
    logger.info("Feature engineering completed.")

    data_processor.save_feature_engineered_data(train_set_spark, test_set_spark, spark)
    logger.info("Feature-engineered data saved back to catalog.")

    model_trainer = ModelTrainer(config=config)

    model_trainer.run()
    logger.info("Model training and registration process completed.")


if __name__ == "__main__":
    main()
