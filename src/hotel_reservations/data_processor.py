import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataProcessor:
    def __init__(self, file_path, config):
        self.df = self.load_data(file_path)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self):
        target = self.config.target
        self.df = self.df.dropna(subset=[target])

        # Combine numerical and categorical features (functional features will be added later)
        self.X = self.df[self.config.num_features + self.config.cat_features]
        self.y = self.df[target]

        # Define transformers for numeric and categorical features
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numeric_features = self.config.num_features
        categorical_features = self.config.cat_features

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def save_to_catalog(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        spark: SparkSession,
    ):
        """Save the train and test sets into Databricks tables."""
        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )

    def create_feature_functions(self, spark: SparkSession):
        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name

        # Create UDF for total_guests
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.total_guests(
            no_of_adults INT, no_of_children INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return no_of_adults + no_of_children
        $$
        """)

        # Create UDF for total_nights
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.total_nights(
            no_of_week_nights INT, no_of_weekend_nights INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return no_of_week_nights + no_of_weekend_nights
        $$
        """)

        # Create UDF for cancellation_rate
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.cancellation_rate(
            no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT)
        RETURNS FLOAT
        LANGUAGE PYTHON AS
        $$
        total_bookings = no_of_previous_cancellations + no_of_previous_bookings_not_canceled
        return 0.0 if total_bookings == 0 else no_of_previous_cancellations / total_bookings
        $$
        """)

        # Create UDF for special_request_ratio
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.special_request_ratio(
            no_of_special_requests INT, total_guests INT)
        RETURNS FLOAT
        LANGUAGE PYTHON AS
        $$
        return 0.0 if total_guests == 0 else no_of_special_requests / total_guests
        $$
        """)

        # Create UDF for family_indicator
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.family_indicator(
            no_of_children INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return 1 if no_of_children > 0 else 0
        $$
        """)

    def feature_engineering(self, train_set_spark, test_set_spark, spark: SparkSession):
        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name

        self.create_feature_functions(spark=spark)

        # Apply feature functions to the training set
        train_set_spark = (
            train_set_spark.withColumn(
                "total_guests", F.expr(f"{catalog_name}.{schema_name}.total_guests(no_of_adults, no_of_children)")
            )
            .withColumn(
                "total_nights",
                F.expr(f"{catalog_name}.{schema_name}.total_nights(no_of_week_nights, no_of_weekend_nights)"),
            )
            .withColumn(
                "cancellation_rate",
                F.expr(
                    f"{catalog_name}.{schema_name}.cancellation_rate(no_of_previous_cancellations, "
                    "no_of_previous_bookings_not_canceled)"
                ),
            )
            .withColumn(
                "special_request_ratio",
                F.expr(f"{catalog_name}.{schema_name}.special_request_ratio(no_of_special_requests, total_guests)"),
            )
            .withColumn("family_indicator", F.expr(f"{catalog_name}.{schema_name}.family_indicator(no_of_children)"))
        )

        # Apply the same feature functions to the test set
        test_set_spark = (
            test_set_spark.withColumn(
                "total_guests", F.expr(f"{catalog_name}.{schema_name}.total_guests(no_of_adults, no_of_children)")
            )
            .withColumn(
                "total_nights",
                F.expr(f"{catalog_name}.{schema_name}.total_nights(no_of_week_nights, no_of_weekend_nights)"),
            )
            .withColumn(
                "cancellation_rate",
                F.expr(
                    f"{catalog_name}.{schema_name}.cancellation_rate(no_of_previous_cancellations, "
                    "no_of_previous_bookings_not_canceled)"
                ),
            )
            .withColumn(
                "special_request_ratio",
                F.expr(f"{catalog_name}.{schema_name}.special_request_ratio(no_of_special_requests, total_guests)"),
            )
            .withColumn("family_indicator", F.expr(f"{catalog_name}.{schema_name}.family_indicator(no_of_children)"))
        )

        return train_set_spark, test_set_spark

    def save_feature_engineered_data(self, train_set_spark, test_set_spark, spark: SparkSession):
        """Save the feature-engineered train and test sets into Databricks tables."""
        # Overwrite the existing train and test tables with feature-engineered data
        train_set_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )
        test_set_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def get_X_y_datasets(self, train_set_spark, test_set_spark):
        """Read the train and test sets and return X and y datasets."""
        train_set = train_set_spark.toPandas()
        test_set = test_set_spark.toPandas()

        features = self.config.num_features + self.config.cat_features + self.config.func_features

        missing_features_train = [f for f in self.config.func_features if f not in train_set.columns]
        missing_features_test = [f for f in self.config.func_features if f not in test_set.columns]

        if missing_features_train:
            raise KeyError(f"Functional features missing from train_set: {missing_features_train}")
        if missing_features_test:
            raise KeyError(f"Functional features missing from test_set: {missing_features_test}")

        X_train = train_set[features]
        y_train = train_set[self.config.target]

        X_test = test_set[features]
        y_test = test_set[self.config.target]

        return X_train, y_train, X_test, y_test
