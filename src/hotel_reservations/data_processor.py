import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        target = self.config['target']
        self.df = self.df.dropna(subset=[target])

        self.X = self.df[self.config['num_features'] + self.config['cat_features']]
        self.y = self.df[target]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['num_features']),
                ('cat', categorical_transformer, self.config['cat_features'])
            ])

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
