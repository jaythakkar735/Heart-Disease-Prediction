"""
Machine Learning Pipeline definition for Heart Disease classification.
Ensures zero data leakage: Imputation and scaling are fit strictly on training folds.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator
from .data_loader import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Build preprocessing ColumnTransformer with separate numeric and categorical flows."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_full_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Combine preprocessor and given classifier into a single executable pipeline."""
    preprocessor = create_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline
