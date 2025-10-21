from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from .data_preprocessing import drop_unnecessary_columns


def create_breast_cancer_pipeline():
    """Creates and returns a scikit-learn pipeline for breast cancer prediction."""

    # Define preprocessing steps
    preprocessing_pipeline = Pipeline(
        [
            (
                "drop_cols",
                FunctionTransformer(drop_unnecessary_columns, validate=False),
            ),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Combine preprocessing and model into a full pipeline
    full_pipeline = Pipeline(
        [
            ("preprocessor", preprocessing_pipeline),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    return full_pipeline
