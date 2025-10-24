from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .data_preprocessing import drop_unnecessary_columns, select_features

def create_breast_cancer_pipeline():
    """Creates and returns an updated scikit-learn pipeline for breast cancer prediction."""

    preprocessing_pipeline = Pipeline(
        [
            ("drop_unnecessary_cols", FunctionTransformer(drop_unnecessary_columns)),
            ("select_features", FunctionTransformer(select_features)),
            ("scaler", StandardScaler()),
        ]
    )
    
    full_pipeline = Pipeline(
        [
            ("preprocessor", preprocessing_pipeline),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    return full_pipeline