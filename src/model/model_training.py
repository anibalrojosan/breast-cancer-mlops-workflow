import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

from .data_ingestion import load_raw_data
from .data_preprocessing import map_diagnosis_to_numerical, prepare_features_and_target
from .pipeline_utils import create_breast_cancer_pipeline

def train_and_save_pipeline(data_path='data/data.csv', model_path='models/model.joblib'):
    """Orchestrates the training process: loads data, preprocesses, trains, and saves the pipeline."""
    # Load the raw dataset
    df_raw = load_raw_data(data_path)

    # Apply diagnosis mapping before splitting features and target
    df_mapped = map_diagnosis_to_numerical(df_raw.copy()) # Use a copy to avoid modifying original df_raw if it's used elsewhere

    # Prepare features (X) and target (y)
    X, y = prepare_features_and_target(df_mapped)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the pipeline
    pipeline = create_breast_cancer_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate the pipeline
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Pipeline Accuracy: {accuracy:.4f}")

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained pipeline using joblib
    joblib.dump(pipeline, model_path)
    print(f"Trained pipeline saved to {model_path}")

if __name__ == "__main__":
    train_and_save_pipeline()
