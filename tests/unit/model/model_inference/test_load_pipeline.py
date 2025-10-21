import joblib
import pytest
from src.model.model_inference import load_pipeline
from sklearn.preprocessing import StandardScaler


def test_load_pipeline_success(tmp_path):
    # Create a simple scikit-learn object to act as a dummy pipeline
    dummy_pipeline = StandardScaler()

    # Define a temporary path for the dummy model
    dummy_model_path = tmp_path / "dummy_model.joblib"

    # Save the dummy pipeline to the temporary path
    joblib.dump(dummy_pipeline, dummy_model_path)

    # Call load_pipeline with the temporary path
    loaded_pipeline = load_pipeline(model_path=dummy_model_path)

    # Assert that the pipeline was loaded and is of the expected type
    assert loaded_pipeline is not None
    assert isinstance(loaded_pipeline, StandardScaler)


def test_load_pipeline_file_not_found():
    # If both this conditions are met, the test will pass
    # 1. FileNotFoundError is raised
    # 2. The error message contains "Model pipeline not found"
    with pytest.raises(FileNotFoundError, match="Model pipeline not found"):
        load_pipeline(model_path="non_existent_path/model.joblib")
