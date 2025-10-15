import pytest
import pandas as pd
import joblib
from pathlib import Path

from src.model.model_training import train_and_save_pipeline

def test_train_and_save_pipeline_saves_model(dummy_data_path, tmp_path):
    """
    Tests if the train_and_save_pipeline function successfully trains and saves a model.
    The dummy_data_path fixture from conftest.py provides the input data.
    """
    model_dir = tmp_path/"temp_models"
    model_path = model_dir/"test_model.joblib"

    # Ensure the directory does not exist initially to test its creation
    assert not model_dir.exists()

    train_and_save_pipeline(data_path=dummy_data_path, model_path=str(model_path))

    # Assert that the model directory and file were created
    assert model_dir.is_dir()
    assert model_path.is_file()

    # Load the model and perform a sanity check (it should have a predict method)
    loaded_pipeline = joblib.load(model_path)
    assert loaded_pipeline is not None
    assert hasattr(loaded_pipeline, 'predict')