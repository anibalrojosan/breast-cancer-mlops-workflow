import pandas as pd
import pytest


@pytest.fixture
def sample_data_1():
    """Provides a sample dataframe for testing preprocessing functions"""
    data = {
        "id": [1, 2, 3],
        "diagnosis": ["M", "B", "M"],
        "feature_1": [0.1, 0.2, 0.3],
        "feature_2": [0.4, 0.5, 0.6],
        "feature_3": [0.7, 0.8, 0.9],
        "Unnamed: 32": [0.1, 0.2, 0.5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_2():
    """Provides a sample dataframe for testing preprocessing functions"""
    data = {
        #        'diagnosis': ['M', 'B', 'M'],
        "feature_1": [0.1, 0.2, 0.3],
        "feature_2": [0.4, 0.5, 0.6],
        "feature_3": [0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_data():
    """Provides an empty dataframe for testing preprocessing functions"""
    return pd.DataFrame(
        columns=[
            "id",
            "diagnosis",
            "feature_1",
            "feature_2",
            "feature_3",
            "Unnamed: 32",
        ]
    )


@pytest.fixture
def sample_data_3():
    """Provides a sample dataframe for testing prepare_features_and_target function"""
    data = {
        "diagnosis": [1, 0, 1],
        "feature_1": [0.1, 0.2, 0.3],
        "feature_2": [0.4, 0.5, 0.6],
        "feature_3": [0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_data_path(tmp_path):
    """
    Creates a temporary dummy CSV file for testing the train_and_save_pipeline function.
    The values of each feature are not important, they are just dummy values to test the function
    behavior (orchestration of the pipeline steps).
    """
    data = {
        "id": range(10),
        "diagnosis": ["M", "B", "M", "B", "M", "B", "M", "B", "M", "B"],
        "radius_mean": [10.0 + i for i in range(10)],
        "texture_mean": [15.0 + i for i in range(10)],
        "perimeter_mean": [70.0 + i for i in range(10)],
        "area_mean": [300.0 + i for i in range(10)],
        "smoothness_mean": [0.1 + i * 0.01 for i in range(10)],
        "compactness_mean": [0.05 + i * 0.005 for i in range(10)],
        "concavity_mean": [0.03 + i * 0.003 for i in range(10)],
        "concave points_mean": [0.01 + i * 0.001 for i in range(10)],
        "symmetry_mean": [0.15 + i * 0.005 for i in range(10)],
        "fractal_dimension_mean": [0.05 + i * 0.001 for i in range(10)],
    }
    for col_suffix in ["se", "worst"]:
        for col_prefix in [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave points",
            "symmetry",
            "fractal_dimension",
        ]:
            data[f"{col_prefix}_{col_suffix}"] = [0.1 + j * 0.001 for j in range(10)]

    df = pd.DataFrame(data)
    dummy_file = tmp_path / "dummy_data.csv"
    df.to_csv(dummy_file, index=False)
    return str(dummy_file)
