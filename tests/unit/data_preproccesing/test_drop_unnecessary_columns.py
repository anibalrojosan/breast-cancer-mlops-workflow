import pandas as pd
import pytest

from src.model.data_preprocessing import drop_unnecessary_columns

@pytest.fixture
def sample_data_1():
    'Provides a sample dataframe for testing preprocessing functions'
    data = {
        'id': [1, 2, 3],
        'diagnosis': ['M', 'B', 'M'],
        'feature_1': [0.1, 0.2, 0.3],
        'feature_2': [0.4, 0.5, 0.6],
        'feature_3': [0.7, 0.8, 0.9],
        'Unnamed: 32': [0.1, 0.2, 0.5],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_2():
    'Provides a sample dataframe for testing preprocessing functions'
    data = {
        'diagnosis': ['M', 'B', 'M'],
        'feature_1': [0.1, 0.2, 0.3],
        'feature_2': [0.4, 0.5, 0.6],
        'feature_3': [0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_data():
    'Provides an empty dataframe for testing preprocessing functions'
    return pd.DataFrame(columns=['id', 'diagnosis', 'feature_1', 'feature_2', 'feature_3', 'Unnamed: 32'])


# Test drop_unnecessary_columns function
def test_drop_unnecessary_columns_success(sample_data_1):
    original_num_rows = sample_data_1.shape[0]
    expected_columns = [col for col in sample_data_1.columns if col not in ['id', 'Unnamed: 32']]

    result = drop_unnecessary_columns(sample_data_1)

    # Assert that the dropped columns are indeed NOT in the result
    assert 'id' not in result.columns
    assert 'Unnamed: 32' not in result.columns

    # Assert that the remaining columns are exactly as expected
    assert sorted(result.columns.tolist()) == sorted(expected_columns)

    # Assert that the number of rows remains the same, and columns decreased by 2
    assert result.shape[0] == original_num_rows
    assert result.shape[1] == (sample_data_1.shape[1] - 2) # ('id', 'Unnamed: 32') were dropped

def test_drop_unnecessary_columns_not_present(sample_data_2):
    original_columns = sample_data_2.columns.tolist()
    original_shape = sample_data_2.shape

    result = drop_unnecessary_columns(sample_data_2)
    assert 'id' not in result.columns
    assert 'Unnamed: 32' not in result.columns
    assert result.columns.tolist() == original_columns
    assert result.shape == original_shape

def test_drop_unnecessary_columns_empty(empty_data):
    result = drop_unnecessary_columns(empty_data)
    assert 'id' not in result.columns
    assert 'Unnamed: 32' not in result.columns