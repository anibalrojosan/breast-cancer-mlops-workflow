import pandas as pd

from src.model.data_preprocessing import drop_unnecessary_columns

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