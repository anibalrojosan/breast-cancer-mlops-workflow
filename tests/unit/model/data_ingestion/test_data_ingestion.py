import pandas as pd
from src.model.data_ingestion import load_raw_data
import os
import pytest

@pytest.fixture
def temp_csv_file():
    """Fixture to create a temporary directory and a dummy CSV file."""
    temp_dir = 'tests/temp_data'
    os.makedirs(temp_dir, exist_ok=True)
    
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    file_path = os.path.join(temp_dir, 'test_data.csv')
    df.to_csv(file_path, index=False)
    
    yield file_path, df
    
    os.remove(file_path)
    os.rmdir(temp_dir)

@pytest.fixture
def temp_empty_csv_file():
    """Fixture to create a temporary empty CSV file."""
    temp_dir = 'tests/temp_data'
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, 'empty_data.csv')
    with open(file_path, 'w') as f:
        pass
    
    yield file_path
    
    os.remove(file_path)
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)

def test_load_raw_data_success(temp_csv_file):
    """Test loading data from a CSV file successfully."""
    file_path, original_df = temp_csv_file
    loaded_df = load_raw_data(file_path)
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.equals(original_df)

def test_load_raw_data_file_not_found():
    """Test FileNotFoundError for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_raw_data('non_existent_file.csv')

def test_load_raw_data_empty_file(temp_empty_csv_file):
    """Test loading data from an empty CSV file."""
    file_path = temp_empty_csv_file
    loaded_df = load_raw_data(file_path)
    assert loaded_df.empty
