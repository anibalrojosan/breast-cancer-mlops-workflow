import pandas as pd
import pytest

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
#        'diagnosis': ['M', 'B', 'M'],
        'feature_1': [0.1, 0.2, 0.3],
        'feature_2': [0.4, 0.5, 0.6],
        'feature_3': [0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_data():
    'Provides an empty dataframe for testing preprocessing functions'
    return pd.DataFrame(columns=['id', 'diagnosis', 'feature_1', 'feature_2', 'feature_3', 'Unnamed: 32'])