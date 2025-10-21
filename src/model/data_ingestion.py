import pandas as pd


def load_raw_data(data_path="data/data.csv"):
    """Loads the raw dataset from the specified path."""
    try:
        df = pd.read_csv(data_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df
