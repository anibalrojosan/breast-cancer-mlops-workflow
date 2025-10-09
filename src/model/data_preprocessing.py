import pandas as pd

def drop_unnecessary_columns(df):
    """Drops the 'id' and 'Unnamed: 32' columns from the DataFrame."""
    return df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

def map_diagnosis_to_numerical(df):
    """Converts the 'diagnosis' column to numerical (M=1, B=0)."""
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def prepare_features_and_target(df):
    """Prepares features (X) and target (y) from the preprocessed DataFrame."""
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    return X, y