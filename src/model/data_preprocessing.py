import pandas as pd

def drop_unnecessary_columns(df):
    """Drops the 'id' and 'Unnamed: 32' columns from the DataFrame."""
    return df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

def map_diagnosis_to_numerical(df):
    """Converts the 'diagnosis' column to numerical (M=1, B=0)."""
    # Use .get() to retrieve the 'diagnosis' Series.If 'diagnosis' doesn't exist, 
    # .get() returns the default value, which we set to an empty Series with 
    # the correct index. This ensures .map() is always called on a Series, 
    # preventing KeyError.
    diagnosis_series = df.get('diagnosis', pd.Series(index=df.index, dtype='object'))
    df['diagnosis'] = diagnosis_series.map({'M': 1, 'B': 0})
    return df

def prepare_features_and_target(df):
    """Prepares features (X) and target (y) from the preprocessed DataFrame."""
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    return X, y