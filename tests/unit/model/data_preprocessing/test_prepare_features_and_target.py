import pytest

from src.model.data_preprocessing import prepare_features_and_target


def test_prepare_features_and_target_success(sample_data_3):
    """Test that the function correctly prepares the features and target"""
    df = sample_data_3.copy()
    X, y = prepare_features_and_target(df)

    # 'diagnosis' column should not be included in the features
    assert "diagnosis" not in X.columns

    # X should contain the other columns
    assert set(X.columns) == {"feature_1", "feature_2", "feature_3"}

    # y should contain the 'diagnosis' column
    assert y.equals(df["diagnosis"])

    # Check the shapes
    assert X.shape == (3, 3)
    assert y.shape == (3,)


def test_prepare_features_and_target_missing_diagnosis(sample_data_2):
    """Test the behavior when the diagnosis column is missing in the dataframe.
    It should raise a KeyError."""
    df = sample_data_2.copy()

    with pytest.raises(KeyError, match="\\['diagnosis'\\] not found in axis"):
        prepare_features_and_target(df)
