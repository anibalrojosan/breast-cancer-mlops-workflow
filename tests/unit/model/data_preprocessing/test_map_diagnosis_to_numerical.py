
from src.model.data_preprocessing import map_diagnosis_to_numerical


def test_map_diagnosis_to_numerical_success(sample_data_1):
    """Test that M and B are correctly mapped to 1 and 0"""
    df = sample_data_1.copy()
    result = map_diagnosis_to_numerical(df)

    # Assert specific mappings
    assert result.iloc[0]["diagnosis"] == 1
    assert result.iloc[1]["diagnosis"] == 0
    assert result.iloc[2]["diagnosis"] == 1

    # Assert that only 0s and 1s are present in the diagnosis column
    assert sorted(result["diagnosis"].unique().tolist()) == [0, 1]

    # Assert that the other columns remain the same
    assert result["feature_1"].equals(sample_data_1["feature_1"])
    assert result["feature_2"].equals(sample_data_1["feature_2"])
    assert result["feature_3"].equals(sample_data_1["feature_3"])


def test_map_diagnosis_to_numerical_missing(sample_data_2):
    """Test the behavior when the diagnosis column is missing in the dataframe.
    The diagnosis column should be created and filled with NaN."""
    df = sample_data_2.copy()
    result = map_diagnosis_to_numerical(df)

    # Assert that diagnosis column it's created
    assert "diagnosis" in result.columns

    # Assert that the 'diagnosis' column is filled with NaN
    assert result["diagnosis"].isnull().all()

    # Assert that the other columns remain the same
    assert result["feature_1"].equals(sample_data_2["feature_1"])
    assert result["feature_2"].equals(sample_data_2["feature_2"])
    assert result["feature_3"].equals(sample_data_2["feature_3"])
